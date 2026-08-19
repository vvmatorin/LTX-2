"""Bake a LoRA checkpoint into the base model weights and write a new base checkpoint.

Training a pose LoRA warm-started from a concept/pretrain LoRA means both live in the
*same* adapter tensors, so pose training is free to erode the pretrain knowledge. Merging
the pretrain LoRA into the base instead freezes that knowledge in the weights and lets the
next run learn a clean delta on top -- which also frees ``lora.rank``/``lora.alpha``, since
a zero-init adapter has no shape to match against a loaded checkpoint.

Key layouts (see ``LTXV_MODEL_COMFY_RENAMING_MAP`` and ``LtxvTrainer._save_checkpoint``):

    base ckpt   ``model.diffusion_model.transformer_blocks.0.attn1.to_q.weight``
    LoRA ckpt   ``diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight``

so the base key is the LoRA module path with ``model.`` prepended and ``.weight`` appended.
The base file is a *combined* checkpoint (transformer + VAEs + vocoder); everything outside
the merged modules is copied through untouched.

For each adapted Linear, PEFT computes ``W_eff = W + (B @ A) * (alpha / rank)``. The delta is
accumulated in float32 and the result cast exactly once, back to the weight's original dtype
(bf16 for LTX-2 Linear weights). By default every other tensor keeps its original dtype: the
base checkpoint deliberately stores the ~300 tiny AdaLN ``scale_shift_table`` parameters in
fp32, and blanket-casting them to bf16 would lose precision for zero benefit. Pass ``--dtype``
only if you explicitly want to force the whole file to one precision.

Note: this loads the whole base checkpoint into RAM (tens of GB for LTX-2). Run it on a box
with headroom, not a laptop.

Example:
    python scripts/merge_lora_to_base.py \\
        --base   /mnt/disks/shared/models/ltx2.safetensors \\
        --lora   /mnt/disks/shared/outputs/pretrain/v2/checkpoints/lora_weights_step_15400_video_only.safetensors \\
        --output /mnt/disks/shared/models/ltx2_pretrain_v2_15400.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# "keep" preserves each tensor's original dtype (merged weights included) instead of
# forcing the whole checkpoint to one precision.
DTYPES: dict[str, torch.dtype | None] = {
    "keep": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

# LoRA keys carry the ComfyUI "diffusion_model." prefix; base keys nest that under "model.".
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def collect_lora_pairs(lora_sd: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Group a LoRA state dict into ``module_path -> (A, B)``, erroring on anything unexpected."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    unknown: list[str] = []
    for key, tensor in lora_sd.items():
        for suffix in LORA_SUFFIXES:
            if key.endswith(suffix):
                path = key[: -len(suffix)]
                pairs.setdefault(path, {})["A" if "lora_A" in suffix else "B"] = tensor
                break
        else:
            unknown.append(key)

    if unknown:
        raise SystemExit(
            f"LoRA checkpoint has {len(unknown)} key(s) that are not lora_A/lora_B, e.g. {unknown[:5]}.\n"
            "DoRA/LoKr checkpoints are not supported by this script."
        )

    incomplete = sorted(p for p, v in pairs.items() if set(v) != {"A", "B"})
    if incomplete:
        raise SystemExit(f"Missing the paired A/B tensor for {len(incomplete)} module(s), e.g. {incomplete[:5]}")

    return {p: (v["A"], v["B"]) for p, v in pairs.items()}


def base_key_for(module_path: str) -> str:
    """Map a LoRA module path to its weight key in the combined base checkpoint."""
    return f"model.{module_path}.weight"


def merge(
    base_path: Path,
    lora_path: Path,
    output: Path,
    alpha: float | None,
    dtype: torch.dtype | None,
    strength: float,
) -> None:
    lora_pairs = collect_lora_pairs(load_file(str(lora_path)))
    if not lora_pairs:
        raise SystemExit("No LoRA tensors found in the checkpoint.")

    print(f"Loading base checkpoint {base_path} (this needs tens of GB of RAM)...")
    base_sd = load_file(str(base_path))

    # Fail before touching anything if the key mapping is wrong, rather than silently
    # merging a fraction of the adapter and producing a subtly broken model.
    missing = [p for p in lora_pairs if base_key_for(p) not in base_sd]
    if missing:
        sample_base = sorted(k for k in base_sd if "transformer_blocks" in k)[:3]
        raise SystemExit(
            f"{len(missing)} of {len(lora_pairs)} LoRA modules have no matching base weight.\n"
            f"  first missing LoRA path: {missing[0]}\n"
            f"  tried base key:          {base_key_for(missing[0])}\n"
            f"  example base keys:       {sample_base}\n"
            "The key convention may have changed -- check LTXV_MODEL_COMFY_RENAMING_MAP."
        )

    merged = 0
    ranks: set[int] = set()
    ratios: list[float] = []
    merged_dtypes: set[torch.dtype] = set()
    for path, (a, b) in sorted(lora_pairs.items()):
        key = base_key_for(path)
        weight = base_sd[key]
        rank = a.shape[0]
        ranks.add(rank)
        scaling = (alpha if alpha is not None else float(rank)) / rank * strength

        # float32 for the low-rank product and the sum: the LoRA tensors are stored in
        # bf16, so a bf16 matmul would accumulate rank-many terms at ~0.4% per-element
        # error before the add. Rounding happens exactly once, on the final sum.
        weight_fp32 = weight.to(torch.float32)
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * scaling
        if delta.shape != weight.shape:
            raise SystemExit(
                f"Shape mismatch for {key}: base {tuple(weight.shape)} vs LoRA delta {tuple(delta.shape)}"
            )
        w_norm = weight_fp32.norm().item()
        if w_norm > 0:
            ratios.append(delta.norm().item() / w_norm)
        out_dtype = dtype if dtype is not None else weight.dtype
        merged_dtypes.add(out_dtype)
        base_sd[key] = (weight_fp32 + delta).to(out_dtype)
        merged += 1

    # Only force a uniform precision when explicitly requested. The default ("keep")
    # preserves original dtypes: LTX-2 base checkpoints store the AdaLN scale_shift_table
    # parameters in fp32 on purpose, and downcasting them to bf16 would discard precision
    # on tensors the LoRA never touched.
    recast = 0
    if dtype is not None:
        for key, tensor in base_sd.items():
            if tensor.is_floating_point() and tensor.dtype != dtype:
                base_sd[key] = tensor.to(dtype)
                recast += 1

    with safe_open(str(base_path), framework="pt") as f:
        metadata = f.metadata() or {}

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output}...")
    save_file(base_sd, str(output), metadata=metadata)

    print("Merged base checkpoint written to", output)
    print(f"  LoRA modules merged: {merged}")
    print(f"  LoRA rank(s):        {sorted(ranks)}  (alpha={'rank' if alpha is None else alpha}, strength={strength})")
    if dtype is None:
        print("  dtypes:              kept original per-tensor dtypes (merged weights included)")
    else:
        print(f"  tensors recast:      {recast} (plus {merged} merged) -> {str(dtype).replace('torch.', '')}")
    print(f"  total tensors:       {len(base_sd)}")
    report_delta_scale(ratios, merged_dtypes)


# bf16 keeps ~8 mantissa bits, so a delta below roughly this fraction of the weight norm is
# largely destroyed by the final rounding regardless of how precisely it was computed.
# Empirically the rounding noise equals the delta at a ratio of ~1.7e-3.
BF16_DELTA_FLOOR = 5e-3


def report_delta_scale(ratios: list[float], merged_dtypes: set[torch.dtype]) -> None:
    """Warn if the merged delta is small enough that the output dtype quantizes it away."""
    if not ratios:
        return
    r = torch.tensor(ratios)
    med = r.median().item()
    print(
        f"  delta/weight norm:   median {med:.2e}  (min {r.min().item():.2e}, max {r.max().item():.2e})"
    )
    if torch.bfloat16 in merged_dtypes and med < BF16_DELTA_FLOOR:
        print(
            f"\n  ⚠️  median delta/weight is {med:.2e}, at or below bf16's resolution (~1.7e-3 is where\n"
            "      rounding noise equals the delta). Much of this LoRA will not survive being baked\n"
            "      into bf16 weights. Note that raising --dtype does not necessarily help: the\n"
            "      trainer loads the transformer as bf16 (trainer.py:557), so an fp32 file is\n"
            "      rounded at load time anyway. If the ratio is this small, keeping the LoRA\n"
            "      unfused at inference preserves its effect better than merging does."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", type=Path, required=True, help="Combined base .safetensors to merge into.")
    parser.add_argument("--lora", type=Path, required=True, help="LoRA .safetensors to bake in.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the merged base checkpoint.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "LoRA alpha used at training time. Scaling is alpha/rank. "
            "Omit to assume alpha == rank (scaling 1.0), which matches the trainer's default configs."
        ),
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Extra multiplier on the merged delta (default 1.0 = merge at full strength).",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="keep",
        help=(
            "Output precision. Default 'keep' preserves each tensor's original dtype "
            "(merged Linear weights stay bf16; the fp32 scale_shift_table params stay fp32). "
            "bf16/fp16/fp32 force every floating-point tensor to that precision."
        ),
    )
    args = parser.parse_args()

    for path in (args.base, args.lora):
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {args.output}")

    merge(args.base, args.lora, args.output, args.alpha, DTYPES[args.dtype], args.strength)


if __name__ == "__main__":
    main()
