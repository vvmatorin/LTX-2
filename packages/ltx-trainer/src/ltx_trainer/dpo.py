"""Live-DPO support: human-in-the-loop preference optimization interleaved with training.

Implements the data plumbing for labeling rounds (https://arxiv.org/abs/2501.13918, Flow-DPO):
sample discovery, on-disk prompt-embedding cache, round manifests, the file-based labeling
handshake with the UI, preference-pair loading, and the Flow-DPO loss.

A labeling round lives in ``{output_dir}/dpo/round_{step:06d}/`` and holds the generated
videos, their final denoised latents, and two JSON files:
- ``pending.json``: written (atomically, last) by the trainer; its presence tells the UI
  that labeling is available.
- ``labels.json``: written (atomically, once) by the UI; its presence unblocks the trainer.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    ModelInputs,
    create_per_token_timesteps,
    get_audio_positions,
    get_video_positions,
)
from ltx_trainer.validation_runner import PromptEmbeddings

PENDING_FILENAME = "pending.json"
LABELS_FILENAME = "labels.json"
NEGATIVE_EMBEDDINGS_STEM = "_negative"


@dataclass(frozen=True)
class DpoEntry:
    """One labeling-round input: a prompt, optionally conditioned on an image."""

    stem: str
    prompt: str
    image_path: Path | None


@dataclass
class DpoPair:
    """A labeled preference pair. Latents are final denoised latents on CPU.

    Video latents are [C, F, H, W]; audio latents are [C, T, mel_bins] and present only
    when the round was generated with audio.
    """

    chosen_latent: Tensor
    rejected_latent: Tensor
    embeddings: PromptEmbeddings
    is_i2v: bool
    stem: str
    chosen_audio_latent: Tensor | None = None
    rejected_audio_latent: Tensor | None = None


@dataclass
class DpoState:
    """Runtime state of Live-DPO within a training run.

    Bundles the discovered entries, the current round's labeled pairs, the reference
    snapshot that generated them, and the round-robin cursor over pairs. Pairs and
    reference are only ever set together (a pair is meaningless without the policy
    that generated it), so mutate this state through its methods.
    """

    entries: list[DpoEntry] = field(default_factory=list)
    pairs: list[DpoPair] = field(default_factory=list)
    reference_weights: dict[str, Tensor] | None = None
    cursor: int = 0

    def start_round(self, named_params: list[tuple[str, Tensor]]) -> None:
        """Snapshot the live trainable weights as this round's reference (CPU copy).

        The snapshot is taken before generation, so it is the exact policy that produces
        the round's samples.
        """
        self.reference_weights = {
            name: param.detach().to("cpu", copy=True) for name, param in named_params
        }

    def load_labels(self, round_path: Path, output_dir: str | Path) -> int:
        """Load the labeled pairs of a round this state snapshotted; returns the pair count."""
        if self.reference_weights is None:
            raise RuntimeError("load_labels called before start_round: pairs need a reference snapshot")
        self.pairs = load_labeled_pairs(round_path, output_dir)
        self.cursor = 0
        return len(self.pairs)

    def next_pair(self, world_size: int, rank: int) -> DpoPair:
        """Strided round-robin over pairs: DDP ranks see different pairs each step."""
        pair = self.pairs[(self.cursor * world_size + rank) % len(self.pairs)]
        self.cursor += 1
        return pair

    def window_steps(self, repeats: int, world_size: int) -> int:
        """Combined-window length: steps needed to visit each pair ``repeats`` times."""
        return max(1, (len(self.pairs) * repeats + world_size - 1) // world_size)

    def in_window(self, global_step: int, interval: int, repeats: int, world_size: int) -> bool:
        """Whether an optimization step falls inside a combined SFT+DPO window."""
        return bool(self.pairs) and (global_step - 1) % interval < self.window_steps(repeats, world_size)

    @contextmanager
    def reference_swapped(self, named_params: list[tuple[str, Tensor]]) -> Iterator[None]:
        """Temporarily load the reference snapshot into the given live params.

        The stash of current weights stays on GPU (transient, LoRA-sized); the snapshot
        itself lives on CPU between DPO steps.
        """
        stash = [param.detach().clone() for _, param in named_params]
        try:
            with torch.no_grad():
                for name, param in named_params:
                    param.copy_(self.reference_weights[name])
            yield
        finally:
            with torch.no_grad():
                for (_, param), current in zip(named_params, stash, strict=True):
                    param.copy_(current)


@contextmanager
def dropout_disabled(module: torch.nn.Module) -> Iterator[None]:
    """Put only the Dropout submodules in eval mode for a deterministic forward.

    Deliberately not ``module.eval()``: the model must stay in training mode so
    gradient checkpointing — gated on ``self.training`` — remains active, or the
    un-checkpointed DPO forward blows up activation memory.

    The context must span the forward AND its backward: non-reentrant checkpointing
    recomputes the forward during backward and requires an identical graph, so
    re-enabling dropout in between raises a CheckpointError.
    """
    dropouts = [m for m in module.modules() if isinstance(m, torch.nn.Dropout) and m.training]
    try:
        for m in dropouts:
            m.eval()
        yield
    finally:
        for m in dropouts:
            m.train()


def pcgrad_combine(
    sft_grads: list[Tensor], dpo_grads: list[Tensor]
) -> tuple[list[Tensor], dict[str, float]]:
    """Blend two gradient sets with two-task PCGrad (arXiv:2001.06782).

    When the gradients conflict (negative inner product over the flattened parameter
    space), each is projected onto the other's normal plane before summing:
    ``g_sft' = g_sft - (dot/|g_dpo|^2)*g_dpo``, ``g_dpo' = g_dpo - (dot/|g_sft|^2)*g_sft``.
    Non-conflicting inputs reduce to plain addition.

    Returns (combined gradients, surgery metrics).
    """
    dot = sum(torch.sum(gs.float() * gd.float()) for gs, gd in zip(sft_grads, dpo_grads, strict=True))
    sft_norm_sq = sum(torch.sum(gs.float() ** 2) for gs in sft_grads)
    dpo_norm_sq = sum(torch.sum(gd.float() ** 2) for gd in dpo_grads)
    cosine = (dot / (sft_norm_sq * dpo_norm_sq).clamp(min=1e-24).sqrt()).item()

    conflict = dot.item() < 0 and sft_norm_sq.item() > 0 and dpo_norm_sq.item() > 0
    sft_coeff = (dot / sft_norm_sq.clamp(min=1e-24)).item() if conflict else 0.0
    dpo_coeff = (dot / dpo_norm_sq.clamp(min=1e-24)).item() if conflict else 0.0
    combined = [
        gs + gd - dpo_coeff * gd - sft_coeff * gs for gs, gd in zip(sft_grads, dpo_grads, strict=True)
    ]
    return combined, {"dpo/grad_cosine": cosine, "dpo/grad_conflict": float(conflict)}


def discover_entries(samples_file: str | Path) -> list[DpoEntry]:
    """Load labeling-round entries from a dataset JSON file in the regular metadata format.

    The file must contain a list of objects with a ``caption`` key and an optional
    ``media_path`` key (image for i2v conditioning; absent means text-to-video).
    Relative ``media_path`` values resolve against the JSON file's directory, matching
    the dataset preprocessing scripts.
    """
    json_path = Path(samples_file).expanduser()
    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"DPO dataset file must contain a list of objects: {json_path}")

    entries = []
    for index, item in enumerate(data):
        stem = f"{json_path.stem}_{index:04d}"
        caption = (item.get("caption") or "").strip()
        if not caption:
            logger.warning(f"Skipping DPO sample '{stem}': empty caption")
            continue

        image_path = None
        media = item.get("media_path")
        if media:
            image_path = Path(str(media).strip())
            if not image_path.is_absolute():
                image_path = json_path.parent / image_path
            if not image_path.is_file():
                logger.warning(f"Skipping DPO sample '{stem}': media_path does not exist: {image_path}")
                continue

        entries.append(DpoEntry(stem=stem, prompt=caption, image_path=image_path))

    if not entries:
        raise ValueError(f"No valid samples found in DPO samples_file: {json_path}")
    return entries


def dpo_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "dpo"


def embeddings_dir(output_dir: str | Path) -> Path:
    return dpo_root(output_dir) / "embeddings"


def round_dir(output_dir: str | Path, step: int) -> Path:
    return dpo_root(output_dir) / f"round_{step:06d}"


def save_embeddings(cache_dir: Path, stem: str, embeddings: dict[str, Tensor | None]) -> None:
    """Save positive-prompt encodings for one entry (tensors only, loadable with weights_only)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in embeddings.items() if v is not None}, cache_dir / f"{stem}.pt")


def load_embeddings(cache_dir: Path, stem: str) -> PromptEmbeddings:
    """Assemble PromptEmbeddings from an entry's cached positives and the shared negatives."""
    positive = torch.load(cache_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    negative = torch.load(cache_dir / f"{NEGATIVE_EMBEDDINGS_STEM}.pt", map_location="cpu", weights_only=True)
    return PromptEmbeddings(
        video_context_positive=positive["video"],
        audio_context_positive=positive.get("audio"),
        video_context_negative=negative["video"],
        audio_context_negative=negative.get("audio"),
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON via tmp+rename so readers never observe a partial file."""
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.rename(path)


def wait_for_labels(round_path: Path, poll_seconds: float = 5.0, reminder_seconds: float = 300.0) -> dict:
    """Block until the UI writes ``labels.json`` into the round directory, then return it parsed."""
    labels_path = round_path / LABELS_FILENAME
    waited = 0.0
    while not labels_path.exists():
        time.sleep(poll_seconds)
        waited += poll_seconds
        if waited % reminder_seconds < poll_seconds:
            logger.info(f"⏸ Still waiting for DPO labels ({waited / 60:.0f} min) — {labels_path}")
    return json.loads(labels_path.read_text())


def load_labeled_pairs(round_path: Path, output_dir: str | Path) -> list[DpoPair]:
    """Build preference pairs from a labeled round. Skipped/incomplete choices are ignored."""
    manifest = json.loads((round_path / PENDING_FILENAME).read_text())
    labels = json.loads((round_path / LABELS_FILENAME).read_text())
    samples_by_index = {s["index"]: s for s in manifest["samples"]}
    cache_dir = embeddings_dir(output_dir)

    pairs = []
    for choice in labels.get("choices", []):
        sample = samples_by_index.get(choice.get("index"))
        best, worst = choice.get("best"), choice.get("worst")
        if sample is None or choice.get("skipped") or best is None or worst is None or best == worst:
            continue

        def _load_latents(seed_idx: int, _sample: dict = sample) -> dict[str, Tensor]:
            return torch.load(round_path / _sample["latents"][seed_idx], map_location="cpu", weights_only=True)

        chosen, rejected = _load_latents(best), _load_latents(worst)
        pairs.append(
            DpoPair(
                chosen_latent=chosen["latents"],
                rejected_latent=rejected["latents"],
                embeddings=load_embeddings(cache_dir, sample["stem"]),
                is_i2v=sample["image"],
                stem=sample["stem"],
                chosen_audio_latent=chosen.get("audio_latents"),
                rejected_audio_latent=rejected.get("audio_latents"),
            )
        )
    return pairs


def prepare_pair_inputs(
    pair: DpoPair,
    fps: float,
    scale_factors: SpatioTemporalScaleFactors,
    device: torch.device,
    timestep_sampler: TimestepSampler,
) -> ModelInputs:
    """Build transformer inputs for a stacked [chosen; rejected] pair (row 0 = chosen).

    Mirrors the text_to_video SFT input preparation: a shared sigma drawn from the same
    timestep sampler as regular training (so preference pressure lands on the sigma range
    SFT actually maintains) and shared noise across the pair, first-frame conditioning
    for i2v pairs, velocity targets. When the pair carries audio latents, the audio
    modality is included (noised at the same sigma, like joint SFT) so the video branch
    sees its audio context.
    """
    latents = torch.stack([pair.chosen_latent, pair.rejected_latent]).to(device=device, dtype=torch.bfloat16)
    _, _, num_frames, height, width = latents.shape
    tokens = VideoLatentPatchifier(patch_size=1).patchify(latents)  # [2, S, C]
    seq_len = tokens.shape[1]
    frame_tokens = height * width

    sigma = timestep_sampler.sample_for(tokens[:1])  # [1], shared across the pair
    noise = torch.randn(1, seq_len, tokens.shape[2], device=device, dtype=tokens.dtype)
    noisy = (1 - sigma) * tokens + sigma * noise
    targets = noise - tokens

    conditioning_mask = torch.zeros(2, seq_len, dtype=torch.bool, device=device)
    if pair.is_i2v:
        conditioning_mask[:, :frame_tokens] = True
        noisy = torch.where(conditioning_mask.unsqueeze(-1), tokens, noisy)

    keyframes_mask = torch.zeros(2, seq_len, 1, device=device, dtype=torch.float32)
    keyframes_mask[:, :frame_tokens] = 1.0

    video_modality = Modality(
        enabled=True,
        sigma=sigma.expand(2),
        latent=noisy,
        timesteps=create_per_token_timesteps(conditioning_mask, sigma.expand(2)),
        positions=get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=2,
            fps=fps,
            scale_factors=scale_factors,
            device=device,
            dtype=torch.float32,
        ),
        context=pair.embeddings.video_context_positive.to(device).expand(2, -1, -1),
        context_mask=None,
        keyframes_mask=keyframes_mask,
    )
    inputs = ModelInputs(
        video=video_modality,
        audio=None,
        video_targets=targets,
        audio_targets=None,
        video_loss_mask=(~conditioning_mask).float(),
        audio_loss_mask=None,
    )

    has_audio = (
        pair.chosen_audio_latent is not None
        and pair.rejected_audio_latent is not None
        and pair.embeddings.audio_context_positive is not None
    )
    if not has_audio:
        return inputs

    audio_latents = torch.stack([pair.chosen_audio_latent, pair.rejected_audio_latent]).to(
        device=device, dtype=torch.bfloat16
    )
    audio_tokens = AudioPatchifier(patch_size=1).patchify(audio_latents)  # [2, T, C*mel_bins]
    audio_seq_len = audio_tokens.shape[1]

    audio_noise = torch.randn(1, audio_seq_len, audio_tokens.shape[2], device=device, dtype=audio_tokens.dtype)

    inputs.audio = Modality(
        enabled=True,
        sigma=sigma.expand(2),
        latent=(1 - sigma) * audio_tokens + sigma * audio_noise,
        timesteps=sigma.expand(2, audio_seq_len),
        positions=get_audio_positions(
            num_time_steps=audio_seq_len, batch_size=2, device=device, dtype=torch.float32
        ),
        context=pair.embeddings.audio_context_positive.to(device).expand(2, -1, -1),
        context_mask=None,
    )
    inputs.audio_targets = audio_noise - audio_tokens
    inputs.audio_loss_mask = torch.ones(2, audio_seq_len, device=device)
    return inputs


def masked_per_sample_mse(pred: Tensor, target: Tensor, loss_mask: Tensor) -> Tensor:
    """Per-sample MSE over non-conditioning tokens.

    Args:
        pred / target: [B, S, C]
        loss_mask: [B, S] float, 0 = excluded (conditioning token), 1 = active
    Returns:
        [B] float32 tensor.
    """
    mask = loss_mask.unsqueeze(-1).float()
    mse = (pred.float() - target.float()).pow(2).mul(mask).mean(dim=[-2, -1])
    return mse / mask.mean(dim=[-2, -1]).clamp(min=1e-8)


def flow_dpo_loss(
    policy_mse: Tensor,
    ref_mse: Tensor,
    beta: float,
    policy_audio_mse: Tensor | None = None,
    ref_audio_mse: Tensor | None = None,
    audio_loss_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Flow-DPO loss with constant beta (arXiv:2501.13918, eq. for rectified-flow models).

    Args:
        policy_mse / ref_mse: [2] per-sample video velocity MSEs, row 0 = chosen, row 1 = rejected.
        beta: KL-regularization strength.
        policy_audio_mse / ref_audio_mse: optional [2] audio velocity MSEs.
        audio_loss_weight: weight of the audio margin (margin = video + weight * audio).
    Returns:
        (scalar loss, metrics) where metrics carry the implicit-reward margin and pair accuracy.
    """
    margin = (policy_mse[0] - ref_mse[0]) - (policy_mse[1] - ref_mse[1])
    metrics = {
        "dpo/video_margin": -margin.item(),
        "dpo/chosen_mse": policy_mse[0].item(),
        "dpo/rejected_mse": policy_mse[1].item(),
    }

    if audio_loss_weight > 0 and policy_audio_mse is not None and ref_audio_mse is not None:
        audio_margin = (policy_audio_mse[0] - ref_audio_mse[0]) - (policy_audio_mse[1] - ref_audio_mse[1])
        metrics["dpo/audio_margin"] = -audio_margin.item()
        margin = margin + audio_loss_weight * audio_margin

    loss = -logsigmoid(-0.5 * beta * margin)
    metrics["dpo/margin"] = -margin.item()
    metrics["dpo/accuracy"] = float(margin.item() < 0)
    return loss, metrics
