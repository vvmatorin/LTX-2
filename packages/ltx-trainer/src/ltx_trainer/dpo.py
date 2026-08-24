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
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
from ltx_trainer import logger
from ltx_trainer.validation_runner import PromptEmbeddings

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

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
class DpoPairInputs:
    """Transformer inputs for a stacked [chosen; rejected] pair (row 0 = chosen)."""

    video: Modality
    video_targets: Tensor  # [2, S, C]
    video_loss_mask: Tensor  # [2, S]
    audio: Modality | None = None
    audio_targets: Tensor | None = None  # [2, T, C]
    audio_loss_mask: Tensor | None = None  # [2, T]


def discover_entries(samples_dir: str | Path) -> list[DpoEntry]:
    """Scan ``samples_dir`` for prompt files (.txt), pairing each with a same-stem image if present."""
    samples_dir = Path(samples_dir).expanduser()
    entries = []
    for txt_path in sorted(samples_dir.glob("*.txt")):
        prompt = txt_path.read_text().strip()
        if not prompt:
            logger.warning(f"Skipping DPO sample '{txt_path.name}': prompt file is empty")
            continue
        image_path = next(
            (p for ext in IMAGE_EXTENSIONS if (p := txt_path.with_suffix(ext)).exists()),
            None,
        )
        entries.append(DpoEntry(stem=txt_path.stem, prompt=prompt, image_path=image_path))

    if not entries:
        raise ValueError(f"No .txt prompt files found in DPO samples_dir: {samples_dir}")
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
) -> DpoPairInputs:
    """Build transformer inputs for a stacked [chosen; rejected] pair.

    Mirrors the text_to_video SFT input preparation: shared sigma (uniform, per Flow-DPO)
    and shared noise across the pair, first-frame conditioning for i2v pairs, velocity
    targets. When the pair carries audio latents, the audio modality is included (noised
    at the same sigma, like joint SFT) so the video branch sees its audio context.
    """
    patchifier = VideoLatentPatchifier(patch_size=1)

    latents = torch.stack([pair.chosen_latent, pair.rejected_latent]).to(device=device, dtype=torch.bfloat16)
    _, _, num_frames, height, width = latents.shape
    tokens = patchifier.patchify(latents)  # [2, S, C]
    seq_len = tokens.shape[1]
    frame_tokens = height * width

    sigma = torch.rand(1, device=device)
    noise = torch.randn(1, seq_len, tokens.shape[2], device=device, dtype=tokens.dtype)
    noisy = (1 - sigma) * tokens + sigma * noise
    targets = noise - tokens

    conditioning_mask = torch.zeros(2, seq_len, dtype=torch.bool, device=device)
    if pair.is_i2v:
        conditioning_mask[:, :frame_tokens] = True
        noisy = torch.where(conditioning_mask.unsqueeze(-1), tokens.float(), noisy)

    timesteps = torch.where(conditioning_mask, torch.zeros_like(sigma), sigma.expand(2, seq_len))

    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(frames=num_frames, height=height, width=width, batch=2, channels=128),
        device=device,
    )
    positions = get_pixel_coords(latent_coords, scale_factors=scale_factors, causal_fix=True).to(torch.float32)
    positions[:, 0, ...] = positions[:, 0, ...] / fps

    keyframes_mask = torch.zeros(2, seq_len, 1, device=device, dtype=torch.float32)
    keyframes_mask[:, :frame_tokens] = 1.0

    video_modality = Modality(
        enabled=True,
        sigma=sigma.expand(2),
        latent=noisy,
        timesteps=timesteps,
        positions=positions,
        context=pair.embeddings.video_context_positive.to(device).expand(2, -1, -1),
        context_mask=None,
        keyframes_mask=keyframes_mask,
    )
    inputs = DpoPairInputs(video=video_modality, video_targets=targets, video_loss_mask=(~conditioning_mask).float())

    has_audio = (
        pair.chosen_audio_latent is not None
        and pair.rejected_audio_latent is not None
        and pair.embeddings.audio_context_positive is not None
    )
    if not has_audio:
        return inputs

    audio_patchifier = AudioPatchifier(patch_size=1)
    audio_latents = torch.stack([pair.chosen_audio_latent, pair.rejected_audio_latent]).to(
        device=device, dtype=torch.bfloat16
    )
    audio_tokens = audio_patchifier.patchify(audio_latents)  # [2, T, C*mel_bins]
    audio_seq_len = audio_tokens.shape[1]

    audio_noise = torch.randn(1, audio_seq_len, audio_tokens.shape[2], device=device, dtype=audio_tokens.dtype)
    noisy_audio = (1 - sigma) * audio_tokens + sigma * audio_noise

    audio_positions = audio_patchifier.get_patch_grid_bounds(
        output_shape=AudioLatentShape(frames=audio_seq_len, mel_bins=16, batch=2, channels=8),
        device=device,
    ).to(torch.float32)

    inputs.audio = Modality(
        enabled=True,
        sigma=sigma.expand(2),
        latent=noisy_audio,
        timesteps=sigma.expand(2, audio_seq_len),
        positions=audio_positions,
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


def latest_round_with_labels(output_dir: str | Path) -> Path | None:
    """Most recent labeled round directory, or None. Used to restore pairs after a resume."""
    root = dpo_root(output_dir)
    if not root.exists():
        return None
    labeled = [p for p in sorted(root.glob("round_*")) if (p / LABELS_FILENAME).exists()]
    return labeled[-1] if labeled else None
