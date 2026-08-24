"""Base class for training strategies.
This module defines the abstract base class that all training strategies must implement,
along with the base configuration class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from ltx_core.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
from ltx_trainer.timestep_samplers import TimestepSampler

# Default frames per second for video missing in the FPS metadata
DEFAULT_FPS = 24


def get_video_positions(
    num_frames: int,
    height: int,
    width: int,
    batch_size: int,
    fps: float,
    scale_factors: SpatioTemporalScaleFactors,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Per-token video positions ([B, 3, seq_len, 2] pixel-coordinate patch bounds).

    Uses ltx_core's native grid with the causal fix; the temporal axis is scaled by
    1/fps so positions are in seconds.
    """
    latent_coords = VideoLatentPatchifier(patch_size=1).get_patch_grid_bounds(
        output_shape=VideoLatentShape(
            frames=num_frames,
            height=height,
            width=width,
            batch=batch_size,
            channels=128,  # Video latent channels
        ),
        device=device,
    )
    pixel_coords = get_pixel_coords(latent_coords=latent_coords, scale_factors=scale_factors, causal_fix=True)
    pixel_coords = pixel_coords.to(dtype)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
    return pixel_coords


def get_audio_positions(
    num_time_steps: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Per-token audio positions ([B, 1, num_time_steps, 2]) for patchified audio latents
    ([B, T, C*mel_bins] with C=8 channels and 16 mel bins, as AudioPatchifier produces)."""
    latent_coords = AudioPatchifier(patch_size=1).get_patch_grid_bounds(
        output_shape=AudioLatentShape(
            frames=num_time_steps,
            mel_bins=16,
            batch=batch_size,
            channels=8,  # Audio latent channels
        ),
        device=device,
    )
    return latent_coords.to(dtype)


def create_per_token_timesteps(conditioning_mask: Tensor, sampled_sigma: Tensor) -> Tensor:
    """Per-token timesteps [B, seq_len]: conditioning tokens (mask True) get 0, the rest
    get their sample's sigma (accepted as [B,] or [B, 1, 1])."""
    expanded_sigma = sampled_sigma.view(-1, 1).expand_as(conditioning_mask)
    return torch.where(conditioning_mask, torch.zeros_like(expanded_sigma), expanded_sigma)


class TrainingStrategyConfigBase(BaseModel):
    """Base configuration class for training strategies.
    All strategy-specific configuration classes should inherit from this.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["text_to_video", "video_to_video"] = Field(
        description="Unique name identifying the training strategy type"
    )


@dataclass
class ModelInputs:
    """Container for model inputs using the Modality-based interface.

    A modality is ``None`` when the batch carries no data for it. The transformer
    skips that branch entirely (including both cross-modal attention directions),
    so nothing is trained against a placeholder stream.
    """

    video: Modality | None
    audio: Modality | None

    # Training targets (for loss computation). None when the modality is absent.
    video_targets: Tensor | None
    audio_targets: Tensor | None

    # Masks for loss computation. None when the modality is absent.
    video_loss_mask: Tensor | None  # Boolean mask: True = compute loss for this token
    audio_loss_mask: Tensor | None

    # Optional per-sample loss weights derived from sigma (e.g. bell weighting).
    # Shape: [batch_size]. Applied after the token-level mask normalization.
    sigma_loss_weights: Tensor | None = None

    # Metadata needed for loss computation in some strategies
    ref_seq_len: int | None = None  # For IC-LoRA: length of reference sequence

    @property
    def sigma(self) -> Tensor:
        """Per-sample sigma, shape [B,]. Both modalities share it, so read whichever is present."""
        modality = self.video if self.video is not None else self.audio
        if modality is None:
            raise ValueError("ModelInputs must carry at least one modality")
        return modality.sigma


class TrainingStrategy(ABC):
    """Abstract base class for training strategies.
    Each strategy encapsulates the logic for a specific training mode,
    handling input preparation and loss computation.
    """

    def __init__(self, config: TrainingStrategyConfigBase):
        """Initialize strategy with configuration.
        Args:
            config: Strategy-specific configuration
        """
        self.config = config
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)
        self.video_scale_factors = SpatioTemporalScaleFactors.default()

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components.
        Override this property in subclasses that support audio training.
        The trainer uses this to determine whether to load audio VAE and vocoder.
        Returns:
            True if audio components should be loaded, False otherwise.
        """
        return False

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Get the required data sources for this training strategy.
        Returns:
            Either a list of data directory names (where output keys match directory names)
            or a dictionary mapping data directory names to custom output keys for the dataset
        """

    def get_optional_data_sources(self) -> set[str]:
        """Get the output keys (values of ``get_data_sources()``) whose files may be
        missing per-sample. Strategies that support optional data override this.
        """
        return set()

    @abstractmethod
    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare training inputs from a raw data batch.
        Args:
            batch: Raw batch data from the dataset. Contains:
                - "latents": Video latent data
                - "conditions": Text embeddings with keys:
                    - "video_prompt_embeds": Already processed by embedding connectors
                    - "audio_prompt_embeds": Already processed by embedding connectors
                    - "prompt_attention_mask": Attention mask
                - Additional keys depending on strategy (e.g., "ref_latents" for IC-LoRA)
            timestep_sampler: Sampler for generating timesteps and noise
        Returns:
            ModelInputs containing Modality objects and training targets
        """

    @abstractmethod
    def compute_loss(
        self,
        video_pred: Tensor | None,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute the training loss.
        Args:
            video_pred: Video prediction from the transformer model (None for audio-only batches)
            audio_pred: Audio prediction from the transformer model (None for video-only batches)
            inputs: The prepared model inputs containing targets and masks
        Returns:
            Per-sample loss tensor of shape [B,].
        """

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get strategy-specific metadata to include in checkpoint files.
        Override this method in subclasses to add custom metadata,
        e.g. any parameters that a downstream inference pipeline may need.
        Returns:
            Dictionary of metadata key-value pairs (values must be JSON-serializable)
        """
        return {}

    @staticmethod
    def _create_first_frame_conditioning_mask(
        batch_size: int,
        sequence_length: int,
        height: int,
        width: int,
        device: torch.device,
        first_frame_conditioning_p: float = 0.0,
    ) -> Tensor:
        """Create conditioning mask for first frame conditioning.
        Args:
            batch_size: Batch size
            sequence_length: Total sequence length
            height: Latent height
            width: Latent width
            device: Target device
            first_frame_conditioning_p: Probability of conditioning on the first frame
        Returns:
            Boolean mask where True indicates first frame tokens (if conditioning is enabled)
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if first_frame_conditioning_p > 0:
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                apply_mask = torch.rand(batch_size, device=device) < first_frame_conditioning_p
                conditioning_mask[apply_mask, :first_frame_end_idx] = True

        return conditioning_mask
