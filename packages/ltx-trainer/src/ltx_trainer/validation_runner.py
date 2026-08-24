"""Validation sampling for LTX-2 training using ltx-core components.
This module provides a simplified validation pipeline for generating samples during training,
using the new ltx-core components (VideoLatentTools, AudioLatentTools, LatentState, etc.).
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal
from io import BytesIO

import av
import torch
from einops import rearrange
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.devices import cuda_activation_budget_bytes
from ltx_core.loader import SafetensorsModelStateDictLoader
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.video_vae import (
    DimensionSizeConfig,
    TileSizeConfig,
    diffvae_tiling_geometry_from_vae_config,
    estimate_diffusion_decoder_weight_bytes,
)
from ltx_core.model.video_vae.diffusion_tiling import recommended_decode_tiling_config
from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from ltx_core.model.video_vae.transformer.config import DiffVAEMode
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape
from ltx_trainer.progress import SamplingContext

if TYPE_CHECKING:
    from ltx_core.model.audio_vae import AudioDecoder, Vocoder
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import LTXGemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor

DEFAULT_IMAGE_CRF = 33


@dataclass
class PromptEmbeddings:
    """Processed text-context tensors for one prompt (positive and optional negative)."""

    video_context_positive: Tensor  # [1, seq_len, hidden_dim]
    audio_context_positive: Tensor  # [1, seq_len, hidden_dim]
    video_context_negative: Tensor | None = None
    audio_context_negative: Tensor | None = None

    def to(self, device: torch.device) -> "PromptEmbeddings":
        """Return a copy with all tensors moved to ``device``."""
        return PromptEmbeddings(
            video_context_positive=self.video_context_positive.to(device),
            audio_context_positive=self.audio_context_positive.to(device),
            video_context_negative=(
                self.video_context_negative.to(device) if self.video_context_negative is not None else None
            ),
            audio_context_negative=(
                self.audio_context_negative.to(device) if self.audio_context_negative is not None else None
            ),
        )


@dataclass
class TiledDecodingConfig:
    """Configuration for tiled video decoding to reduce VRAM usage.
    Tiled decoding splits the latent tensor into overlapping tiles, decodes each
    tile individually, and blends them together. This significantly reduces peak
    VRAM usage at the cost of slightly slower decoding.
    Defaults match the recommended values from ltx-core tests.
    """

    enabled: bool = False  # Whether to use tiled decoding (disabled by default)
    tile_size_pixels: int = 192  # Spatial tile size in pixels (must be ≥64 and divisible by 32)
    tile_overlap_pixels: int = 64  # Spatial tile overlap in pixels (must be divisible by 32)
    tile_size_frames: int = 48  # Temporal tile size in frames (must be ≥16 and divisible by 8)
    tile_overlap_frames: int = 24  # Temporal tile overlap in frames (must be divisible by 8)


@dataclass
class SampleOutput:
    """One generated sample."""

    video: Tensor  # [C, F, H, W] float32 CPU tensor in [0, 1]
    audio: Tensor | None  # [C, samples] waveform, or None
    latent: Tensor  # Final denoised video latent [C, F, H, W], bfloat16 CPU tensor
    audio_latent: Tensor | None = None  # Final denoised audio latent [C, T, mel_bins], bfloat16 CPU tensor


@dataclass
class GenerationConfig:
    """Configuration for video/audio generation."""

    prompt: str  # Text prompt for generation
    negative_prompt: str = ""  # Negative prompt to avoid unwanted artifacts
    height: int = 544  # Output video height in pixels
    width: int = 960  # Output video width in pixels
    num_frames: int = 97  # Number of frames to generate
    frame_rate: float = 24.0  # Frame rate for temporal position scaling
    num_inference_steps: int = 30  # Number of denoising steps
    guidance_scale: float = 4.0  # CFG guidance scale
    seed: int = 42  # Random seed for reproducibility
    # When set, one sample per seed is generated sequentially in a single call, sharing
    # the per-call costs (prompt embeddings, conditioning-image encode, VAE transfers)
    # across all seeds. `seed` is ignored.
    seeds: list[int] | None = None
    condition_image: Tensor | None = None  # Optional first frame image for image-to-video
    reference_video: Tensor | None = None  # For IC-LoRA: [F, C, H, W] in [0, 1]
    reference_downscale_factor: int = 1  # For IC-LoRA: downscale factor (1 = same resolution, 2 = half resolution)
    generate_audio: bool = True  # Whether to generate audio alongside video
    include_reference_in_output: bool = False  # For IC-LoRA: concatenate original reference with generated output
    cached_embeddings: PromptEmbeddings | None = None  # Pre-computed text embeddings (avoids loading Gemma)
    stg_scale: float = 0.0  # STG strength (0.0 = disabled, recommended: 1.0)
    stg_blocks: list[int] | None = None  # Transformer blocks to perturb (None = all, recommended: [29])
    stg_mode: Literal["stg_av", "stg_v"] = "stg_av"  # STG mode: "stg_av" (audio+video) or "stg_v" (video only)
    # Tiled decoding config: None or False = use defaults (disabled), or TiledDecodingConfig for custom settings
    tiled_decoding: TiledDecodingConfig | Literal[False] | None = None

    def __post_init__(self) -> None:
        """Apply default tiled decoding config if not provided."""
        if self.tiled_decoding is None:
            # Use default config with tiling disabled
            object.__setattr__(self, "tiled_decoding", TiledDecodingConfig())
        elif self.tiled_decoding is False:
            # Explicitly disabled - use config with enabled=False
            object.__setattr__(self, "tiled_decoding", TiledDecodingConfig(enabled=False))

    @property
    def resolved_seeds(self) -> list[int]:
        return self.seeds if self.seeds else [self.seed]


class ValidationSampler:
    """Generates validation samples during training using ltx-core components.
    This class provides a simplified interface for generating video (and optionally audio)
    samples during training validation. It supports:
    - Text-to-video generation
    - Image-to-video generation (first frame conditioning)
    - Video-to-video generation (IC-LoRA reference video conditioning)
    - Optional audio generation
    The implementation follows the patterns from ltx_pipelines.single_stage.
    Text embeddings can be provided either via:
    - A full text_encoder (encodes prompts on-the-fly)
    - Pre-computed cached_embeddings (avoids loading Gemma during validation)
    """

    def __init__(
        self,
        transformer: "LTXModel",
        vae_decoder: "VideoDecoder",
        vae_encoder: "VideoEncoder | None",
        text_encoder: "LTXGemmaTextEncoder | None" = None,
        audio_decoder: "AudioDecoder | None" = None,
        vocoder: "Vocoder | None" = None,
        sampling_context: SamplingContext | None = None,
        embeddings_processor: "EmbeddingsProcessor | None" = None,
        video_scale_factors: SpatioTemporalScaleFactors | None = None,
        video_vae_path: str | None = None,
    ):
        """Initialize the validation sampler.
        Args:
            transformer: LTX-2 transformer model
            vae_decoder: Video VAE decoder
            vae_encoder: Video VAE encoder (for image/video conditioning), can be None if not needed
            text_encoder: Gemma text encoder (optional if cached_embeddings in config)
            audio_decoder: Optional audio VAE decoder (for audio generation)
            vocoder: Optional vocoder (for audio generation)
            sampling_context: Optional SamplingContext for progress display during denoising
            embeddings_processor: Optional embeddings processor (required if text_encoder provided)
        """
        self._transformer = transformer
        self._video_scale_factors = video_scale_factors or SpatioTemporalScaleFactors.default()
        self._vae_decoder = vae_decoder
        # The diffusion decoder derives its tile overlaps from its receptive-field halos and
        # rejects smaller ones, so its tiling comes from the checkpoint geometry instead of
        # the conv-oriented TiledDecodingConfig numbers.
        self._diffvae_tiling_kwargs = None
        if isinstance(vae_decoder, DiffusionVideoDecoder) and video_vae_path is not None:
            metadata = SafetensorsModelStateDictLoader().metadata(str(video_vae_path))
            self._diffvae_tiling_kwargs = {
                **diffvae_tiling_geometry_from_vae_config(metadata.get("config", {}).get("vae", {})),
                "model_bytes": estimate_diffusion_decoder_weight_bytes(str(video_vae_path)),
            }
        self._vae_encoder = vae_encoder
        self._text_encoder = text_encoder
        self._embeddings_processor = embeddings_processor
        self._audio_decoder = audio_decoder
        self._vocoder = vocoder
        self._sampling_context = sampling_context

        # Patchifiers
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid
    # FSDP inplace update errors after validation
    @torch.no_grad()
    def generate(
        self,
        config: GenerationConfig,
        device: torch.device | str = "cuda",
    ) -> list[SampleOutput]:
        """Generate one sample per seed (see ``GenerationConfig.seeds``; defaults to one sample).

        Seeds run sequentially (batch size stays 1), but per-call costs are paid once:
        prompt embeddings, the conditioning-image encode, and the VAE decoder/vocoder
        transfers are shared across all seeds.
        """
        device = torch.device(device) if isinstance(device, str) else device
        self._validate_config(config)

        # Route to appropriate generation method
        if config.reference_video is not None:
            return self._generate_with_reference(config, device)
        return self._generate_standard(config, device)

    def _generate_standard(self, config: GenerationConfig, device: torch.device) -> list[SampleOutput]:
        """Standard generation (text-to-video or image-to-video), sequential over seeds."""
        # Get prompt embeddings (from cache or encode on-the-fly)
        embeddings = self._get_prompt_embeddings(config, device)

        # Create latent tools
        video_tools = self._create_video_latent_tools(config)
        audio_tools = self._create_audio_latent_tools(config) if config.generate_audio else None

        # Encode the conditioning image once; it is identical for every seed
        encoded_image = None
        if config.condition_image is not None:
            encoded_image = self._encode_conditioning_image(
                config.condition_image, config.height, config.width, device
            )

        video_latents: list[Tensor] = []
        audio_latents: list[Tensor] = []
        for seed in config.resolved_seeds:
            if self._sampling_context is not None:
                self._sampling_context.advance_video()
            generator = torch.Generator(device=device).manual_seed(seed)

            # Create initial states
            video_clean = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
            audio_clean = (
                audio_tools.create_initial_state(device=device, dtype=torch.bfloat16) if audio_tools else None
            )

            if encoded_image is not None:
                video_clean = self._apply_image_conditioning(video_clean, encoded_image)

            # Add noise
            noiser = GaussianNoiser(generator=generator)
            video_state = noiser(latent_state=video_clean, noise_scale=1.0)
            audio_state = noiser(latent_state=audio_clean, noise_scale=1.0) if audio_clean else None

            # Run denoising loop
            video_state, audio_state = self._run_denoising(
                config=config,
                video_state=video_state,
                audio_state=audio_state,
                video_clean=video_clean,
                audio_clean=audio_clean,
                embeddings=embeddings,
                num_target_tokens=video_tools.patchifier.get_token_count(video_tools.target_shape),
                device=device,
            )

            video_state = video_tools.clear_conditioning(video_state)
            video_state = video_tools.unpatchify(video_state)
            video_latents.append(video_state.latent.to(torch.bfloat16))

            if audio_state is not None and audio_tools is not None:
                audio_state = audio_tools.clear_conditioning(audio_state)
                audio_state = audio_tools.unpatchify(audio_state)
                audio_latents.append(audio_state.latent.to(torch.bfloat16))

        # Decode all seeds' outputs with a single VAE decoder (and vocoder) transfer
        videos = self._decode_video_rows(video_latents, device, config.tiled_decoding)
        audios: list[Tensor | None] = (
            self._decode_audio_rows(audio_latents, device) if audio_latents else [None] * len(videos)
        )

        return [
            SampleOutput(
                video=videos[i],
                audio=audios[i],
                latent=video_latents[i][0].cpu(),
                audio_latent=audio_latents[i][0].cpu() if audio_latents else None,
            )
            for i in range(len(videos))
        ]

    def _generate_with_reference(self, config: GenerationConfig, device: torch.device) -> list[SampleOutput]:
        """Generate with reference video conditioning (IC-LoRA style).
        For IC-LoRA:
        - Reference video latents are concatenated with target latents
        - Reference latents have timestep=0 (clean, not denoised)
        - Target latents are denoised normally
        - If condition_image is also provided, the first frame of the target is conditioned
        - If include_reference_in_output is True, the preprocessed reference video
          is concatenated side-by-side with the generated video
        """
        if config.seeds is not None and len(config.seeds) > 1:
            raise ValueError("Multi-seed batching is not supported with reference video conditioning")

        # Get prompt embeddings (from cache or encode on-the-fly)
        embeddings = self._get_prompt_embeddings(config, device)

        # Setup generator
        generator = torch.Generator(device=device).manual_seed(config.resolved_seeds[0])

        # Preprocess and encode reference video
        ref_video_preprocessed = self._preprocess_reference_video(config)
        ref_latent, ref_positions = self._encode_video(ref_video_preprocessed, config.frame_rate, device)
        ref_seq_len = ref_latent.shape[1]

        # Scale reference positions to match target coordinate space
        # Position tensor shape: [B, 3, seq_len, 2] where dim 1 is (time, height, width)
        if config.reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= config.reference_downscale_factor  # height axis
            ref_positions[:, 2, ...] *= config.reference_downscale_factor  # width axis
            # Time axis (index 0) remains unchanged

        # Create target video state
        video_tools = self._create_video_latent_tools(config)
        target_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)

        # Apply first-frame image conditioning to target if provided
        if config.condition_image is not None:
            encoded_image = self._encode_conditioning_image(
                config.condition_image, config.height, config.width, device
            )
            target_clean_state = self._apply_image_conditioning(target_clean_state, encoded_image)

        # Create combined state (reference + target)
        # denoise_mask shape is [B, seq_len, 1] after patchification
        ref_denoise_mask = torch.zeros(1, ref_seq_len, 1, device=device, dtype=torch.float32)
        # Reference tokens are never keyframes (matches VideoConditionByReferenceLatent,
        # which extends the mask with marked=False); only the target's marks survive.
        ref_keyframes_mask = torch.zeros(1, ref_seq_len, 1, device=device, dtype=torch.float32)
        combined_clean_state = LatentState(
            latent=torch.cat([ref_latent, target_clean_state.latent], dim=1),
            denoise_mask=torch.cat([ref_denoise_mask, target_clean_state.denoise_mask], dim=1),
            positions=torch.cat([ref_positions, target_clean_state.positions], dim=2),
            clean_latent=torch.cat([ref_latent, target_clean_state.clean_latent], dim=1),
            keyframes_mask=torch.cat([ref_keyframes_mask, target_clean_state.keyframes_mask], dim=1),
        )

        # Add noise (only to the target portion via denoise_mask)
        noiser = GaussianNoiser(generator=generator)
        combined_state = noiser(latent_state=combined_clean_state, noise_scale=1.0)

        # Create audio state if needed
        audio_tools = self._create_audio_latent_tools(config) if config.generate_audio else None
        audio_clean = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16) if audio_tools else None
        audio_state = noiser(latent_state=audio_clean, noise_scale=1.0) if audio_clean else None

        # Run denoising loop
        if self._sampling_context is not None:
            self._sampling_context.advance_video()
        combined_state, audio_state = self._run_denoising(
            config=config,
            video_state=combined_state,
            audio_state=audio_state,
            video_clean=combined_clean_state,
            audio_clean=audio_clean,
            embeddings=embeddings,
            num_target_tokens=video_tools.patchifier.get_token_count(video_tools.target_shape),
            device=device,
        )

        # Extract target portion, unpatchify, and decode
        target_latent = self._unpatchify_target_latent(combined_state.latent[:, ref_seq_len:], config)
        video_output = self._decode_video_rows([target_latent], device, config.tiled_decoding)[0]

        # Optionally concatenate original reference video side-by-side
        if config.include_reference_in_output:
            # Use preprocessed reference (already resized/cropped, in pixel space)
            # Convert from [B, C, F, H, W] to [C, F, H, W]
            ref_video_pixels = ref_video_preprocessed[0].cpu()
            # Normalize from [-1, 1] to [0, 1]
            ref_video_pixels = ((ref_video_pixels + 1.0) / 2.0).clamp(0.0, 1.0)
            video_output = self._concatenate_videos_side_by_side(ref_video_pixels, video_output)

        # Decode audio
        audio_output = None
        audio_latent = None
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            audio_latent = audio_state.latent.to(torch.bfloat16)
            audio_output = self._decode_audio_rows([audio_latent], device)[0]

        return [
            SampleOutput(
                video=video_output,
                audio=audio_output,
                latent=target_latent[0].cpu(),
                audio_latent=audio_latent[0].cpu() if audio_latent is not None else None,
            )
        ]

    def _create_video_latent_tools(self, config: GenerationConfig) -> VideoLatentTools:
        """Create video latent tools for the given configuration."""
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.frame_rate,
        )
        return VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(shape=pixel_shape),
            fps=config.frame_rate,
            scale_factors=self._video_scale_factors,
            causal_fix=True,
        )

    def _create_audio_latent_tools(self, config: GenerationConfig) -> AudioLatentTools:
        """Create audio latent tools for the given configuration."""
        return AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=config.num_frames / config.frame_rate),
        )

    def _apply_image_conditioning(self, video_state: LatentState, encoded_image: Tensor) -> LatentState:
        """Apply first-frame conditioning to the video state from a pre-encoded image latent."""
        # Patchify the encoded image (single frame)
        patchified_image = self._video_patchifier.patchify(encoded_image)  # [1, 1, C] -> [1, num_patches, C]
        num_image_tokens = patchified_image.shape[1]

        # Update the first frame tokens in the latent
        new_latent = video_state.latent.clone()
        new_latent[:, :num_image_tokens] = patchified_image.to(new_latent.dtype)

        # Update clean_latent as well (conditioning image is clean)
        new_clean_latent = video_state.clean_latent.clone()
        new_clean_latent[:, :num_image_tokens] = patchified_image.to(new_clean_latent.dtype)

        # Set denoise_mask to 0 for conditioned tokens (don't denoise them)
        new_denoise_mask = video_state.denoise_mask.clone()
        new_denoise_mask[:, :num_image_tokens] = 0.0

        return LatentState(
            latent=new_latent,
            denoise_mask=new_denoise_mask,
            positions=video_state.positions,
            clean_latent=new_clean_latent,
            keyframes_mask=video_state.keyframes_mask,
        )

    @staticmethod
    def _preprocess_reference_video(config: GenerationConfig) -> Tensor:
        """Preprocess reference video: resize, crop, and convert to model input format.
        When reference_downscale_factor > 1, the reference video is downscaled to a smaller
        resolution for more efficient inference. The positions will be scaled up later
        to match the target coordinate space.
        Args:
            config: Generation configuration
        Returns:
            Preprocessed video tensor [B, C, F, H, W] in [-1, 1] range
        """
        ref_video = config.reference_video  # [F, C, H, W] in [0, 1]
        scale_factor = config.reference_downscale_factor

        # Target dimensions for reference (scaled down if scale_factor > 1)
        target_height = config.height // scale_factor
        target_width = config.width // scale_factor

        # Validate scaled dimensions
        if target_height % 32 != 0 or target_width % 32 != 0:
            raise ValueError(
                f"Scaled reference dimensions ({target_height}x{target_width}) must be divisible by 32. "
                f"Original: {config.height}x{config.width}, scale_factor: {scale_factor}"
            )

        current_height, current_width = ref_video.shape[2:]

        # Resize maintaining aspect ratio and center crop if needed
        if current_height != target_height or current_width != target_width:
            aspect_ratio = current_width / current_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                resize_height, resize_width = target_height, int(target_height * aspect_ratio)
            else:
                resize_height, resize_width = int(target_width / aspect_ratio), target_width

            ref_video = torch.nn.functional.interpolate(
                ref_video, size=(resize_height, resize_width), mode="bilinear", align_corners=False
            )

            # Center crop
            h_start = (resize_height - target_height) // 2
            w_start = (resize_width - target_width) // 2
            ref_video = ref_video[:, :, h_start: h_start + target_height, w_start: w_start + target_width]

        # Convert to [B, C, F, H, W] and trim to valid frame count (k*8 + 1)
        ref_video = rearrange(ref_video, "f c h w -> 1 c f h w")
        valid_frames = (ref_video.shape[2] - 1) // 8 * 8 + 1
        ref_video = ref_video[:, :, :valid_frames]

        # Convert to [-1, 1] range
        return ref_video * 2.0 - 1.0

    def _encode_video(self, video: Tensor, fps: float, device: torch.device) -> tuple[Tensor, Tensor]:
        """Encode video to patchified latents and compute positions.
        Args:
            video: Video tensor [B, C, F, H, W] in [-1, 1] range
            fps: Frame rate for temporal position scaling
            device: Device to run encoding on
        Returns:
            Tuple of (patchified_latents, positions)
        """
        video = video.to(device=device, dtype=torch.float32)

        # Encode with VAE
        self._vae_encoder.to(device)
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            latents = self._vae_encoder(video)
        self._vae_encoder.to("cpu")

        latents = latents.to(torch.bfloat16)
        patchified = self._video_patchifier.patchify(latents)

        # Compute positions
        latent_shape = VideoLatentShape(
            batch=1,
            channels=latents.shape[1],
            frames=latents.shape[2],
            height=latents.shape[3],
            width=latents.shape[4],
        )
        latent_coords = self._video_patchifier.get_patch_grid_bounds(output_shape=latent_shape, device=device)
        positions = get_pixel_coords(latent_coords, scale_factors=self._video_scale_factors, causal_fix=True)
        positions = positions.to(torch.bfloat16)
        positions[:, 0, ...] = positions[:, 0, ...] / fps

        return patchified, positions

    def _run_denoising(
        self,
        config: GenerationConfig,
        video_state: LatentState,
        audio_state: LatentState | None,
        video_clean: LatentState,
        audio_clean: LatentState | None,
        embeddings: PromptEmbeddings,
        num_target_tokens: int,
        device: torch.device,
    ) -> tuple[LatentState, LatentState | None]:
        """Run the denoising loop using X0 prediction with CFG and optional STG."""
        scheduler = LTX2Scheduler()
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(config.guidance_scale)
        stg_guider = STGGuider(config.stg_scale)

        sigmas = (
            scheduler.execute(steps=config.num_inference_steps, default_number_of_tokens=num_target_tokens)
            .to(device)
            .float()
        )

        # Build STG perturbation config if STG is enabled
        stg_perturbation_config = self._build_stg_perturbation_config(config) if stg_guider.enabled() else None

        # Create initial modalities (will be updated each step via replace())
        video = Modality(
            enabled=True,
            latent=video_state.latent,
            sigma=sigmas[0].repeat(video_state.latent.shape[0]),
            timesteps=video_state.denoise_mask,
            positions=video_state.positions,
            context=embeddings.video_context_positive,
            context_mask=None,
            keyframes_mask=video_state.keyframes_mask,
        )

        # Audio modality is None when not generating audio
        audio: Modality | None = None
        if audio_state is not None:
            audio = Modality(
                enabled=True,
                latent=audio_state.latent,
                sigma=sigmas[0].repeat(audio_state.latent.shape[0]),
                timesteps=audio_state.denoise_mask,
                positions=audio_state.positions,
                context=embeddings.audio_context_positive,
                context_mask=None,
            )

        # Wrap transformer with X0Model to convert velocity predictions to denoised outputs
        self._transformer.to(device)
        x0_model = X0Model(self._transformer)

        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            for step_idx, sigma in enumerate(sigmas[:-1]):
                # Update modalities with current state and timesteps
                video = replace(
                    video,
                    latent=video_state.latent,
                    sigma=sigma.repeat(video_state.latent.shape[0]),
                    timesteps=sigma * video_state.denoise_mask,
                    positions=video_state.positions,
                )

                if audio is not None and audio_state is not None:
                    audio = replace(
                        audio,
                        latent=audio_state.latent,
                        sigma=sigma.repeat(audio_state.latent.shape[0]),
                        timesteps=sigma * audio_state.denoise_mask,
                        positions=audio_state.positions,
                    )

                # Run model (positive pass) - X0Model returns denoised outputs
                pos_video, pos_audio = x0_model(video=video, audio=audio, perturbations=None)
                denoised_video, denoised_audio = pos_video, pos_audio

                # Apply CFG if guidance_scale != 1.0
                if cfg_guider.enabled() and embeddings.video_context_negative is not None:
                    video_neg = replace(video, context=embeddings.video_context_negative)
                    audio_neg = replace(
                        audio, context=embeddings.audio_context_negative) if audio is not None else None
                    neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)

                    denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)
                    if audio is not None and denoised_audio is not None:
                        denoised_audio = denoised_audio + cfg_guider.delta(pos_audio, neg_audio)

                # Apply STG if stg_scale != 0.0
                if stg_guider.enabled() and stg_perturbation_config is not None:
                    perturbed_video, perturbed_audio = x0_model(
                        video=video, audio=audio, perturbations=stg_perturbation_config
                    )
                    denoised_video = denoised_video + stg_guider.delta(pos_video, perturbed_video)
                    if audio is not None and denoised_audio is not None and perturbed_audio is not None:
                        denoised_audio = denoised_audio + stg_guider.delta(pos_audio, perturbed_audio)

                # Apply conditioning mask (keep conditioned tokens clean)
                denoised_video = denoised_video * video_state.denoise_mask + video_clean.latent.float() * (
                    1 - video_state.denoise_mask
                )
                if audio is not None and audio_state is not None and audio_clean is not None:
                    denoised_audio = denoised_audio * audio_state.denoise_mask + audio_clean.latent.float() * (
                        1 - audio_state.denoise_mask
                    )

                # Euler step
                video_state = replace(
                    video_state,
                    latent=stepper.step(
                        sample=video.latent, denoised_sample=denoised_video, sigmas=sigmas, step_index=step_idx
                    ),
                )
                if audio is not None and audio_state is not None:
                    audio_state = replace(
                        audio_state,
                        latent=stepper.step(
                            sample=audio.latent, denoised_sample=denoised_audio, sigmas=sigmas, step_index=step_idx
                        ),
                    )

                # Update progress
                if self._sampling_context is not None:
                    self._sampling_context.advance_step()

        return video_state, audio_state

    def _build_stg_perturbation_config(self, config: GenerationConfig) -> BatchedPerturbationConfig:
        """Build the perturbation config for STG based on the stg_mode."""
        # Always skip video self-attention for STG
        perturbations: list[Perturbation] = [
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=config.stg_blocks)
        ]

        # Optionally also skip audio self-attention (stg_av mode)
        if config.stg_mode == "stg_av":
            perturbations.append(Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=config.stg_blocks))

        perturbation_config = PerturbationConfig(perturbations=perturbations)
        # Batch size is 1 for validation
        return BatchedPerturbationConfig(
            perturbations=[perturbation_config],
            num_blocks=self._transformer.num_blocks,
        )

    def _decode_tiling_config(
        self, latent: Tensor, device: torch.device, tiled_config: TiledDecodingConfig | None
    ) -> TileSizeConfig | None:
        """Tile layout for decoding ``latent``, or None for an untiled decode."""
        if tiled_config is None or not tiled_config.enabled:
            return None

        if self._diffvae_tiling_kwargs is not None:
            scale = self._video_scale_factors
            _, _, latent_frames, latent_height, latent_width = latent.shape
            return recommended_decode_tiling_config(
                **self._diffvae_tiling_kwargs,
                height=latent_height * scale.height,
                width=latent_width * scale.width,
                num_frames=(latent_frames - 1) * scale.time + 1,
                # Matches the ModuleOps that load_video_vae_decoder applies to the diffusion decoder.
                mode=DiffVAEMode.CHUNKED_EAGER,
                free_bytes=cuda_activation_budget_bytes(device) if device.type == "cuda" else 0,
            )

        return TileSizeConfig(
            frames=DimensionSizeConfig(
                tile_size=tiled_config.tile_size_frames,
                overlap=tiled_config.tile_overlap_frames,
            ),
            height=DimensionSizeConfig(
                tile_size=tiled_config.tile_size_pixels,
                overlap=tiled_config.tile_overlap_pixels,
            ),
            width=DimensionSizeConfig(
                tile_size=tiled_config.tile_size_pixels,
                overlap=tiled_config.tile_overlap_pixels,
            ),
        )

    def _unpatchify_target_latent(self, latent: Tensor, config: GenerationConfig) -> Tensor:
        """Unpatchify [1, seq_len, C] target tokens to [1, C, F, H, W] bfloat16."""
        latent_frames = config.num_frames // self._video_scale_factors.time + 1
        latent_height = config.height // self._video_scale_factors.height
        latent_width = config.width // self._video_scale_factors.width

        unpatchified = self._video_patchifier.unpatchify(
            latent,
            output_shape=VideoLatentShape(
                height=latent_height,
                width=latent_width,
                frames=latent_frames,
                batch=1,
                channels=128,
            ),
        )
        return unpatchified.to(dtype=torch.bfloat16)

    def _validate_config(self, config: GenerationConfig) -> None:
        """Validate generation configuration."""
        if config.height % 32 != 0 or config.width % 32 != 0:
            raise ValueError(f"height and width must be divisible by 32, got {config.height}x{config.width}")
        if config.num_frames % 8 != 1:
            raise ValueError(f"num_frames must satisfy num_frames % 8 == 1, got {config.num_frames}")
        if config.generate_audio and (self._audio_decoder is None or self._vocoder is None):
            raise ValueError("Audio generation requires audio_decoder and vocoder")
        if config.condition_image is not None and self._vae_encoder is None:
            raise ValueError("Image conditioning requires vae_encoder")
        if config.reference_video is not None and self._vae_encoder is None:
            raise ValueError("Reference video conditioning requires vae_encoder")

        # Validate prompt embedding source
        if config.cached_embeddings is None and self._text_encoder is None:
            raise ValueError("Either text_encoder or config.cached_embeddings must be provided")
        if config.cached_embeddings is None and self._embeddings_processor is None:
            raise ValueError("embeddings_processor is required when encoding prompts on-the-fly")

    def _get_prompt_embeddings(self, config: GenerationConfig, device: torch.device) -> PromptEmbeddings:
        """Get prompt embeddings from config cache or encode on-the-fly."""
        if config.cached_embeddings is not None:
            return config.cached_embeddings.to(device)
        return self._encode_prompts(config, device)

    def _encode_prompts(self, config: GenerationConfig, device: torch.device) -> PromptEmbeddings:
        """Encode positive and negative prompts using the text encoder + embeddings processor."""
        self._text_encoder.to(device)
        self._embeddings_processor.to(device)

        pos_hs, pos_mask = self._text_encoder.encode([config.prompt])[0]
        pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)

        v_ctx_neg, a_ctx_neg = None, None
        if config.guidance_scale != 1.0:
            neg_hs, neg_mask = self._text_encoder.encode([config.negative_prompt])[0]
            neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)
            v_ctx_neg, a_ctx_neg = neg_out.video_encoding, neg_out.audio_encoding

        # Move the base Gemma model to CPU
        self._text_encoder.model.to("cpu")

        return PromptEmbeddings(
            video_context_positive=pos_out.video_encoding,
            audio_context_positive=pos_out.audio_encoding,
            video_context_negative=v_ctx_neg,
            audio_context_negative=a_ctx_neg,
        )

    def _decode_video_rows(
        self, latents: list[Tensor], device: torch.device, tiled_config: TiledDecodingConfig | None = None
    ) -> list[Tensor]:
        """Decode a list of [1, C, F, H, W] video latents with a single decoder transfer.
        Args:
            latents: Video latents to decode, one [1, C, F, H, W] tensor per sample
            device: Device to run decoding on
            tiled_config: Optional tiled decoding configuration for reduced VRAM usage
        Returns:
            Decoded video tensors [C, F, H, W] in [0, 1] range, one per input latent
        """
        self._vae_decoder.to(device)
        videos = []
        for row in latents:
            # Ensure latent is bfloat16 to match decoder weights
            latent = row.to(device=device, dtype=torch.bfloat16)

            tiling_config = self._decode_tiling_config(latent, device, tiled_config)
            if tiling_config is not None:
                chunks = list(self._vae_decoder.tiled_decode(latent, tiling_config=tiling_config))
                decoded_video = torch.cat(chunks, dim=2)
            else:
                decoded_video = self._vae_decoder(latent)

            decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
            videos.append(decoded_video[0].float().cpu())
        self._vae_decoder.to("cpu")
        return videos

    def _decode_audio_rows(self, latents: list[Tensor], device: torch.device) -> list[Tensor]:
        """Decode a list of audio latents to waveforms with a single decoder/vocoder transfer."""
        self._audio_decoder.to(device)
        first_param = next(self._audio_decoder.parameters(), None)
        decoder_dtype = first_param.dtype if first_param is not None else latents[0].dtype
        decoded = [self._audio_decoder(latent.to(dtype=decoder_dtype, device=device)) for latent in latents]
        self._audio_decoder.to("cpu")

        self._vocoder.to(device)
        waveforms = [self._vocoder(decoded_audio).squeeze(0).float().cpu() for decoded_audio in decoded]
        self._vocoder.to("cpu")

        return waveforms

    @staticmethod
    def _concatenate_videos_side_by_side(left_video: Tensor, right_video: Tensor) -> Tensor:
        """Concatenate two videos side-by-side (horizontally).
        If the videos have different frame counts, the shorter one is padded with
        its last frame repeated.
        Args:
            left_video: Left video tensor [C, F1, H1, W1] in [0, 1]
            right_video: Right video tensor [C, F2, H2, W2] in [0, 1]
        Returns:
            Concatenated video tensor [C, max(F1,F2), H2, W1_scaled+W2] in [0, 1]
        """
        left_height, left_width = left_video.shape[2], left_video.shape[3]
        right_height = right_video.shape[2]

        # Resize left video to match right video's height if needed
        if left_height != right_height:
            # Scale width proportionally to maintain aspect ratio
            scale = right_height / left_height
            new_width = int(left_width * scale)
            # Interpolate expects [N, C, H, W], we have [C, F, H, W]
            # Reshape to [C*F, 1, H, W] -> interpolate -> reshape back
            c, f, h, w = left_video.shape
            left_video = left_video.reshape(c * f, 1, h, w)
            left_video = torch.nn.functional.interpolate(
                left_video, size=(right_height, new_width), mode="bilinear", align_corners=False
            )
            left_video = left_video.reshape(c, f, right_height, new_width)

        left_frames = left_video.shape[1]
        right_frames = right_video.shape[1]

        # Pad shorter video by repeating last frame
        if left_frames < right_frames:
            padding = left_video[:, -1:, :, :].expand(-1, right_frames - left_frames, -1, -1)
            left_video = torch.cat([left_video, padding], dim=1)
        elif right_frames < left_frames:
            padding = right_video[:, -1:, :, :].expand(-1, left_frames - right_frames, -1, -1)
            right_video = torch.cat([right_video, padding], dim=1)

        # Concatenate along width dimension
        return torch.cat([left_video, right_video], dim=3)

    @staticmethod
    def _preprocess_conditioning_image(
        image: Tensor,
        target_height: int,
        target_width: int,
        crf: int = DEFAULT_IMAGE_CRF,
    ) -> Tensor:
        """Preprocess a conditioning image: apply H.264 CRF compression, resize, and center-crop.
        The CRF round-trip matches the inference pipeline (ltx-pipelines) to simulate
        video codec artifacts the model was trained on.
        Args:
            image: Image tensor [C, H, W] in [0, 1]
            target_height: Target height in pixels (must be divisible by 32)
            target_width: Target width in pixels (must be divisible by 32)
            crf: H.264 CRF quality (0=lossless, 33=default matching inference pipeline)
        Returns:
            Preprocessed image tensor [1, C, 1, H, W] in [-1, 1]
        """
        # Apply H.264 CRF compression round-trip
        if crf > 0:
            image = image.permute(1, 2, 0).clamp(0, 1).mul_(255.0).to(torch.uint8).cpu().numpy()
            h, w = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2

            image = image[:h, :w]
            with BytesIO() as buffer:
                with av.open(buffer, "w", format="mp4") as container:
                    stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
                    stream.height, stream.width = h, w
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
                    container.mux(stream.encode(frame))
                    container.mux(stream.encode())

                buffer.seek(0)
                with av.open(buffer) as container:
                    stream = next(s for s in container.streams if s.type == "video")
                    decoded = next(container.decode(stream)).to_ndarray(format="rgb24")

                image = torch.from_numpy(decoded).float().div_(255.0).permute(2, 0, 1)

        # Resize maintaining aspect ratio and center-crop
        current_height, current_width = image.shape[1:]
        if current_height != target_height or current_width != target_width:
            aspect_ratio = current_width / current_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                # Image is wider than target - resize to match height, crop width
                resize_height = target_height
                resize_width = int(target_height * aspect_ratio)
            else:
                # Image is taller than target - resize to match width, crop height
                resize_height = int(target_width / aspect_ratio)
                resize_width = target_width

            image = rearrange(image, "c h w -> 1 c h w")
            image = torch.nn.functional.interpolate(
                image, size=(resize_height, resize_width), mode="bilinear", align_corners=False
            )

            # Center crop to target dimensions
            h_start = (resize_height - target_height) // 2
            w_start = (resize_width - target_width) // 2
            image = image[:, :, h_start: h_start + target_height, w_start: w_start + target_width]
        else:
            image = rearrange(image, "c h w -> 1 c h w")

        # Add frame dimension and convert to [-1, 1]
        image = rearrange(image, "b c h w -> b c 1 h w")
        image = image * 2.0 - 1.0

        return image

    def _encode_conditioning_image(
        self,
        image: Tensor,
        target_height: int,
        target_width: int,
        device: torch.device,
    ) -> Tensor:
        """Preprocess and encode a conditioning image to latent space."""
        image = self._preprocess_conditioning_image(image, target_height, target_width)
        image = image.to(device=device, dtype=torch.float32)

        self._vae_encoder.to(device)
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            encoded = self._vae_encoder(image)
        self._vae_encoder.to("cpu")

        return encoded
