import { NextResponse } from "next/server";
import { db } from "@/db";
import { jobs, trainingDatasets } from "@/db/schema";
import { nextQueuePosition } from "@/db/queries";
import { eq } from "drizzle-orm";
import fs from "fs";
import path from "path";
import YAML from "yaml";
import type { TrainingConfig } from "@/lib/types";

export async function POST(req: Request) {
  const body = await req.json();
  const uiConfig = body.config as TrainingConfig;

  const outputDir: string = uiConfig?.outputDir || "/tmp/ltx-training";
  fs.mkdirSync(outputDir, { recursive: true });

  let preprocessedDataRoot: string | null = null;
  if (body.datasetName) {
    const dataset = db
      .select()
      .from(trainingDatasets)
      .where(eq(trainingDatasets.name, body.datasetName))
      .get();
    if (dataset) {
      preprocessedDataRoot = path.join(dataset.path, ".precomputed");
    }
  }
  if (!preprocessedDataRoot && uiConfig?.data?.preprocessedDataRoot) {
    preprocessedDataRoot = uiConfig.data.preprocessedDataRoot as string;
  }

  const yamlConfig = buildYamlConfig(uiConfig, preprocessedDataRoot);
  const configPath = path.join(outputDir, "training_config.yaml");
  fs.writeFileSync(configPath, YAML.stringify(yamlConfig), "utf-8");

  const result = db
    .insert(jobs)
    .values({
      type: "training",
      name: body.name || path.basename(outputDir),
      status: "queued",
      config: JSON.stringify({
        ...uiConfig,
        configPath,
        gpuMode: body.gpuMode,
        gpuIds: body.gpuIds,
        datasetName: body.datasetName,
        preprocessedDataRoot,
      }),
      queuePosition: nextQueuePosition(),
      logFile: path.join(outputDir, "train.log"),
    })
    .returning()
    .get();

  return NextResponse.json(
    { ...result, config: JSON.parse(result.config) },
    { status: 201 },
  );
}

function buildYamlConfig(
  uiConfig: TrainingConfig,
  preprocessedDataRoot: string | null,
): Record<string, unknown> {
  return {
    model: {
      model_path: uiConfig.model?.modelPath || null,
      text_encoder_path: uiConfig.model?.textEncoderPath || null,
      training_mode: uiConfig.model?.trainingMode || "lora",
      load_checkpoint: uiConfig.model?.loadCheckpoint || null,
    },
    lora: uiConfig.model?.trainingMode !== "full" ? {
      rank: uiConfig.lora?.rank ?? 48,
      alpha: uiConfig.lora?.alpha ?? 48,
      dropout: uiConfig.lora?.dropout ?? 0.05,
      target_modules: uiConfig.lora?.targetModules ?? [
        "to_k", "to_q", "to_v", "to_out.0",
        "to_gate_logits", "net.0.proj", "net.2",
      ],
    } : undefined,
    training_strategy: {
      name: uiConfig.trainingStrategy?.name || "text_to_video",
      first_frame_conditioning_p: uiConfig.trainingStrategy?.firstFrameConditioningP ?? 1.0,
      with_audio: uiConfig.trainingStrategy?.withAudio ?? true,
      audio_latents_dir: uiConfig.trainingStrategy?.audioLatentsDir || "audio_latents",
      h_flip: uiConfig.trainingStrategy?.hFlip ?? true,
      first_frame_conditioning_noise: uiConfig.trainingStrategy?.firstFrameConditioningNoise ?? 0.0,
      temporal_boundary_loss_weight: uiConfig.trainingStrategy?.temporalBoundaryLossWeight ?? 1.2,
      temporal_boundary_frames: uiConfig.trainingStrategy?.temporalBoundaryFrames ?? 3,
    },
    optimization: {
      learning_rate: uiConfig.optimization?.learningRate ?? 1e-5,
      steps: uiConfig.optimization?.steps ?? 11200,
      batch_size: uiConfig.optimization?.batchSize ?? 1,
      gradient_accumulation_steps: uiConfig.optimization?.gradientAccumulationSteps ?? 1,
      max_grad_norm: uiConfig.optimization?.maxGradNorm ?? 1.0,
      optimizer_type: uiConfig.optimization?.optimizerType || "muon",
      optimizer_params: uiConfig.optimization?.optimizerType === "muon"
        ? { adjust_lr_fn: "match_rms_adamw", weight_decay: 0.0001 }
        : undefined,
      scheduler_type: uiConfig.optimization?.schedulerType || "lambda_warmup",
      scheduler_params: {
        num_warmup_steps: uiConfig.optimization?.numWarmupSteps ?? 560,
      },
      enable_gradient_checkpointing: uiConfig.optimization?.enableGradientCheckpointing ?? true,
    },
    acceleration: {
      mixed_precision_mode: "bf16",
    },
    data: {
      preprocessed_data_root: preprocessedDataRoot || "",
      num_dataloader_workers: 4,
    },
    validation: {
      prompts: (uiConfig.validation?.prompts ?? []).filter(Boolean),
      images: (uiConfig.validation?.images ?? []).filter(Boolean),
      negative_prompt: uiConfig.validation?.negativePrompt || "worst quality, inconsistent motion, blurry, jittery, distorted",
      video_dims: uiConfig.validation?.videoDims ?? [416, 608, 241],
      frame_rate: uiConfig.validation?.frameRate ?? 24.0,
      seed: uiConfig.validation?.seed ?? 42,
      inference_steps: uiConfig.validation?.inferenceSteps ?? 30,
      interval: uiConfig.validation?.interval ?? 560,
      videos_per_prompt: uiConfig.validation?.videosPerPrompt ?? 1,
      guidance_scale: uiConfig.validation?.guidanceScale ?? 3.0,
      stg_scale: 0.0,
      stg_blocks: [28],
      stg_mode: "stg_av",
      generate_audio: uiConfig.validation?.generateAudio ?? true,
      skip_initial_validation: false,
    },
    checkpoints: {
      interval: uiConfig.checkpoints?.interval ?? 560,
      keep_last_n: uiConfig.checkpoints?.keepLastN ?? -1,
      precision: uiConfig.checkpoints?.precision || "bfloat16",
      no_resume: true,
    },
    flow_matching: {
      timestep_sampling_mode: "uniform",
      timestep_sampling_params: {},
      timestep_loss_weighting: "weighted",
    },
    tensorboard: {
      enabled: true,
    },
    hub: {
      push_to_hub: false,
      hub_model_id: null,
    },
    output_dir: uiConfig.outputDir,
    seed: uiConfig.seed ?? 42,
  };
}
