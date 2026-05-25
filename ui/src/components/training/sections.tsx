"use client";

import type { TrainingConfig } from "@/lib/types";
import { Label } from "@/components/ui/label";
import {
  Section,
  NumberField,
  TextField,
  SelectField,
  SwitchField,
  TagInput,
  ListInput,
  VideoDimsField,
} from "./fields";

type UpdateFn = (
  section: keyof TrainingConfig,
  patch: Record<string, unknown>,
) => void;

interface SectionProps {
  config: TrainingConfig;
  update: UpdateFn;
}

interface GeneralSectionProps {
  config: TrainingConfig;
  onChange: (config: TrainingConfig) => void;
}

export function ModelSection({ config, update }: SectionProps) {
  return (
    <Section title="Model">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <TextField
            label="Checkpoint Path"
            value={config.model.modelPath}
            onChange={(v) => update("model", { modelPath: v })}
            mono
          />
        </div>
        <div className="sm:col-span-2">
          <TextField
            label="Text Encoder Path"
            value={config.model.textEncoderPath}
            onChange={(v) => update("model", { textEncoderPath: v })}
            mono
          />
        </div>
        <div className="sm:col-span-2">
          <TextField
            label="Load Checkpoint (optional)"
            value={config.model.loadCheckpoint || ""}
            onChange={(v) => update("model", { loadCheckpoint: v || null })}
            placeholder="None — resume from a previous checkpoint path"
            mono
          />
        </div>
      </div>
    </Section>
  );
}

export function LoraSection({ config, update }: SectionProps) {
  return (
    <Section title="LoRA">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <NumberField
          label="Rank"
          value={config.lora.rank}
          onChange={(v) => update("lora", { rank: v })}
        />
        <NumberField
          label="Alpha"
          value={config.lora.alpha}
          onChange={(v) => update("lora", { alpha: v })}
        />
        <NumberField
          label="Dropout"
          value={config.lora.dropout}
          onChange={(v) => update("lora", { dropout: v })}
          step={0.01}
        />
      </div>
      <div className="space-y-2">
        <Label className="text-xs">Target Modules</Label>
        <TagInput
          value={config.lora.targetModules}
          onChange={(v) => update("lora", { targetModules: v })}
        />
      </div>
    </Section>
  );
}

export function StrategySection({ config, update }: SectionProps) {
  return (
    <Section title="Training Strategy">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <NumberField
          label="First Frame Cond. P"
          value={config.trainingStrategy.firstFrameConditioningP}
          onChange={(v) =>
            update("trainingStrategy", { firstFrameConditioningP: v })
          }
          step={0.1}
        />
        <NumberField
          label="Boundary Loss Weight"
          value={config.trainingStrategy.temporalBoundaryLossWeight}
          onChange={(v) =>
            update("trainingStrategy", { temporalBoundaryLossWeight: v })
          }
          step={0.1}
        />
        <NumberField
          label="Boundary Frames"
          value={config.trainingStrategy.temporalBoundaryFrames}
          onChange={(v) =>
            update("trainingStrategy", { temporalBoundaryFrames: v })
          }
        />
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        <SwitchField
          label="Audio"
          checked={config.trainingStrategy.withAudio}
          onChange={(v) => update("trainingStrategy", { withAudio: v })}
        />
        <SwitchField
          label="H-Flip Augmentation"
          checked={config.trainingStrategy.hFlip}
          onChange={(v) => update("trainingStrategy", { hFlip: v })}
        />
      </div>
    </Section>
  );
}

export function OptimizationSection({ config, update }: SectionProps) {
  return (
    <Section title="Optimization">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <NumberField
          label="Learning Rate"
          value={config.optimization.learningRate}
          onChange={(v) => update("optimization", { learningRate: v })}
          step={0.000001}
          mono
        />
        <NumberField
          label="Steps"
          value={config.optimization.steps}
          onChange={(v) => update("optimization", { steps: v })}
        />
        <NumberField
          label="Batch Size"
          value={config.optimization.batchSize}
          onChange={(v) => update("optimization", { batchSize: v })}
        />
        <NumberField
          label="Grad Accumulation"
          value={config.optimization.gradientAccumulationSteps}
          onChange={(v) =>
            update("optimization", { gradientAccumulationSteps: v })
          }
        />
        <NumberField
          label="Max Grad Norm"
          value={config.optimization.maxGradNorm}
          onChange={(v) => update("optimization", { maxGradNorm: v })}
          step={0.1}
        />
        <NumberField
          label="Warmup Steps"
          value={config.optimization.numWarmupSteps}
          onChange={(v) => update("optimization", { numWarmupSteps: v })}
        />
        <SelectField
          label="Optimizer"
          value={config.optimization.optimizerType}
          onChange={(v) => update("optimization", { optimizerType: v })}
          options={[
            { value: "muon", label: "Muon" },
            { value: "adamw", label: "AdamW" },
            { value: "adamw8bit", label: "AdamW 8-bit" },
          ]}
        />
        <SelectField
          label="Scheduler"
          value={config.optimization.schedulerType}
          onChange={(v) => update("optimization", { schedulerType: v })}
          options={[
            { value: "cosine", label: "Cosine" },
            { value: "constant", label: "Constant" },
            { value: "lambda_warmup", label: "Lambda Warmup" },
          ]}
        />
      </div>
      <SwitchField
        label="Gradient Checkpointing"
        checked={config.optimization.enableGradientCheckpointing}
        onChange={(v) =>
          update("optimization", { enableGradientCheckpointing: v })
        }
      />
    </Section>
  );
}

export function ValidationSection({ config, update }: SectionProps) {
  return (
    <Section title="Validation" defaultOpen={false}>
      <div className="space-y-4">
        <div className="space-y-2">
          <Label className="text-xs">Prompts (one per line)</Label>
          <ListInput
            value={config.validation.prompts}
            onChange={(v) => update("validation", { prompts: v })}
            placeholder="A serene mountain lake at sunrise..."
          />
        </div>
        <div className="space-y-2">
          <Label className="text-xs">
            Conditioning Images (one path per line)
          </Label>
          <ListInput
            value={config.validation.images}
            onChange={(v) => update("validation", { images: v })}
            placeholder="/path/to/image.jpeg"
          />
        </div>
        <TextField
          label="Negative Prompt"
          value={config.validation.negativePrompt}
          onChange={(v) => update("validation", { negativePrompt: v })}
        />
        <VideoDimsField
          value={config.validation.videoDims}
          onChange={(v) => update("validation", { videoDims: v })}
        />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <NumberField
            label="Inference Steps"
            value={config.validation.inferenceSteps}
            onChange={(v) => update("validation", { inferenceSteps: v })}
          />
          <NumberField
            label="Interval"
            value={config.validation.interval}
            onChange={(v) => update("validation", { interval: v })}
          />
          <NumberField
            label="Guidance Scale"
            value={config.validation.guidanceScale}
            onChange={(v) => update("validation", { guidanceScale: v })}
            step={0.5}
          />
          <NumberField
            label="Seed"
            value={config.validation.seed}
            onChange={(v) => update("validation", { seed: v })}
          />
        </div>
        <SwitchField
          label="Generate Audio"
          checked={config.validation.generateAudio}
          onChange={(v) => update("validation", { generateAudio: v })}
        />
      </div>
    </Section>
  );
}

export function CheckpointsSection({ config, update }: SectionProps) {
  return (
    <Section title="Checkpoints" defaultOpen={false}>
      <div className="grid grid-cols-3 gap-4">
        <NumberField
          label="Save Interval"
          value={config.checkpoints.interval}
          onChange={(v) => update("checkpoints", { interval: v })}
        />
        <NumberField
          label="Keep Last N (-1 = all)"
          value={config.checkpoints.keepLastN}
          onChange={(v) => update("checkpoints", { keepLastN: v })}
        />
        <SelectField
          label="Precision"
          value={config.checkpoints.precision}
          onChange={(v) => update("checkpoints", { precision: v })}
          options={[
            { value: "bfloat16", label: "bfloat16" },
            { value: "float16", label: "float16" },
            { value: "float32", label: "float32" },
          ]}
        />
      </div>
    </Section>
  );
}

export function LoggingSection({ config, update }: SectionProps) {
  return (
    <Section title="Logging (TrackIO)" defaultOpen={false}>
      <SwitchField
        label="Enable TrackIO Logging"
        checked={config.trackio.enabled}
        onChange={(v) => update("trackio", { enabled: v })}
      />
      {config.trackio.enabled && (
        <div className="grid grid-cols-2 gap-4">
          <TextField
            label="Project"
            value={config.trackio.project}
            onChange={(v) => update("trackio", { project: v })}
          />
          <TextField
            label="Space ID"
            value={config.trackio.spaceId}
            onChange={(v) => update("trackio", { spaceId: v })}
          />
          <div className="col-span-2">
            <SwitchField
              label="Log Validation Videos"
              checked={config.trackio.logValidationVideos}
              onChange={(v) => update("trackio", { logValidationVideos: v })}
            />
          </div>
        </div>
      )}
    </Section>
  );
}

export function GeneralSection({ config, onChange }: GeneralSectionProps) {
  return (
    <Section title="General" defaultOpen={false}>
      <div className="grid grid-cols-2 gap-4">
        <TextField
          label="Output Directory"
          value={config.outputDir}
          onChange={(v) => onChange({ ...config, outputDir: v })}
          mono
        />
        <NumberField
          label="Seed"
          value={config.seed}
          onChange={(v) => onChange({ ...config, seed: v })}
        />
      </div>
    </Section>
  );
}
