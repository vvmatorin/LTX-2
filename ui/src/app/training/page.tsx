"use client";

import { useState, useEffect } from "react";
import type { TrainingConfig } from "@/lib/types";
import { useDatasets } from "@/hooks/useDatasets";
import { useRuns } from "@/hooks/useRuns";
import { useSettings } from "@/hooks/useSettings";
import { TrainingConfigForm } from "@/components/TrainingConfigForm";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Play, Loader2 } from "lucide-react";

function buildDefaultConfig(
  modelPath: string,
  textEncoderPath: string,
  outputDir: string,
): TrainingConfig {
  return {
    model: {
      modelPath,
      textEncoderPath,
      trainingMode: "lora",
      loadCheckpoint: null,
    },
    lora: {
      rank: 48,
      alpha: 48,
      dropout: 0.05,
      targetModules: [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "to_gate_logits",
        "net.0.proj",
        "net.2",
      ],
    },
    trainingStrategy: {
      name: "text_to_video",
      firstFrameConditioningP: 1.0,
      withAudio: true,
      audioLatentsDir: "audio_latents",
      hFlip: true,
      firstFrameConditioningNoise: 0.0,
      temporalBoundaryLossWeight: 1.2,
      temporalBoundaryFrames: 3,
    },
    optimization: {
      learningRate: 1e-5,
      steps: 11200,
      batchSize: 1,
      gradientAccumulationSteps: 1,
      maxGradNorm: 1.0,
      optimizerType: "muon",
      weightDecay: 0.0001,
      schedulerType: "lambda_warmup",
      numWarmupSteps: 560,
      enableGradientCheckpointing: true,
    },
    flowMatching: {
      timestepSamplingMode: "uniform",
      timestepLossWeighting: "weighted",
    },
    validation: {
      prompts: [],
      images: [],
      negativePrompt:
        "worst quality, inconsistent motion, blurry, jittery, distorted",
      videoDims: [416, 608, 241],
      frameRate: 24.0,
      seed: 42,
      inferenceSteps: 30,
      interval: 560,
      videosPerPrompt: 1,
      guidanceScale: 3.0,
      generateAudio: true,
      skipInitialValidation: false,
    },
    checkpoints: {
      interval: 560,
      keepLastN: -1,
      precision: "bfloat16",
    },
    outputDir: outputDir || "/tmp/ltx-training",
    seed: 42,
  };
}

export default function TrainingPage() {
  const { settings } = useSettings();
  const [config, setConfig] = useState<TrainingConfig | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [gpuMode, setGpuMode] = useState<"single" | "ddp">("single");
  const [gpuIds, setGpuIds] = useState("0");
  const [startError, setStartError] = useState<string | null>(null);

  const { datasets } = useDatasets();
  const { createRun, isCreating } = useRuns();

  useEffect(() => {
    if (settings && !config) {
      setConfig(
        buildDefaultConfig(
          settings.modelPath || "",
          settings.textEncoderPath || "",
          settings.outputDir || "/tmp/ltx-training",
        ),
      );
    }
  }, [settings, config]);

  if (!config) {
    return (
      <div className="flex items-center justify-center p-12">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const activeDataset = selectedDataset || datasets[0]?.name || "";

  const handleStartTraining = async () => {
    setStartError(null);
    const outputName = config.outputDir.split("/").pop() || "training-run";
    try {
      await createRun({
        name: outputName,
        config: config as unknown as Record<string, unknown>,
        gpuMode,
        gpuIds,
        datasetName: activeDataset,
      });
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="space-y-6 p-3 md:p-4">
      <div>
        <h1 className="title-gradient page-title">Training</h1>
        <p className="page-subtitle mt-2.5">
          Configure and launch a training run
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_300px]">
        <div className="space-y-6">
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-lg">Run Setup</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label className="text-xs">Training Dataset</Label>
                  <Select
                    value={activeDataset}
                    onValueChange={(v) => v && setSelectedDataset(v)}
                  >
                    <SelectTrigger className="w-full text-xs">
                      <SelectValue placeholder="Select a dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets.map((ds) => (
                        <SelectItem key={ds.name} value={ds.name}>
                          {ds.name} ({ds.buckets.length} buckets)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">GPU Configuration</Label>
                  <div className="flex gap-2">
                    <Select
                      value={gpuMode}
                      onValueChange={(v) => {
                        if (!v) return;
                        const mode = v as "single" | "ddp";
                        setGpuMode(mode);
                        setGpuIds(mode === "single" ? "0" : "0,1");
                      }}
                    >
                      <SelectTrigger className="text-xs w-[140px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="single">Single GPU</SelectItem>
                        <SelectItem value="ddp">Multi-GPU (DDP)</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select
                      value={gpuIds}
                      onValueChange={(v) => v && setGpuIds(v)}
                    >
                      <SelectTrigger className="text-xs flex-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {gpuMode === "single" ? (
                          <>
                            <SelectItem value="0">GPU 0</SelectItem>
                            <SelectItem value="1">GPU 1</SelectItem>
                            <SelectItem value="2">GPU 2</SelectItem>
                            <SelectItem value="3">GPU 3</SelectItem>
                          </>
                        ) : (
                          <>
                            <SelectItem value="0,1">GPU 0, 1</SelectItem>
                            <SelectItem value="0,1,2">GPU 0, 1, 2</SelectItem>
                            <SelectItem value="0,1,2,3">
                              GPU 0, 1, 2, 3
                            </SelectItem>
                          </>
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <TrainingConfigForm config={config} onChange={setConfig} />
        </div>

        <div className="space-y-4">
          <Card className="sticky top-6">
            <CardHeader className="py-3">
              <CardTitle className="text-lg">Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Dataset</span>
                <Badge variant="outline" className="text-[10px]">
                  {activeDataset || "—"}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Mode</span>
                <Badge variant="secondary" className="text-[10px]">
                  {config.model.trainingMode === "full" ? "Full" : "LoRA"}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">LoRA Rank</span>
                <span>{config.lora.rank}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Steps</span>
                <span>{config.optimization.steps.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">LR</span>
                <span className="font-mono">
                  {config.optimization.learningRate}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Optimizer</span>
                <span>{config.optimization.optimizerType}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">GPU</span>
                <span>
                  {gpuMode === "ddp" ? `DDP (${gpuIds})` : `GPU ${gpuIds}`}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Audio</span>
                <Badge
                  variant={
                    config.trainingStrategy.withAudio ? "default" : "secondary"
                  }
                  className="text-[10px]"
                >
                  {config.trainingStrategy.withAudio ? "Yes" : "No"}
                </Badge>
              </div>

              {startError && (
                <p className="text-xs text-destructive">{startError}</p>
              )}

              <Button
                className="w-full"
                onClick={handleStartTraining}
                disabled={isCreating || !activeDataset}
              >
                <Play className="mr-1.5 h-4 w-4" />
                {isCreating ? "Starting..." : "Start Training"}
              </Button>

              {!activeDataset && (
                <p className="text-[10px] text-center text-muted-foreground">
                  Select a dataset to enable training
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
