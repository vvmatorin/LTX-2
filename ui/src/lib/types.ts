export interface SourceFolder {
  id: number;
  path: string;
  name: string;
  mediaType: 'images' | 'videos' | 'mixed' | 'audio';
  fileCount: number;
  createdAt: string;
}

export interface ProcessingJob {
  id: number;
  type: 'preprocess' | 'merge' | 'training';
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  config: Record<string, unknown>;
  queuePosition: number;
  pid: number | null;
  logFile: string | null;
  stopRequested: boolean;
  progress: number | null;
  error: string | null;
  startedAt: string | null;
  completedAt: string | null;
  createdAt: string;
  /** Present only for completed preprocess jobs: whether the .precomputed output directory exists on disk */
  outputExists?: boolean;
}

export type ModelStream = 'ltx-2.3' | 'ltx-2.5';

export const MODEL_STREAMS: ModelStream[] = ['ltx-2.3', 'ltx-2.5'];

export const MODEL_STREAM_LABELS: Record<ModelStream, string> = {
  'ltx-2.3': 'LTX-2.3',
  'ltx-2.5': 'LTX-2.5',
};

export interface StreamModelPaths {
  modelPath: string;
  textEncoderPath: string;
  videoVaePath: string;
  audioVaePath: string;
}

export type StreamPathSettings = Pick<
  AppSettings,
  | 'ltx23ModelPath'
  | 'ltx23TextEncoderPath'
  | 'ltx25ModelPath'
  | 'ltx25TextEncoderPath'
  | 'ltx25VideoVaePath'
  | 'ltx25AudioVaePath'
>;

export function streamPaths(settings: StreamPathSettings, stream: ModelStream): StreamModelPaths {
  if (stream === 'ltx-2.5') {
    return {
      modelPath: settings.ltx25ModelPath || '',
      textEncoderPath: settings.ltx25TextEncoderPath || '',
      videoVaePath: settings.ltx25VideoVaePath || '',
      audioVaePath: settings.ltx25AudioVaePath || '',
    };
  }
  return {
    modelPath: settings.ltx23ModelPath || '',
    textEncoderPath: settings.ltx23TextEncoderPath || '',
    videoVaePath: '',
    audioVaePath: '',
  };
}

export function pickDefaultStream(settings: AppSettings): ModelStream {
  if (settings.ltx25ModelPath) return 'ltx-2.5';
  if (settings.ltx23ModelPath) return 'ltx-2.3';
  return 'ltx-2.5';
}

export interface TrainingDataset {
  id: number;
  name: string;
  path: string;
  modelStream: ModelStream;
  buckets: DatasetBucket[];
  pathExists: boolean;
  buildStatus: 'queued' | 'running' | null;
  createdAt: string;
}

export interface DatasetBucket {
  folderName: string;
  folderPath: string;
  jobId: number;
  stream: ModelStream;
  resolution: number;
  frameCount: number;
  bucketKeys: string[];
  hasAudio: boolean;
  hasHFlip: boolean;
  isAudioOnly?: boolean;
}

export interface TrainingConfig {
  model: {
    modelStream: ModelStream;
    modelPath: string;
    textEncoderPath: string;
    videoVaePath: string;
    audioVaePath: string;
    trainingMode: 'lora' | 'full';
    loadCheckpoint: string | null;
  };
  lora: {
    rank: number;
    alpha: number;
    dropout: number;
    targetModules: string[];
    freezeExtraModules: boolean;
  };
  trainingStrategy: {
    name: 'text_to_video' | 'video_to_video';
    firstFrameConditioningP: number;
    // text_to_video only
    withAudio: boolean;
    audioLatentsDir: string;
    audioLossWeight?: number;
    hFlip: boolean;
    // shared
    temporalBoundaryLossWeight: number;
    temporalBoundaryFrames: number;
    captionDropoutP: number;
  };
  optimization: {
    learningRate: number;
    steps: number;
    batchSize: number;
    gradientAccumulationSteps: number;
    maxGradNorm: number;
    optimizerType: string;
    weightDecay: number;
    optimizerParams?: Record<string, unknown>;
    schedulerType: string;
    numWarmupSteps: number;
    enableGradientCheckpointing: boolean;
    gradientCheckpointingRatio: number;
    weightNoise: {
      mode: 'none' | 'relative' | 'absolute';
      sigma: number;
    };
  };
  flowMatching: {
    timestepSamplingMode: 'uniform' | 'shifted_logit_normal';
    timestepLossWeighting: 'none' | 'bell' | 'weighted';
    timestepLossWeightingGamma?: number;
  };
  data?: {
    preprocessedDataRoot?: string;
  };
  validation: {
    prompts: string[];
    images: string[];
    referenceVideos: string[];
    referenceDownscaleFactor: number;
    includeReferenceInOutput: boolean;
    negativePrompt: string;
    videoDims: [number, number, number];
    frameRate: number;
    seed: number;
    inferenceSteps: number;
    interval: number;
    guidanceScale: number;
    generateAudio: boolean;
    skipInitialValidation: boolean;
  };
  dpo: {
    enabled: boolean;
    samplesFile: string;
    numSamples: number;
    numSeeds: number;
    interval: number;
    repeats: number;
    beta: number;
    generateAudio: boolean;
    audioLossWeight: number;
  };
  checkpoints: {
    interval: number;
    keepLastN: number;
    precision: string;
    saveTrainingState: 'full' | 'minimal' | 'off';
    resume: boolean;
  };
  outputDir: string;
  seed: number;
}

export interface DpoManifestSample {
  index: number;
  stem: string;
  prompt: string;
  image: boolean;
  seeds: number[];
  videos: string[];
  latents: string[];
}

export interface DpoManifest {
  step: number;
  created_at?: string;
  num_seeds: number;
  video_dims?: [number, number, number];
  with_audio?: boolean;
  samples: DpoManifestSample[];
}

export interface DpoChoice {
  index: number;
  best?: number | null;
  worst?: number | null;
  skipped?: boolean;
}

export interface DpoLabels {
  submitted_at: string;
  choices: DpoChoice[];
}

export interface DpoRound {
  step: number;
  dir: string;
  pending: DpoManifest;
  labels: DpoLabels | null;
}

export interface AppSettings {
  ltx23ModelPath: string;
  ltx23TextEncoderPath: string;
  ltx25ModelPath: string;
  ltx25TextEncoderPath: string;
  ltx25VideoVaePath: string;
  ltx25AudioVaePath: string;
  outputDir: string;
  datasetDir: string;
  scriptsDir: string;
}

export type ResolutionOption = 512 | 768 | 1024 | 1280 | 1440;
export type FrameCountOption = 1 | 25 | 49 | 73 | 121 | 145;

export const RESOLUTION_OPTIONS: ResolutionOption[] = [512, 768, 1024, 1280, 1440];
export const FRAME_COUNT_OPTIONS: FrameCountOption[] = [1, 25, 49, 73, 121, 145];
