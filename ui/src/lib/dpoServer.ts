import fs from 'fs';
import path from 'path';
import { db } from '@/db';
import { jobs } from '@/db/schema';
import { eq } from 'drizzle-orm';
import { parseJobConfig } from '@/lib/utils';
import type { DpoChoice, DpoLabels, DpoManifest, DpoRound } from '@/lib/types';

/**
 * Server-side helpers for the Live-DPO labeling handshake.
 *
 * The trainer writes rounds into `<outputDir>/dpo/round_<step>/`: generated videos,
 * latents, and `pending.json` (written last — its presence marks a complete round).
 * The UI writes `labels.json` exactly once; the trainer polls for it to resume.
 */

export const PENDING_FILENAME = 'pending.json';
export const LABELS_FILENAME = 'labels.json';

const ROUND_DIR_RE = /^round_\d{6}$/;

export function getJobDpoRoot(jobId: number): { dpoRoot: string } | { error: string; status: number } {
  const job = db.select().from(jobs).where(eq(jobs.id, jobId)).get();
  if (!job) return { error: 'Job not found', status: 404 };
  if (job.type !== 'training') return { error: 'Not a training job', status: 400 };

  const config = parseJobConfig(job.config);
  const outputDir = (config?.outputDir as string) || '';
  if (!outputDir) return { error: 'Job has no output directory', status: 400 };

  return { dpoRoot: path.join(outputDir, 'dpo') };
}

function readJson<T>(filePath: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8')) as T;
  } catch {
    return null;
  }
}

/** All complete rounds (those with a parseable pending.json), newest first. */
export function listRounds(dpoRoot: string): DpoRound[] {
  let entries: string[];
  try {
    entries = fs.readdirSync(dpoRoot);
  } catch {
    return [];
  }

  const rounds: DpoRound[] = [];
  for (const dir of entries) {
    if (!ROUND_DIR_RE.test(dir)) continue;
    const pending = readJson<DpoManifest>(path.join(dpoRoot, dir, PENDING_FILENAME));
    if (!pending) continue;
    rounds.push({
      step: pending.step,
      dir,
      pending,
      labels: readJson<DpoLabels>(path.join(dpoRoot, dir, LABELS_FILENAME)),
    });
  }
  return rounds.sort((a, b) => b.step - a.step);
}

/** Validate submitted choices against a round manifest. Returns an error message or null. */
export function validateChoices(manifest: DpoManifest, choices: DpoChoice[]): string | null {
  const sampleIndices = new Set(manifest.samples.map(s => s.index));
  const seen = new Set<number>();

  for (const choice of choices) {
    if (!sampleIndices.has(choice.index)) return `Unknown sample index ${choice.index}`;
    if (seen.has(choice.index)) return `Duplicate choice for sample ${choice.index}`;
    seen.add(choice.index);

    if (choice.skipped) continue;
    const { best, worst } = choice;
    if (
      typeof best !== 'number' ||
      typeof worst !== 'number' ||
      !Number.isInteger(best) ||
      !Number.isInteger(worst) ||
      best < 0 ||
      worst < 0 ||
      best >= manifest.num_seeds ||
      worst >= manifest.num_seeds
    ) {
      return `Sample ${choice.index}: best and worst must be seed indices in [0, ${manifest.num_seeds})`;
    }
    if (best === worst) return `Sample ${choice.index}: best and worst must differ`;
  }

  if (seen.size !== sampleIndices.size) {
    return 'Every sample must be labeled or explicitly skipped';
  }
  return null;
}
