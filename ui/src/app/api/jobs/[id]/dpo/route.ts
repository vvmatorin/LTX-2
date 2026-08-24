import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { safeId, toErrorMessage } from '@/lib/utils';
import { LABELS_FILENAME, PENDING_FILENAME, getJobDpoRoot, listRounds, validateChoices } from '@/lib/dpoServer';
import type { DpoChoice, DpoManifest } from '@/lib/types';

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id: idStr } = await params;
  const id = safeId(idStr);
  if (!id) return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });

  const root = getJobDpoRoot(id);
  if ('error' in root) return NextResponse.json({ error: root.error }, { status: root.status });

  return NextResponse.json({ rounds: listRounds(root.dpoRoot) });
}

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id: idStr } = await params;
  const id = safeId(idStr);
  if (!id) return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });

  const root = getJobDpoRoot(id);
  if ('error' in root) return NextResponse.json({ error: root.error }, { status: root.status });

  let body: { step?: number; choices?: DpoChoice[] };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  if (typeof body.step !== 'number' || !Array.isArray(body.choices)) {
    return NextResponse.json({ error: 'Body must contain step and choices' }, { status: 400 });
  }

  const roundDir = path.join(root.dpoRoot, `round_${String(body.step).padStart(6, '0')}`);
  const pendingPath = path.join(roundDir, PENDING_FILENAME);
  const labelsPath = path.join(roundDir, LABELS_FILENAME);

  if (!fs.existsSync(pendingPath)) {
    return NextResponse.json({ error: `No labeling round found for step ${body.step}` }, { status: 404 });
  }
  // Labels are immutable: one submission per round.
  if (fs.existsSync(labelsPath)) {
    return NextResponse.json({ error: 'Labels for this round were already submitted' }, { status: 409 });
  }

  let manifest: DpoManifest;
  try {
    manifest = JSON.parse(fs.readFileSync(pendingPath, 'utf-8'));
  } catch (err) {
    return NextResponse.json({ error: `Failed to read round manifest: ${toErrorMessage(err)}` }, { status: 500 });
  }

  const validationError = validateChoices(manifest, body.choices);
  if (validationError) {
    return NextResponse.json({ error: validationError }, { status: 400 });
  }

  const labels = {
    submitted_at: new Date().toISOString(),
    choices: body.choices.map(c => ({
      index: c.index,
      best: c.skipped ? null : c.best,
      worst: c.skipped ? null : c.worst,
      skipped: !!c.skipped,
    })),
  };

  // tmp+rename so the polling trainer never reads a partial file.
  try {
    const tmpPath = labelsPath + '.tmp';
    fs.writeFileSync(tmpPath, JSON.stringify(labels, null, 2), 'utf-8');
    fs.renameSync(tmpPath, labelsPath);
  } catch (err) {
    return NextResponse.json({ error: `Failed to write labels: ${toErrorMessage(err)}` }, { status: 500 });
  }

  return NextResponse.json({ ok: true, labels }, { status: 201 });
}
