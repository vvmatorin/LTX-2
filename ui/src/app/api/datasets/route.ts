import { NextResponse } from 'next/server';
import { db } from '@/db';
import { jobs, trainingDatasets } from '@/db/schema';
import { nextQueuePosition } from '@/db/queries';
import { eq, inArray } from 'drizzle-orm';
import { parseJobConfig, safeId } from '@/lib/utils';
import fs from 'fs';
import path from 'path';
import { MODEL_STREAMS, type DatasetBucket } from '@/lib/types';

export async function GET() {
  const rows = db.select().from(trainingDatasets).all();

  const activeBuilds = db
    .select({ name: jobs.name, status: jobs.status })
    .from(jobs)
    .where(inArray(jobs.status, ['queued', 'running']))
    .all()
    .filter(j => j.name.startsWith('Dataset: '));

  const buildStatusByDataset = new Map(activeBuilds.map(j => [j.name.slice('Dataset: '.length), j.status]));

  return NextResponse.json(
    rows.map(r => {
      const streams = r.path
        ? MODEL_STREAMS.filter(stream => fs.existsSync(path.join(r.path, '.precomputed', stream)))
        : [];
      return {
        ...r,
        buckets: parseJobConfig(r.buckets) ?? [],
        pathExists: streams.length > 0,
        streams,
        buildStatus: buildStatusByDataset.get(r.path) ?? null,
      };
    }),
  );
}

export async function POST(req: Request) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  if (!body.name || !body.path || !body.buckets) {
    return NextResponse.json({ error: 'name, path, and buckets are required' }, { status: 400 });
  }

  const buckets = body.buckets as DatasetBucket[];

  const missingPaths = buckets.map(b => b.folderPath).filter((p): p is string => !!p && !fs.existsSync(p));

  if (missingPaths.length > 0) {
    return NextResponse.json(
      {
        error: `Bucket source path${missingPaths.length > 1 ? 's' : ''} do not exist on disk: ${missingPaths.join(', ')}`,
      },
      { status: 400 },
    );
  }

  const sources = buckets
    .filter(b => Boolean(b.folderPath) && Boolean(b.stream))
    .map(b => ({ dir: b.folderPath, stream: b.stream }));

  if (sources.length === 0) {
    return NextResponse.json({ error: 'Dataset has no source buckets with folder paths' }, { status: 400 });
  }

  const result = db
    .insert(trainingDatasets)
    .values({
      name: body.name as string,
      path: body.path as string,
      buckets: JSON.stringify(buckets),
    })
    .returning()
    .get();

  db.insert(jobs)
    .values({
      type: 'merge',
      name: `Dataset: ${result.path}`,
      status: 'queued',
      config: JSON.stringify({ sources, destDir: result.path }),
      queuePosition: nextQueuePosition(),
    })
    .run();

  return NextResponse.json({ ...result, buckets: parseJobConfig(result.buckets) ?? [] }, { status: 201 });
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const id = safeId(searchParams.get('id'));
  if (!id) {
    return NextResponse.json({ error: 'id is required' }, { status: 400 });
  }
  db.delete(trainingDatasets).where(eq(trainingDatasets.id, id)).run();
  return NextResponse.json({ ok: true });
}
