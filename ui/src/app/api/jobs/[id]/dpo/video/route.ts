import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { safeId } from '@/lib/utils';
import { getJobDpoRoot } from '@/lib/dpoServer';

/**
 * Streams a labeling-round video (`?path=round_000560/sample_00_seed_0.mp4`) with HTTP
 * Range support so <video> elements can seek. The path is contained to the job's dpo/
 * directory and restricted to .mp4 files.
 */
export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id: idStr } = await params;
  const id = safeId(idStr);
  if (!id) return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });

  const root = getJobDpoRoot(id);
  if ('error' in root) return NextResponse.json({ error: root.error }, { status: root.status });

  const rel = new URL(req.url).searchParams.get('path') || '';
  const resolvedRoot = path.resolve(root.dpoRoot);
  const filePath = path.resolve(resolvedRoot, rel);
  if (!filePath.startsWith(resolvedRoot + path.sep) || !filePath.endsWith('.mp4')) {
    return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
  }

  let stat: fs.Stats;
  try {
    stat = fs.statSync(filePath);
    if (!stat.isFile()) throw new Error('not a file');
  } catch {
    return NextResponse.json({ error: 'Video not found' }, { status: 404 });
  }

  const baseHeaders: Record<string, string> = {
    'Content-Type': 'video/mp4',
    'Accept-Ranges': 'bytes',
    'Cache-Control': 'no-store',
  };

  const toBody = (stream: fs.ReadStream) => Readable.toWeb(stream) as ReadableStream<Uint8Array>;

  const range = req.headers.get('range');
  const match = range?.match(/^bytes=(\d*)-(\d*)$/);
  if (match && (match[1] !== '' || match[2] !== '')) {
    const start = match[1] !== '' ? parseInt(match[1], 10) : Math.max(0, stat.size - parseInt(match[2], 10));
    const end = match[1] !== '' && match[2] !== '' ? Math.min(parseInt(match[2], 10), stat.size - 1) : stat.size - 1;
    if (start >= stat.size || start > end) {
      return new NextResponse(null, { status: 416, headers: { 'Content-Range': `bytes */${stat.size}` } });
    }
    return new NextResponse(toBody(fs.createReadStream(filePath, { start, end })), {
      status: 206,
      headers: {
        ...baseHeaders,
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Content-Length': String(end - start + 1),
      },
    });
  }

  return new NextResponse(toBody(fs.createReadStream(filePath)), {
    status: 200,
    headers: { ...baseHeaders, 'Content-Length': String(stat.size) },
  });
}
