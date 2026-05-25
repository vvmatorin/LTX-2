import { NextResponse } from 'next/server';
import { db } from '@/db';
import { sourceFolders } from '@/db/schema';
import { eq } from 'drizzle-orm';
import { safeId } from '@/lib/utils';
import fs from 'fs';
import path from 'path';

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png']);
const VIDEO_EXTS = new Set(['.mp4', '.mov', '.avi', '.mkv', '.webm']);

function scanFolderMeta(folderPath: string): {
  fileCount: number;
  mediaType: 'images' | 'videos' | 'mixed';
} {
  if (!fs.existsSync(folderPath)) {
    return { fileCount: 0, mediaType: 'videos' };
  }

  const entries = fs.readdirSync(folderPath);
  let images = 0;
  let videos = 0;

  for (const entry of entries) {
    const ext = path.extname(entry).toLowerCase();
    if (IMAGE_EXTS.has(ext)) images++;
    else if (VIDEO_EXTS.has(ext)) videos++;
  }

  const mediaType: 'images' | 'videos' | 'mixed' =
    images > 0 && videos > 0 ? 'mixed' : images > 0 ? 'images' : 'videos';

  return { fileCount: images + videos, mediaType };
}

export async function GET() {
  const rows = db.select().from(sourceFolders).all();
  return NextResponse.json(rows);
}

export async function POST(req: Request) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  const folderPath = body.path as string | undefined;

  if (!folderPath) {
    return NextResponse.json({ error: 'path is required' }, { status: 400 });
  }

  const parts = folderPath.split('/').filter(Boolean);
  const name = (body.name as string) || parts[parts.length - 1] || 'folder';

  const meta = scanFolderMeta(folderPath);

  const result = db
    .insert(sourceFolders)
    .values({
      path: folderPath,
      name,
      mediaType: meta.mediaType,
      fileCount: meta.fileCount,
    })
    .returning()
    .get();

  return NextResponse.json(result, { status: 201 });
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const id = safeId(searchParams.get('id'));
  if (!id) {
    return NextResponse.json({ error: 'id is required' }, { status: 400 });
  }
  db.delete(sourceFolders).where(eq(sourceFolders.id, id)).run();
  return NextResponse.json({ ok: true });
}
