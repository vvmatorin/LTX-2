import { NextResponse } from "next/server";
import { db } from "@/db";
import { trainingDatasets } from "@/db/schema";
import { eq } from "drizzle-orm";
import fs from "fs";
import path from "path";

export async function GET() {
  const rows = db.select().from(trainingDatasets).all();
  return NextResponse.json(
    rows.map((r) => ({
      ...r,
      buckets: JSON.parse(r.buckets),
      pathExists: r.path ? fs.existsSync(path.join(r.path, ".precomputed")) : false,
    })),
  );
}

export async function POST(req: Request) {
  const body = await req.json();

  if (!body.name || !body.path || !body.buckets) {
    return NextResponse.json(
      { error: "name, path, and buckets are required" },
      { status: 400 },
    );
  }

  const missingPaths = (body.buckets as Array<{ folderPath?: string }>)
    .map((b) => b.folderPath)
    .filter((p): p is string => !!p && !fs.existsSync(p));

  if (missingPaths.length > 0) {
    return NextResponse.json(
      { error: `Bucket source path${missingPaths.length > 1 ? "s" : ""} do not exist on disk: ${missingPaths.join(", ")}` },
      { status: 400 },
    );
  }

  const result = db
    .insert(trainingDatasets)
    .values({
      name: body.name,
      path: body.path,
      buckets: JSON.stringify(body.buckets),
    })
    .returning()
    .get();

  return NextResponse.json(
    { ...result, buckets: JSON.parse(result.buckets) },
    { status: 201 },
  );
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const id = Number(searchParams.get("id"));
  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }
  db.delete(trainingDatasets).where(eq(trainingDatasets.id, id)).run();
  return NextResponse.json({ ok: true });
}
