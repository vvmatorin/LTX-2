import { NextResponse } from "next/server";
import { db } from "@/db";
import { jobs, sourceFolders } from "@/db/schema";
import { nextQueuePosition } from "@/db/queries";
import { eq, desc, and, type SQL } from "drizzle-orm";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";

const execFileAsync = promisify(execFile);

async function scanBuckets(
  folderPath: string,
  resolution: number,
): Promise<Array<{ key: string; fileCount: number }>> {
  const bucketScript = path.resolve(
    /* turbopackIgnore: true */ process.cwd(),
    "scripts",
    "bucket_sizes.py",
  );

  const { stdout } = await execFileAsync(
    "python3",
    [bucketScript, folderPath, "--resolution", String(resolution)],
    { timeout: 60_000 },
  );

  const result = JSON.parse(stdout) as {
    resolutions: Record<string, Array<{ key: string; fileCount: number }>>;
  };

  return result.resolutions?.[String(resolution)] || [];
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const status = searchParams.get("status");
  const type = searchParams.get("type");

  type JobStatus = typeof jobs.status.enumValues[number];
  type JobType = typeof jobs.type.enumValues[number];

  const conditions: SQL[] = [];
  if (status) conditions.push(eq(jobs.status, status as JobStatus));
  if (type) conditions.push(eq(jobs.type, type as JobType));

  let query = db.select().from(jobs).orderBy(desc(jobs.createdAt));
  if (conditions.length > 0) {
    query = query.where(and(...conditions)) as typeof query;
  }
  const rows = query.all();

  return NextResponse.json(
    rows.map((r) => {
      const config = JSON.parse(r.config) as Record<string, unknown>;
      let outputExists: boolean | undefined;
      if (r.type === "preprocess" && r.status === "completed") {
        const datasetPath = config.datasetPath as string | undefined;
        outputExists = datasetPath ? fs.existsSync(datasetPath) : false;
      }
      return {
        ...r,
        config,
        ...(outputExists !== undefined ? { outputExists } : {}),
      };
    }),
  );
}

export async function POST(req: Request) {
  const body = await req.json();
  const config: Record<string, unknown> = { ...(body.config || {}) };

  if (body.type === "preprocess" && config.folderId) {
    const folder = db
      .select()
      .from(sourceFolders)
      .where(eq(sourceFolders.id, config.folderId as number))
      .get();

    if (folder) {
      const datasetFilename =
        (config.datasetFilename as string) || "dataset.json";
      config.datasetPath = path.join(folder.path, datasetFilename);

      const resolution = config.resolution as number;
      const frameCounts = (config.frameCounts as number[]) || [];

      if (resolution && frameCounts.length > 0) {
        try {
          const buckets = await scanBuckets(folder.path, resolution);

          const bucketStrings: string[] = [];
          for (const bucket of buckets) {
            for (const fc of frameCounts) {
              bucketStrings.push(`${bucket.key}x${fc}`);
            }
          }

          if (bucketStrings.length > 0) {
            config.resolutionBuckets = bucketStrings.join(";");
          }
        } catch (err) {
          console.error("Bucket scan failed, falling back to square:", err);
        }

        if (!config.resolutionBuckets) {
          config.resolutionBuckets = frameCounts
            .map((fc) => `${resolution}x${resolution}x${fc}`)
            .join(";");
        }
      }
    }
  }

  const result = db
    .insert(jobs)
    .values({
      type: body.type,
      name: body.name,
      status: "queued",
      config: JSON.stringify(config),
      queuePosition: nextQueuePosition(),
    })
    .returning()
    .get();

  return NextResponse.json(
    { ...result, config: JSON.parse(result.config) },
    { status: 201 },
  );
}
