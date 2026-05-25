import { NextResponse } from "next/server";
import { db } from "@/db";
import { jobs } from "@/db/schema";
import { eq } from "drizzle-orm";

export async function POST(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id: idStr } = await params;
  const id = Number(idStr);

  const job = db.select().from(jobs).where(eq(jobs.id, id)).get();
  if (!job) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }

  if (job.status === "running") {
    db.update(jobs)
      .set({ stopRequested: true })
      .where(eq(jobs.id, id))
      .run();

    if (job.pid) {
      try {
        process.kill(job.pid, "SIGINT");
      } catch {
        // Process may have already exited
      }
    }

    db.update(jobs)
      .set({
        status: "cancelled",
        completedAt: new Date().toISOString(),
      })
      .where(eq(jobs.id, id))
      .run();
  } else if (job.status === "queued") {
    db.update(jobs)
      .set({
        status: "cancelled",
        completedAt: new Date().toISOString(),
      })
      .where(eq(jobs.id, id))
      .run();
  }

  return NextResponse.json({ ok: true });
}
