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

  if (job.status !== "running" && job.status !== "queued") {
    return NextResponse.json({ ok: true });
  }

  if (job.status === "running" && job.pid) {
    try {
      process.kill(job.pid, "SIGINT");
    } catch {
      /* process may have already exited */
    }
  }

  db.update(jobs)
    .set({
      status: "cancelled",
      stopRequested: true,
      completedAt: new Date().toISOString(),
    })
    .where(eq(jobs.id, id))
    .run();

  return NextResponse.json({ ok: true });
}
