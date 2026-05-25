import { NextResponse } from "next/server";
import { db } from "@/db";
import { jobs } from "@/db/schema";
import { eq } from "drizzle-orm";
import fs from "fs";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id: idStr } = await params;
  const id = Number(idStr);

  const job = db.select().from(jobs).where(eq(jobs.id, id)).get();
  if (!job) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }

  if (!job.logFile || !fs.existsSync(job.logFile)) {
    return NextResponse.json({ error: "No log file" }, { status: 404 });
  }

  const { searchParams } = new URL(req.url);
  const mode = searchParams.get("mode");

  if (mode === "sse") {
    const encoder = new TextEncoder();
    let offset = 0;
    let closed = false;
    let pollInterval: ReturnType<typeof setInterval> | null = null;
    let doneInterval: ReturnType<typeof setInterval> | null = null;

    const cleanup = () => {
      closed = true;
      if (pollInterval) clearInterval(pollInterval);
      if (doneInterval) clearInterval(doneInterval);
      pollInterval = null;
      doneInterval = null;
    };

    const stream = new ReadableStream({
      start(controller) {
        const sendChunk = () => {
          if (closed) return;
          try {
            const stat = fs.statSync(job.logFile!);
            if (stat.size > offset) {
              const fd = fs.openSync(job.logFile!, "r");
              const buf = Buffer.alloc(stat.size - offset);
              fs.readSync(fd, buf, 0, buf.length, offset);
              fs.closeSync(fd);
              offset = stat.size;
              const lines = buf.toString("utf-8");
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(lines)}\n\n`));
            }
          } catch {
            // File may have been removed
          }
        };

        sendChunk();

        pollInterval = setInterval(sendChunk, 200);

        doneInterval = setInterval(() => {
          const currentJob = db.select().from(jobs).where(eq(jobs.id, id)).get();
          if (!currentJob || ["completed", "failed", "cancelled"].includes(currentJob.status)) {
            sendChunk();
            controller.enqueue(encoder.encode(`data: ${JSON.stringify("__DONE__")}\n\n`));
            cleanup();
            controller.close();
          }
        }, 1000);
      },
      cancel() {
        cleanup();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }

  const content = fs.readFileSync(job.logFile, "utf-8");
  return NextResponse.json({ content });
}
