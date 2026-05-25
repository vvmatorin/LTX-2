import { NextResponse } from "next/server";
import { getSetting, setSetting } from "@/lib/settings";
import { spawn } from "child_process";

const TB_PORT = 6006;

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function clearTbState(): void {
  setSetting("tbPid", "");
  setSetting("tbPort", "");
  setSetting("tbLogDir", "");
}

function killTbProcess(): void {
  const pidStr = getSetting("tbPid");
  if (pidStr) {
    const pid = parseInt(pidStr, 10);
    if (!isNaN(pid) && pid > 0 && isProcessAlive(pid)) {
      try {
        process.kill(pid, "SIGTERM");
      } catch {
        /* already dead */
      }
    }
  }
  clearTbState();
}

export async function GET() {
  const pidStr = getSetting("tbPid");
  const port = parseInt(getSetting("tbPort") || "0", 10);
  const logDir = getSetting("tbLogDir") || null;

  if (!pidStr || !parseInt(pidStr, 10)) {
    return NextResponse.json({ running: false, port: 0, logDir: null });
  }

  const pid = parseInt(pidStr, 10);
  if (!isProcessAlive(pid)) {
    clearTbState();
    return NextResponse.json({ running: false, port: 0, logDir: null });
  }

  return NextResponse.json({ running: true, port, logDir });
}

export async function POST(req: Request) {
  const body = await req.json();
  const logDir: string = body.logDir;

  if (!logDir) {
    return NextResponse.json(
      { error: "logDir is required" },
      { status: 400 },
    );
  }

  killTbProcess();

  try {
    const child = spawn(
      "tensorboard",
      ["--logdir", logDir, "--port", String(TB_PORT), "--host", "localhost", "--reload_interval", "5"],
      {
        detached: true,
        stdio: "ignore",
      },
    );

    if (!child.pid) {
      return NextResponse.json(
        { error: "Failed to spawn tensorboard process" },
        { status: 500 },
      );
    }

    child.unref();

    setSetting("tbPid", String(child.pid));
    setSetting("tbPort", String(TB_PORT));
    setSetting("tbLogDir", logDir);

    return NextResponse.json({ running: true, port: TB_PORT, logDir });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function DELETE() {
  killTbProcess();
  return NextResponse.json({ running: false });
}
