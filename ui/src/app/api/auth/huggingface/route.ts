import { NextResponse } from "next/server";
import { db } from "@/db";
import { settings } from "@/db/schema";
import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

export async function GET() {
  try {
    const { stdout } = await execFileAsync("huggingface-cli", ["whoami"], {
      timeout: 10_000,
    });
    const username = stdout.trim();
    if (username && !username.includes("Not logged in")) {
      return NextResponse.json({ loggedIn: true, username });
    }
    return NextResponse.json({ loggedIn: false, username: null });
  } catch {
    return NextResponse.json({ loggedIn: false, username: null });
  }
}

export async function POST(req: Request) {
  const body = await req.json();
  const token: string = body.token;

  if (!token) {
    return NextResponse.json({ error: "token is required" }, { status: 400 });
  }

  try {
    await execFileAsync("huggingface-cli", ["login", "--token", token], {
      timeout: 30_000,
      env: { ...process.env, HF_TOKEN: token },
    });

    const { stdout } = await execFileAsync("huggingface-cli", ["whoami"], {
      timeout: 10_000,
    });
    const username = stdout.trim();

    db.insert(settings)
      .values({ key: "hfLoggedIn", value: "true" })
      .onConflictDoUpdate({ target: settings.key, set: { value: "true" } })
      .run();

    db.insert(settings)
      .values({ key: "hfUsername", value: username })
      .onConflictDoUpdate({ target: settings.key, set: { value: username } })
      .run();

    return NextResponse.json({ loggedIn: true, username });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: message, loggedIn: false },
      { status: 500 },
    );
  }
}

export async function DELETE() {
  try {
    await execFileAsync("huggingface-cli", ["logout"], { timeout: 10_000 });
  } catch {
    // May already be logged out
  }

  db.insert(settings)
    .values({ key: "hfLoggedIn", value: "false" })
    .onConflictDoUpdate({ target: settings.key, set: { value: "false" } })
    .run();

  return NextResponse.json({ loggedIn: false });
}
