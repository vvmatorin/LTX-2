import { NextResponse } from "next/server";
import { db } from "@/db";
import { settings } from "@/db/schema";

const DEFAULT_SETTINGS: Record<string, string> = {
  modelPath: "",
  textEncoderPath: "",
  outputDir: "",
  datasetDir: "",
  scriptsDir: "",
  hfLoggedIn: "false",
};

export async function GET() {
  const rows = db.select().from(settings).all();
  const result: Record<string, string> = { ...DEFAULT_SETTINGS };
  for (const row of rows) {
    result[row.key] = row.value;
  }
  return NextResponse.json(result);
}

export async function PUT(req: Request) {
  const body = await req.json();
  for (const [key, value] of Object.entries(body)) {
    if (typeof value === "string") {
      db.insert(settings)
        .values({ key, value })
        .onConflictDoUpdate({ target: settings.key, set: { value } })
        .run();
    }
  }
  return NextResponse.json({ ok: true });
}
