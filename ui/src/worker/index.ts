import { processQueue } from "./processQueue";
import { getWorkerDb } from "./db";

console.log("[worker] LTX-UI job queue worker started");

const POLL_INTERVAL_MS = 1000;

function writeHeartbeat() {
  const db = getWorkerDb();
  db.prepare(
    "INSERT INTO settings (key, value) VALUES ('worker_heartbeat', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
  ).run(new Date().toISOString());
}

async function tick() {
  try {
    writeHeartbeat();
    await processQueue();
  } catch (err) {
    console.error("[worker] processQueue error:", err);
  }
}

setInterval(tick, POLL_INTERVAL_MS);

process.on("SIGINT", () => {
  console.log("[worker] Received SIGINT, shutting down");
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log("[worker] Received SIGTERM, shutting down");
  process.exit(0);
});
