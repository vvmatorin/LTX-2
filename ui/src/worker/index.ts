import { processQueue } from "./processQueue";

console.log("[worker] LTX-UI job queue worker started");

const POLL_INTERVAL_MS = 1000;

async function tick() {
  try {
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
