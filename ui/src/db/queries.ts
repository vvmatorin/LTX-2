import { db } from "./index";
import { jobs } from "./schema";

export function nextQueuePosition(): number {
  const maxPos = db
    .select()
    .from(jobs)
    .all()
    .filter((j) => j.status === "queued")
    .reduce((max, j) => Math.max(max, j.queuePosition), -1);
  return maxPos + 1;
}
