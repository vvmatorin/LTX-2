"use client";

import type { ProcessingJob } from "@/lib/types";
import { RESOLUTION_OPTIONS, FRAME_COUNT_OPTIONS } from "@/lib/types";
import { JOB_STATUS } from "@/lib/jobStatus";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/EmptyState";
import { cn } from "@/lib/utils";

interface Props {
  jobs: ProcessingJob[];
  folderId: number;
}

export function ProcessingMatrix({ jobs, folderId }: Props) {
  // Jobs arrive newest-first (DESC). Keep the first occurrence per key so the
  // latest re-queued job is shown rather than the original failed one.
  const jobMap = new Map<string, ProcessingJob>();
  for (const job of jobs) {
    if (job.type !== "preprocess") continue;
    const cfg = job.config as {
      folderId?: number;
      resolution?: number;
      frameCounts?: number[];
    };
    if (cfg.folderId !== folderId) continue;
    const res = cfg.resolution;
    const frames = cfg.frameCounts;
    if (res && frames) {
      for (const f of frames) {
        const key = `${res}_${f}`;
        if (!jobMap.has(key)) jobMap.set(key, job);
      }
    }
  }

  if (jobMap.size === 0) {
    return (
      <EmptyState>
        No processing jobs for this folder yet. Configure and queue above.
      </EmptyState>
    );
  }

  const activeResolutions = RESOLUTION_OPTIONS.filter((res) =>
    FRAME_COUNT_OPTIONS.some((f) => jobMap.has(`${res}_${f}`)),
  );

  return (
    <div className="overflow-x-auto rounded-lg border border-border p-2">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="px-2 py-1.5 text-left text-muted-foreground font-medium">
              Res \ Frames
            </th>
            {FRAME_COUNT_OPTIONS.map((f) => (
              <th
                key={f}
                className="px-2 py-1.5 text-center text-muted-foreground font-medium"
              >
                {f === 1 ? "img" : `${f}f`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {activeResolutions.map((res) => (
            <tr key={res} className="border-t border-white/10">
              <td className="px-2 py-1.5 font-medium">{res}px</td>
              {FRAME_COUNT_OPTIONS.map((f) => {
                const job = jobMap.get(`${res}_${f}`);
                if (!job) {
                  return (
                    <td
                      key={f}
                      className="px-2 py-1.5 text-center text-muted-foreground/30"
                    >
                      —
                    </td>
                  );
                }
                const style = JOB_STATUS[job.status];
                const Icon = style.icon;
                return (
                  <td key={f} className="px-2 py-1.5 text-center">
                    <Badge
                      variant="outline"
                      className={cn("text-[10px] gap-1", style.class)}
                    >
                      <Icon
                        className={cn(
                          "h-3 w-3",
                          job.status === "running" && "animate-spin",
                        )}
                      />
                      {job.status === "running" && job.progress != null
                        ? `${job.progress}%`
                        : job.status}
                    </Badge>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
