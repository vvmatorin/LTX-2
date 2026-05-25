"use client";

import type { ProcessingJob } from "@/lib/types";
import { JOB_STATUS } from "@/lib/jobStatus";
import { formatDuration } from "@/lib/format";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useWorkerStatus } from "@/hooks/useWorkerStatus";
import { X, GraduationCap, Layers, Package, AlertTriangle } from "lucide-react";

interface Props {
  jobs: ProcessingJob[];
  onCancel?: (id: number) => void;
}

const TYPE_ICONS = {
  preprocess: Layers,
  merge: Package,
  training: GraduationCap,
} as const;

export function JobQueueList({ jobs, onCancel }: Props) {
  const { alive } = useWorkerStatus();
  const hasQueuedJobs = jobs.some((j) => j.status === "queued" || j.status === "running");

  if (jobs.length === 0) {
    return (
      <div className="surface-neo-inset rounded-2xl border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
        No jobs in queue
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {!alive && hasQueuedJobs && (
        <div className="flex items-center gap-2 rounded-xl border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-400">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>
            Worker process is not running — queued jobs will not start.
            Run <code className="font-mono text-xs bg-yellow-500/10 px-1 rounded">npm run worker</code> or restart with <code className="font-mono text-xs bg-yellow-500/10 px-1 rounded">npm run dev</code>.
          </span>
        </div>
      )}
      {jobs.map((job) => {
        const TypeIcon = TYPE_ICONS[job.type] || Layers;
        const status = JOB_STATUS[job.status];
        const StatusIcon = status.icon;

        return (
          <div
            key={job.id}
            className="surface-neo flex items-center gap-3 rounded-xl border border-border px-3 py-2"
          >
            <TypeIcon className="h-4 w-4 text-muted-foreground shrink-0" />

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium">
                  {job.name}
                </span>
                <Badge
                  variant="outline"
                  className={cn("text-[10px] shrink-0", status.class)}
                >
                  <StatusIcon
                    className={cn(
                      "mr-1 h-3 w-3",
                      job.status === "running" && "animate-spin",
                    )}
                  />
                  {status.label}
                </Badge>
              </div>
              {job.startedAt && (
                <span className="text-[10px] text-muted-foreground">
                  {formatDuration(job.startedAt, job.completedAt)}
                  {job.error && ` — ${job.error}`}
                </span>
              )}
            </div>

            {job.status === "running" && job.progress != null && (
              <div className="w-16 shrink-0">
                <div className="surface-neo-inset h-1.5 w-full rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all"
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
                <span className="text-[10px] text-muted-foreground tabular-nums">
                  {job.progress}%
                </span>
              </div>
            )}

            {(job.status === "queued" || job.status === "running") &&
              onCancel && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive shrink-0"
                  onClick={() => onCancel(job.id)}
                  title={job.status === "running" ? "Stop" : "Cancel"}
                >
                  <X className="h-3.5 w-3.5" />
                </Button>
              )}
          </div>
        );
      })}
    </div>
  );
}
