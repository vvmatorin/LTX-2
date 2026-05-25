"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useJobs } from "@/hooks/useJobs";
import { useTensorboard } from "@/hooks/useTensorboard";
import { formatDuration } from "@/lib/format";
import { LogViewer } from "@/components/LogViewer";
import { JobQueueList } from "@/components/JobQueueList";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Separator } from "@/components/ui/separator";
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible";
import { Loader2, Square, ChevronDown, Clock, Activity, BarChart3, X } from "lucide-react";
import { cn } from "@/lib/utils";

export default function RunsPage() {
  const router = useRouter();
  const { jobs, stopJob } = useJobs();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [tbFullscreen, setTbFullscreen] = useState(false);

  const activeJob = jobs.find((j) => j.status === "running") ?? null;
  const queuedJobs = useMemo(() => jobs.filter((j) => j.status === "queued"), [jobs]);
  const historyJobs = useMemo(
    () => jobs.filter((j) => ["completed", "failed", "cancelled"].includes(j.status)),
    [jobs],
  );

  const tb = useTensorboard(activeJob);

  const handleStop = async (id: number) => {
    try {
      await stopJob(id);
    } catch (err) {
      console.error("Failed to stop job:", err);
    }
  };

  const elapsedDisplay = activeJob?.startedAt ? formatDuration(activeJob.startedAt) : null;

  return (
    <div className="space-y-6 p-3 md:p-4">
      <PageHeader
        title="Runs"
        subtitle="Monitor active jobs, manage the queue, and review history"
      />

      {activeJob ? (
        <Card className="ring-glow border-blue-400/30">
          <CardHeader className="py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="surface-neo-inset flex h-8 w-8 items-center justify-center rounded-full bg-blue-500/10">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
                </div>
                <div>
                  <CardTitle className="text-sm">{activeJob.name}</CardTitle>
                  <div className="mt-0.5 flex items-center gap-2">
                    <Badge
                      variant="outline"
                      className="border-blue-500/30 bg-blue-500/10 text-[10px] text-blue-400"
                    >
                      {activeJob.type}
                    </Badge>
                    {elapsedDisplay && (
                      <span className="text-muted-foreground flex items-center gap-1 text-[10px]">
                        <Clock className="h-3 w-3" />
                        {elapsedDisplay}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {activeJob.type === "training" && (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!tb.ready}
                    onClick={() => setTbFullscreen(true)}
                  >
                    {tb.running && !tb.ready ? (
                      <Loader2 className="mr-1.5 h-3 w-3 animate-spin" />
                    ) : (
                      <BarChart3 className="mr-1.5 h-3 w-3" />
                    )}
                    TensorBoard
                  </Button>
                )}
                <ConfirmDialog
                  title="Stop running job?"
                  description={`This will terminate "${activeJob.name}". Any unsaved progress will be lost.`}
                  confirmLabel="Stop"
                  onConfirm={() => handleStop(activeJob.id)}
                >
                  <Button variant="destructive" size="sm">
                    <Square className="mr-1.5 h-3 w-3" />
                    Stop
                  </Button>
                </ConfirmDialog>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {activeJob.progress != null && (
              <div className="space-y-1">
                <div className="text-muted-foreground flex items-center justify-between text-xs">
                  <span>Progress</span>
                  <span className="tabular-nums">{activeJob.progress}%</span>
                </div>
                <div className="surface-neo-inset bg-muted h-2 w-full rounded-full">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-500"
                    style={{ width: `${activeJob.progress}%` }}
                  />
                </div>
              </div>
            )}

            <LogViewer jobId={activeJob.id} />
          </CardContent>
        </Card>
      ) : (
        <Card className="border-dashed">
          <CardContent className="text-muted-foreground flex flex-col items-center justify-center py-12">
            <div className="surface-neo-inset bg-muted/50 mb-3 flex h-12 w-12 items-center justify-center rounded-full">
              <Activity className="h-5 w-5" />
            </div>
            <p className="text-sm font-medium">No active job</p>
            <p className="mt-1 text-xs">
              {queuedJobs.length > 0
                ? `${queuedJobs.length} job${queuedJobs.length !== 1 ? "s" : ""} in queue`
                : "Queue a preprocessing or training job to get started"}
            </p>
          </CardContent>
        </Card>
      )}

      {queuedJobs.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-muted-foreground text-sm font-medium">Queue ({queuedJobs.length})</h2>
          <JobQueueList jobs={queuedJobs} onCancel={handleStop} />
        </div>
      )}

      <Separator />

      <Collapsible open={historyOpen}>
        <button
          type="button"
          className="flex w-full items-center justify-between py-1"
          onClick={() => setHistoryOpen(!historyOpen)}
        >
          <h2 className="text-muted-foreground text-sm font-medium">
            History ({historyJobs.length})
          </h2>
          <ChevronDown
            className={cn(
              "text-muted-foreground h-4 w-4 transition-transform",
              historyOpen && "rotate-180",
            )}
          />
        </button>
        <CollapsibleContent className="pt-3">
          <JobQueueList
            jobs={historyJobs}
            onReuse={(job) => router.push(`/training?fromJob=${job.id}`)}
          />
        </CollapsibleContent>
      </Collapsible>

      {tbFullscreen && (
        <div className="bg-background fixed inset-0 z-50 flex flex-col">
          <div className="flex items-center justify-between border-b px-4 py-2">
            <span className="text-sm font-medium">TensorBoard</span>
            <Button variant="ghost" size="sm" onClick={() => setTbFullscreen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          {tb.ready ? (
            <iframe
              src="/tensorboard/"
              className="w-full flex-1 border-0"
              title="TensorBoard Fullscreen"
            />
          ) : (
            <div className="flex flex-1 items-center justify-center">
              <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
