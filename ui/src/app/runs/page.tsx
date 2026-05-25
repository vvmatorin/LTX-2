"use client";

import { useState, useMemo } from "react";
import { useJobs } from "@/hooks/useJobs";
import { useSSELog } from "@/hooks/useSSELog";
import { useTensorboard } from "@/hooks/useTensorboard";
import { formatDuration } from "@/lib/format";
import { parseLossPoints } from "@/lib/logParsing";
import { LogViewer } from "@/components/LogViewer";
import { LossGraph } from "@/components/LossGraph";
import { JobQueueList } from "@/components/JobQueueList";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible";
import {
  Loader2,
  Square,
  ChevronDown,
  Clock,
  Activity,
  Maximize2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function RunsPage() {
  const { jobs, stopJob } = useJobs();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [tbFullscreen, setTbFullscreen] = useState(false);

  const activeJob = jobs.find((j) => j.status === "running") ?? null;
  const queuedJobs = useMemo(
    () => jobs.filter((j) => j.status === "queued"),
    [jobs],
  );
  const historyJobs = useMemo(
    () =>
      jobs.filter((j) =>
        ["completed", "failed", "cancelled"].includes(j.status),
      ),
    [jobs],
  );

  const { lines: logLines } = useSSELog(activeJob?.id ?? null);
  const lossPoints = useMemo(() => parseLossPoints(logLines), [logLines]);
  const tb = useTensorboard(activeJob);

  const handleStop = async (id: number) => {
    try {
      await stopJob(id);
    } catch (err) {
      console.error("Failed to stop job:", err);
    }
  };

  const elapsedDisplay = activeJob?.startedAt
    ? formatDuration(activeJob.startedAt)
    : null;

  const isTrainingJob = activeJob?.type === "training";

  return (
    <div className="space-y-6 p-3 md:p-4">
      <div>
        <h1 className="title-gradient page-title">Runs</h1>
        <p className="page-subtitle mt-2.5">
          Monitor active jobs, manage the queue, and review history
        </p>
      </div>

      {activeJob ? (
        <Card className="ring-glow border-blue-400/30">
          <CardHeader className="py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="surface-neo-inset flex h-8 w-8 items-center justify-center rounded-full bg-blue-500/10">
                  <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />
                </div>
                <div>
                  <CardTitle className="text-sm">{activeJob.name}</CardTitle>
                  <div className="flex items-center gap-2 mt-0.5">
                    <Badge variant="outline" className="text-[10px] bg-blue-500/10 text-blue-400 border-blue-500/30">
                      {activeJob.type}
                    </Badge>
                    {elapsedDisplay && (
                      <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        {elapsedDisplay}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => handleStop(activeJob.id)}
              >
                <Square className="mr-1.5 h-3 w-3" />
                Stop
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {activeJob.progress != null && (
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>Progress</span>
                  <span className="tabular-nums">{activeJob.progress}%</span>
                </div>
                <div className="surface-neo-inset h-2 w-full rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-500"
                    style={{ width: `${activeJob.progress}%` }}
                  />
                </div>
              </div>
            )}

            {lossPoints.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-xs font-medium text-muted-foreground">Loss</span>
                  <span className="text-xs text-muted-foreground tabular-nums ml-auto">
                    {lossPoints[lossPoints.length - 1].loss.toFixed(4)} @ step{" "}
                    {lossPoints[lossPoints.length - 1].step}
                  </span>
                </div>
                <LossGraph points={lossPoints} className="h-48" />
              </div>
            )}

            {isTrainingJob && (
              <TensorBoardPanel
                running={tb.running}
                port={tb.port}
                error={tb.error}
                fullscreen={tbFullscreen}
                onToggleFullscreen={() => setTbFullscreen(!tbFullscreen)}
              />
            )}

            <LogViewer lines={logLines} />
          </CardContent>
        </Card>
      ) : (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <div className="surface-neo-inset mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-muted/50">
              <Activity className="h-5 w-5" />
            </div>
            <p className="text-sm font-medium">No active job</p>
            <p className="text-xs mt-1">
              {queuedJobs.length > 0
                ? `${queuedJobs.length} job${queuedJobs.length !== 1 ? "s" : ""} in queue`
                : "Queue a preprocessing or training job to get started"}
            </p>
          </CardContent>
        </Card>
      )}

      {queuedJobs.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-medium text-muted-foreground">
            Queue ({queuedJobs.length})
          </h2>
          <JobQueueList jobs={queuedJobs} onCancel={handleStop} />
        </div>
      )}

      <Separator />

      <Collapsible open={historyOpen} onOpenChange={setHistoryOpen}>
        <button
          type="button"
          className="flex w-full items-center justify-between py-1"
          onClick={() => setHistoryOpen(!historyOpen)}
        >
          <h2 className="text-sm font-medium text-muted-foreground">
            History ({historyJobs.length})
          </h2>
          <ChevronDown
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform",
              historyOpen && "rotate-180",
            )}
          />
        </button>
        <CollapsibleContent className="pt-3">
          <JobQueueList jobs={historyJobs} />
        </CollapsibleContent>
      </Collapsible>

      {tbFullscreen && (
        <div className="fixed inset-0 z-50 bg-background flex flex-col">
          <div className="flex items-center justify-between px-4 py-2 border-b">
            <span className="text-sm font-medium">TensorBoard</span>
            <Button variant="ghost" size="sm" onClick={() => setTbFullscreen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <iframe
            src={`http://localhost:${tb.port}`}
            className="flex-1 w-full border-0"
            title="TensorBoard Fullscreen"
          />
        </div>
      )}
    </div>
  );
}

function TensorBoardPanel({
  running,
  port,
  error,
  fullscreen,
  onToggleFullscreen,
}: {
  running: boolean;
  port: number;
  error: string | null;
  fullscreen: boolean;
  onToggleFullscreen: () => void;
}) {
  if (fullscreen) return null;

  return (
    <div className="rounded-lg border overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "h-2 w-2 rounded-full",
              running ? "bg-green-500" : "bg-yellow-500 animate-pulse",
            )}
          />
          <span className="text-xs font-medium">
            {running ? "TensorBoard" : "TensorBoard starting..."}
          </span>
        </div>
        {running && (
          <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={onToggleFullscreen}>
            <Maximize2 className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
      {error && (
        <div className="px-3 py-2 text-xs text-destructive bg-destructive/5">
          {error}
        </div>
      )}
      {running ? (
        <iframe
          src={`http://localhost:${port}`}
          className="w-full border-0"
          style={{ height: "520px" }}
          title="TensorBoard"
        />
      ) : (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      )}
    </div>
  );
}
