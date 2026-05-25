"use client";

import { useState } from "react";
import type { ProcessingJob, TrainingDataset, DatasetBucket } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Package, Plus, CheckCircle2, Square, CheckSquare, Trash2, AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  completedJobs: ProcessingJob[];
  datasets: TrainingDataset[];
  onBuildDataset: (name: string, buckets: DatasetBucket[]) => void;
  onDeleteDataset: (id: number) => void;
  onRefresh: () => void;
  isRefreshing?: boolean;
}

export function DatasetBuilder({ completedJobs, datasets, onBuildDataset, onDeleteDataset, onRefresh, isRefreshing }: Props) {
  const [datasetName, setDatasetName] = useState("");
  const [selectedJobIds, setSelectedJobIds] = useState<Set<number>>(new Set());

  const toggleJob = (id: number) => {
    setSelectedJobIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleBuild = () => {
    if (!datasetName.trim()) return;
    const buckets: DatasetBucket[] = completedJobs
      .filter((j) => selectedJobIds.has(j.id))
      .map((j) => {
        const cfg = j.config as {
          folderId?: number;
          folderPath?: string;
          resolution?: number;
          frameCounts?: number[];
          withAudio?: boolean;
          hFlip?: boolean;
          resolutionBuckets?: string;
        };
        const bucketKeys = cfg.resolutionBuckets
          ? cfg.resolutionBuckets.split(";")
          : [];
        return {
          folderName: j.name.split(" / ")[0] || "unknown",
          folderPath: cfg.folderPath || "",
          jobId: j.id,
          resolution: cfg.resolution || 0,
          frameCount: (cfg.frameCounts || [0])[0],
          bucketKeys,
          hasAudio: cfg.withAudio || false,
          hasHFlip: cfg.hFlip || false,
        };
      });
    onBuildDataset(datasetName, buckets);
    setDatasetName("");
    setSelectedJobIds(new Set());
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">
            Existing Training Datasets
          </Label>
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1.5 text-xs"
            onClick={onRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={cn("h-3 w-3", isRefreshing && "animate-spin")} />
            Refresh
          </Button>
        </div>
        {datasets.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border p-4 text-center text-sm text-muted-foreground">
            No training datasets yet.
          </div>
        ) : (
          <>
          {datasets.map((ds) => (
            <Card
              key={ds.id}
              className={cn(
                "bg-muted/30",
                !ds.pathExists && "border-destructive/30 bg-destructive/5",
              )}
            >
              <CardContent className="flex items-center gap-3 p-3">
                <Package
                  className={cn(
                    "h-4 w-4 shrink-0",
                    ds.pathExists ? "text-emerald-400" : "text-destructive/70",
                  )}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">{ds.name}</span>
                    {!ds.pathExists && (
                      <span className="flex items-center gap-1 text-[10px] text-destructive">
                        <AlertTriangle className="h-3 w-3" />
                        path missing
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground font-mono break-all">{ds.path}</p>
                </div>
                <Badge variant="secondary" className="text-[10px] shrink-0">
                  {ds.buckets.length} bucket{ds.buckets.length !== 1 ? "s" : ""}
                </Badge>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 w-7 shrink-0 p-0 text-muted-foreground hover:bg-destructive/15 hover:text-destructive"
                  onClick={() => onDeleteDataset(ds.id)}
                  title="Delete dataset record"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </CardContent>
            </Card>
          ))}
          </>
        )}
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Build Training Dataset</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="dsName" className="text-xs">Dataset Name</Label>
            <Input
              id="dsName"
              placeholder="e.g. nature-multi-res"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
            />
          </div>

          {completedJobs.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
              No completed preprocessing jobs. Process some folders first.
            </div>
          ) : (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground uppercase tracking-wider">
                Select Processed Buckets
              </Label>
              <div className="space-y-1.5 max-h-64 overflow-y-auto">
                {completedJobs.map((job) => {
                  const checked = selectedJobIds.has(job.id);
                  return (
                    <button
                      key={job.id}
                      type="button"
                      onClick={() => toggleJob(job.id)}
                      className={cn(
                        "flex w-full cursor-pointer items-center gap-3 rounded-md border px-3 py-2 text-left transition-colors",
                        checked
                          ? "border-primary bg-primary/10"
                          : "border-border hover:border-primary/30",
                      )}
                    >
                      {checked ? (
                        <CheckSquare className="h-4 w-4 text-primary shrink-0" />
                      ) : (
                        <Square className="h-4 w-4 text-muted-foreground shrink-0" />
                      )}
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                      <span className="text-sm">{job.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {selectedJobIds.size > 0 && datasetName.trim() && (
            <div className="flex items-center justify-between pt-2">
              <span className="text-sm text-muted-foreground">
                {selectedJobIds.size} bucket{selectedJobIds.size !== 1 ? "s" : ""} selected
              </span>
              <Button onClick={handleBuild}>
                <Plus className="mr-1.5 h-4 w-4" />
                Build Dataset
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
