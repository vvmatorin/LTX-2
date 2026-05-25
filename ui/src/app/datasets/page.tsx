"use client";

import { useState, useMemo } from "react";
import type { DatasetBucket } from "@/lib/types";

import { useFolders } from "@/hooks/useFolders";
import { useJobs } from "@/hooks/useJobs";
import { useDatasets } from "@/hooks/useDatasets";
import { useSettings } from "@/hooks/useSettings";
import { SourceFolderCard } from "@/components/SourceFolderCard";
import { FolderConfigPanel, type ResFrameConfig } from "@/components/FolderConfigPanel";
import { ProcessingMatrix } from "@/components/ProcessingMatrix";
import { DatasetBuilder } from "@/components/DatasetBuilder";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FolderPlus } from "lucide-react";

export default function DatasetsPage() {
  const { folders, addFolder, removeFolder } = useFolders();
  const { jobs, createJob, refreshJobs, isRefreshingJobs } = useJobs();
  const { datasets, createDataset, buildDataset, deleteDataset, refreshDatasets, isRefreshing: isRefreshingDatasets } = useDatasets();
  const { settings } = useSettings();

  const [selectedFolderId, setSelectedFolderId] = useState<number | null>(null);
  const [newFolderPath, setNewFolderPath] = useState("");
  const [buildError, setBuildError] = useState<string | null>(null);

  const effectiveFolderId = selectedFolderId ?? folders[0]?.id ?? null;
  const selectedFolder = folders.find((f) => f.id === effectiveFolderId) ?? null;
  const completedJobs = useMemo(
    () => jobs.filter((j) => j.type === "preprocess" && j.status === "completed" && j.outputExists !== false),
    [jobs],
  );

  const handleRefresh = () => {
    refreshDatasets();
    refreshJobs();
  };
  const isRefreshing = isRefreshingDatasets || isRefreshingJobs;

  const handleAddFolder = async () => {
    if (!newFolderPath.trim()) return;
    try {
      await addFolder(newFolderPath.trim());
      setNewFolderPath("");
    } catch (err) {
      console.error("Failed to add folder:", err);
    }
  };

  const handleRemoveFolder = async (id: number) => {
    try {
      await removeFolder(id);
    } catch (err) {
      console.error("Failed to remove folder:", err);
    }
    if (effectiveFolderId === id) setSelectedFolderId(null);
  };

  const handleQueueProcessing = async (
    configs: Array<{ resolution: number; frameCount: number; config: ResFrameConfig }>,
  ) => {
    for (const c of configs) {
      await createJob({
        type: "preprocess",
        name: `${selectedFolder?.path ?? "?"} / ${c.resolution} / ${c.frameCount === 1 ? "img" : `${c.frameCount}f`}`,
        config: {
          folderId: effectiveFolderId,
          resolution: c.resolution,
          frameCounts: [c.frameCount],
          folderPath: selectedFolder?.path,
          hFlip: c.config.hFlip,
          frameSampling: c.config.frameSampling,
          withAudio: c.config.withAudio,
          datasetFilename: c.config.datasetFilename,
        },
      });
    }
  };

  const handleBuildDataset = async (name: string, buckets: DatasetBucket[]) => {
    setBuildError(null);
    const datasetDir = settings?.datasetDir?.replace(/\/+$/, "") || "/tmp/ltx-datasets";
    const datasetPath = `${datasetDir}/${name}`;

    try {
      const dataset = await createDataset({ name, path: datasetPath, buckets });
      await buildDataset(dataset.id);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setBuildError(msg);
    }
  };

  return (
    <div className="space-y-6 p-3 md:p-4">
      <div>
        <h1 className="title-gradient page-title">Datasets</h1>
        <p className="page-subtitle mt-2.5">
          Manage source folders, process videos into latents, and build training datasets
        </p>
      </div>

      <Tabs defaultValue="folders" className="space-y-4">
        <TabsList>
          <TabsTrigger value="folders">Source Folders</TabsTrigger>
          <TabsTrigger value="processing">Processing</TabsTrigger>
          <TabsTrigger value="build">Training Datasets</TabsTrigger>
        </TabsList>

        <TabsContent value="folders" className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter folder path, e.g. /data/my-dataset/v1/videos"
              value={newFolderPath}
              onChange={(e) => setNewFolderPath(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddFolder()}
              className="font-mono text-xs"
            />
            <Button onClick={handleAddFolder} className="shrink-0">
              <FolderPlus className="mr-1.5 h-4 w-4" />
              Add
            </Button>
          </div>

          <div className="grid gap-3">
            {folders.map((folder) => (
              <SourceFolderCard
                key={folder.id}
                folder={folder}
                selected={effectiveFolderId === folder.id}
                onSelect={() => setSelectedFolderId(folder.id)}
                onRemove={() => handleRemoveFolder(folder.id)}
              />
            ))}
            {folders.length === 0 && (
              <div className="rounded-lg border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
                No folders added yet. Enter a path above to get started.
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent value="processing" className="space-y-4">
          {selectedFolder ? (
            <>
              <FolderConfigPanel
                folder={selectedFolder}
                onQueueProcessing={handleQueueProcessing}
              />
              <ProcessingMatrix jobs={jobs} folderId={selectedFolder.id} />
            </>
          ) : (
            <div className="rounded-lg border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
              Select a folder from the Source Folders tab to configure processing.
            </div>
          )}
        </TabsContent>

        <TabsContent value="build" className="space-y-4">
          {buildError && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
              {buildError}
            </div>
          )}
          <DatasetBuilder
            completedJobs={completedJobs}
            datasets={datasets}
            onBuildDataset={handleBuildDataset}
            onDeleteDataset={async (id) => { try { await deleteDataset(id); } catch (err) { console.error(err); } }}
            onRefresh={handleRefresh}
            isRefreshing={isRefreshing}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
