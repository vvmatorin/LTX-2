'use client';

import { useState, useMemo } from 'react';
import type { DatasetBucket, ModelStream } from '@/lib/types';
import { MODEL_STREAMS, MODEL_STREAM_LABELS, pickDefaultStream } from '@/lib/types';
import { toErrorMessage } from '@/lib/utils';

import { useFolders } from '@/hooks/useFolders';
import { useJobs } from '@/hooks/useJobs';
import { useDatasets } from '@/hooks/useDatasets';
import { useSettings } from '@/hooks/useSettings';
import { SourceFolderCard } from '@/components/SourceFolderCard';
import { FolderConfigPanel, type ResFrameConfig, type AudioConfig } from '@/components/FolderConfigPanel';
import { ProcessingMatrix } from '@/components/ProcessingMatrix';
import { AudioProcessingStatus } from '@/components/AudioProcessingStatus';
import { DatasetBuilder } from '@/components/DatasetBuilder';
import { PageHeader } from '@/components/PageHeader';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { EmptyState } from '@/components/EmptyState';
import { FolderPlus } from 'lucide-react';

export default function DatasetsPage() {
  const { folders, addFolder, removeFolder, refreshFolder, error: foldersError } = useFolders();
  const { jobs, createJob, refreshJobs, error: jobsError } = useJobs();
  const { datasets, createDataset, deleteDataset, refreshDatasets, error: datasetsError } = useDatasets();
  const { settings } = useSettings();

  const fetchError = foldersError || jobsError || datasetsError;

  const [selectedFolderId, setSelectedFolderId] = useState<number | null>(null);
  const [selectedStream, setSelectedStream] = useState<ModelStream | null>(null);
  const [newFolderPath, setNewFolderPath] = useState('');
  const [folderError, setFolderError] = useState<string | null>(null);
  const [buildError, setBuildError] = useState<string | null>(null);
  const [isManualRefreshing, setIsManualRefreshing] = useState(false);
  const [refreshingFolderId, setRefreshingFolderId] = useState<number | null>(null);

  const effectiveFolderId = selectedFolderId ?? folders[0]?.id ?? null;
  const stream: ModelStream = selectedStream ?? (settings ? pickDefaultStream(settings) : 'ltx-2.5');
  const selectedFolder = folders.find(f => f.id === effectiveFolderId) ?? null;
  const completedJobs = useMemo(
    () => jobs.filter(j => j.type === 'preprocess' && j.status === 'completed' && j.outputExists !== false),
    [jobs],
  );

  const handleRefresh = async () => {
    setIsManualRefreshing(true);
    try {
      await Promise.all([refreshDatasets(), refreshJobs()]);
    } finally {
      setIsManualRefreshing(false);
    }
  };

  const handleAddFolder = async () => {
    if (!newFolderPath.trim()) return;
    setFolderError(null);
    try {
      await addFolder(newFolderPath.trim());
      setNewFolderPath('');
    } catch (err) {
      setFolderError(toErrorMessage(err));
    }
  };

  const handleRemoveFolder = async (id: number) => {
    setFolderError(null);
    try {
      await removeFolder(id);
    } catch (err) {
      setFolderError(toErrorMessage(err));
    }
    if (effectiveFolderId === id) setSelectedFolderId(null);
  };

  const handleRefreshFolder = async (id: number) => {
    setRefreshingFolderId(id);
    try {
      await refreshFolder(id);
    } catch (err) {
      setFolderError(toErrorMessage(err));
    } finally {
      setRefreshingFolderId(null);
    }
  };

  const handleQueueProcessing = async (
    configs: Array<{ resolution: number; frameCount: number; config: ResFrameConfig }>,
  ) => {
    for (const c of configs) {
      await createJob({
        type: 'preprocess',
        name: `Preprocess: ${selectedFolder?.path ?? '?'}/_buckets/${c.resolution}_${c.frameCount} [${stream}]`,
        config: {
          folderId: effectiveFolderId,
          modelStream: stream,
          resolution: c.resolution,
          frameCounts: [c.frameCount],
          folderPath: selectedFolder?.path,
          hFlip: c.config.hFlip,
          frameSampling: c.config.frameSampling,
          withAudio: c.config.withAudio,
          datasetFilename: c.config.datasetFilename,
          referenceColumn: c.config.referenceColumn,
          referenceDownscaleFactor: c.config.referenceDownscaleFactor,
        },
      });
    }
  };

  const handleQueueAudioProcessing = async (config: AudioConfig) => {
    await createJob({
      type: 'preprocess',
      name: `Preprocess: ${selectedFolder?.path ?? '?'}/_buckets/audio_only [${stream}]`,
      config: {
        folderId: effectiveFolderId,
        modelStream: stream,
        folderPath: selectedFolder?.path,
        audioOnly: true,
        withAudio: true,
        datasetFilename: config.datasetFilename,
        maxDuration: config.maxDuration,
      },
    });
  };

  const handleDeleteDataset = async (id: number) => {
    setBuildError(null);
    try {
      await deleteDataset(id);
    } catch (err) {
      setBuildError(toErrorMessage(err));
    }
  };

  const handleBuildDataset = async (name: string, buckets: DatasetBucket[]) => {
    setBuildError(null);
    const datasetDir = settings?.datasetDir?.replace(/\/+$/, '') || '/tmp/ltx-datasets';
    const datasetPath = `${datasetDir}/${name}`;

    try {
      await createDataset({ name, path: datasetPath, buckets });
    } catch (err) {
      setBuildError(toErrorMessage(err));
      throw err;
    }
  };

  return (
    <div className="space-y-6 p-3 md:p-4">
      <PageHeader
        title="Datasets"
        subtitle="Manage source folders, process videos into latents, and build training datasets"
      />

      {fetchError && (
        <div className="border-destructive/30 bg-destructive/10 text-destructive rounded-lg border p-3 text-sm">
          {toErrorMessage(fetchError)}
        </div>
      )}

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
              onChange={e => setNewFolderPath(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleAddFolder()}
              className="font-mono text-xs"
            />
            <Button onClick={handleAddFolder} className="shrink-0">
              <FolderPlus className="mr-1.5 h-4 w-4" />
              Add
            </Button>
          </div>

          {folderError && (
            <div className="border-destructive/30 bg-destructive/10 text-destructive rounded-lg border p-3 text-sm">
              {folderError}
            </div>
          )}

          <div className="grid gap-3">
            {folders.map(folder => (
              <SourceFolderCard
                key={folder.id}
                folder={folder}
                selected={effectiveFolderId === folder.id}
                onSelect={() => setSelectedFolderId(folder.id)}
                onRemove={() => handleRemoveFolder(folder.id)}
                onRefresh={() => handleRefreshFolder(folder.id)}
                isRefreshing={refreshingFolderId === folder.id}
              />
            ))}
            {folders.length === 0 && (
              <EmptyState className="p-8">No folders added yet. Enter a path above to get started.</EmptyState>
            )}
          </div>
        </TabsContent>

        <TabsContent value="processing" className="space-y-4">
          {selectedFolder ? (
            <>
              <div className="flex items-center gap-3">
                <Label className="text-xs">Model</Label>
                <Select value={stream} onValueChange={v => v && setSelectedStream(v as ModelStream)}>
                  <SelectTrigger className="w-[140px] text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MODEL_STREAMS.map(ms => (
                      <SelectItem key={ms} value={ms}>
                        {MODEL_STREAM_LABELS[ms]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <span className="text-muted-foreground text-[11px]">
                  Latents and text embeddings are model-specific; each stream is processed separately.
                </span>
              </div>
              <FolderConfigPanel
                folder={selectedFolder}
                onQueueProcessing={handleQueueProcessing}
                onQueueAudioProcessing={handleQueueAudioProcessing}
              />
              {selectedFolder.mediaType === 'audio' ? (
                <AudioProcessingStatus jobs={jobs} folderId={selectedFolder.id} stream={stream} />
              ) : (
                <ProcessingMatrix jobs={jobs} folderId={selectedFolder.id} stream={stream} />
              )}
            </>
          ) : (
            <EmptyState className="p-8">
              Select a folder from the Source Folders tab to configure processing.
            </EmptyState>
          )}
        </TabsContent>

        <TabsContent value="build" className="space-y-4">
          {buildError && (
            <div className="border-destructive/30 bg-destructive/10 text-destructive rounded-lg border p-3 text-sm">
              {buildError}
            </div>
          )}
          <DatasetBuilder
            completedJobs={completedJobs}
            datasets={datasets}
            onBuildDataset={handleBuildDataset}
            onDeleteDataset={handleDeleteDataset}
            onRefresh={handleRefresh}
            isRefreshing={isManualRefreshing}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
