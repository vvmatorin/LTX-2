'use client';

import { useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { Crown, Loader2, Pause, Play, ThumbsDown, Undo2 } from 'lucide-react';
import { cn, toErrorMessage } from '@/lib/utils';
import type { DpoChoice, DpoRound } from '@/lib/types';

interface RowChoice {
  best: number | null;
  worst: number | null;
  skipped: boolean;
}

interface Props {
  jobId: number;
  round: DpoRound;
  submitLabels: (args: { step: number; choices: DpoChoice[] }) => Promise<unknown>;
  isSubmitting: boolean;
}

/**
 * Best/worst labeling grid for one Live-DPO round: rows are samples, columns are seeds.
 * Labels are immutable — once the round carries labels (or a submit succeeds), the grid
 * renders read-only.
 */
export function DpoLabeling({ jobId, round, submitLabels, isSubmitting }: Props) {
  const readOnly = !!round.labels;

  const [choices, setChoices] = useState<Record<number, RowChoice>>(() => initialChoices(round));
  const [playingRow, setPlayingRow] = useState<number | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const videoRefs = useRef<Map<string, HTMLVideoElement>>(new Map());
  // Hover unmutes only one video at a time (rounds are generated with audio by default).
  const withAudio = !!round.pending.with_audio;
  const unmutedVideo = useRef<HTMLVideoElement | null>(null);

  const unmuteOnHover = (el: HTMLVideoElement) => {
    if (unmutedVideo.current && unmutedVideo.current !== el) unmutedVideo.current.muted = true;
    el.muted = false;
    unmutedVideo.current = el;
  };

  const muteOnLeave = (el: HTMLVideoElement) => {
    el.muted = true;
    if (unmutedVideo.current === el) unmutedVideo.current = null;
  };

  // Server labels win over local state so a labeled round always shows what was submitted.
  const effectiveChoices = useMemo(() => (readOnly ? initialChoices(round) : choices), [readOnly, round, choices]);

  const labeledCount = round.pending.samples.filter(s => isRowComplete(effectiveChoices[s.index])).length;
  const allComplete = labeledCount === round.pending.samples.length;

  const setRow = (index: number, patch: Partial<RowChoice>) => {
    if (readOnly) return;
    setChoices(prev => ({ ...prev, [index]: { ...prev[index], ...patch } }));
  };

  const pick = (index: number, kind: 'best' | 'worst', seedIdx: number) => {
    const row = effectiveChoices[index];
    const other = kind === 'best' ? 'worst' : 'best';
    setRow(index, {
      skipped: false,
      [kind]: row[kind] === seedIdx ? null : seedIdx,
      // Best and worst must differ; picking the same seed for the other role clears it.
      [other]: row[other] === seedIdx ? null : row[other],
    });
  };

  const toggleRowPlayback = (index: number) => {
    const next = playingRow === index ? null : index;
    for (const [key, video] of videoRefs.current) {
      const [rowStr] = key.split(':');
      const rowIdx = Number(rowStr);
      if (rowIdx === next) {
        video.currentTime = 0;
        void video.play().catch(() => {});
      } else {
        video.pause();
      }
    }
    setPlayingRow(next);
  };

  const handleSubmit = async () => {
    setSubmitError(null);
    try {
      await submitLabels({
        step: round.step,
        choices: round.pending.samples.map(s => {
          const row = effectiveChoices[s.index];
          return row.skipped
            ? { index: s.index, skipped: true }
            : { index: s.index, best: row.best, worst: row.worst, skipped: false };
        }),
      });
    } catch (err) {
      setSubmitError(toErrorMessage(err));
    }
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="bg-background/95 sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b px-4 py-2 backdrop-blur">
        <span className="text-sm font-medium">Step {round.step}</span>
        <Badge variant="outline" className="text-[10px]">
          {round.pending.samples.length} samples × {round.pending.num_seeds} seeds
        </Badge>
        {readOnly ? (
          <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-[10px] text-emerald-400">
            Submitted{round.labels?.submitted_at ? ` · ${new Date(round.labels.submitted_at).toLocaleString()}` : ''}
          </Badge>
        ) : (
          <span className="text-muted-foreground text-xs">
            Training is halted until labels are submitted. Pick the best and worst seed per sample, or skip rows
            without a clear winner.{withAudio && ' 🔊 Hover a video to hear its audio.'}
          </span>
        )}
        <div className="ml-auto flex items-center gap-3">
          {submitError && <span className="text-destructive text-xs">{submitError}</span>}
          {!readOnly && (
            <>
              <span className="text-muted-foreground text-xs tabular-nums">
                {labeledCount}/{round.pending.samples.length} labeled
              </span>
              <ConfirmDialog
                title="Submit labels?"
                description="Labels cannot be edited after submission. Training resumes immediately."
                confirmLabel="Submit"
                onConfirm={handleSubmit}
              >
                <Button size="sm" disabled={!allComplete || isSubmitting}>
                  {isSubmitting && <Loader2 className="mr-1.5 h-3 w-3 animate-spin" />}
                  Submit Labels
                </Button>
              </ConfirmDialog>
            </>
          )}
        </div>
      </div>

      <div className="space-y-6 p-4">
        {round.pending.samples.map(sample => {
          const row = effectiveChoices[sample.index];
          return (
            <div key={sample.index} className={cn('space-y-2', row.skipped && 'opacity-50')}>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 rounded-full p-0"
                  onClick={() => toggleRowPlayback(sample.index)}
                  title={playingRow === sample.index ? 'Pause row' : 'Play row'}
                >
                  {playingRow === sample.index ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                </Button>
                <span className="text-muted-foreground max-w-[70ch] truncate text-xs" title={sample.prompt}>
                  {sample.index + 1}. {sample.prompt}
                </span>
                {isRowComplete(row) && (
                  <Badge variant="outline" className="text-[10px]">
                    {row.skipped ? 'skipped' : 'labeled'}
                  </Badge>
                )}
                {!readOnly && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground ml-auto h-7 text-xs"
                    onClick={() => setRow(sample.index, { skipped: !row.skipped, best: null, worst: null })}
                  >
                    {row.skipped ? (
                      <>
                        <Undo2 className="mr-1 h-3 w-3" /> Unskip
                      </>
                    ) : (
                      'Skip'
                    )}
                  </Button>
                )}
              </div>
              <div
                className="grid gap-3"
                style={{ gridTemplateColumns: `repeat(${round.pending.num_seeds}, minmax(0, 1fr))` }}
              >
                {sample.videos.map((file, seedIdx) => {
                  const isBest = row.best === seedIdx;
                  const isWorst = row.worst === seedIdx;
                  return (
                    <div key={file} className="space-y-1.5">
                      <div
                        className={cn(
                          'relative overflow-hidden rounded-lg bg-black ring-2 ring-transparent transition-shadow',
                          isBest && 'ring-emerald-500',
                          isWorst && 'ring-red-500',
                        )}
                      >
                        <video
                          ref={el => {
                            const key = `${sample.index}:${seedIdx}`;
                            if (el) videoRefs.current.set(key, el);
                            else videoRefs.current.delete(key);
                          }}
                          src={`/api/jobs/${jobId}/dpo/video?path=${encodeURIComponent(`${round.dir}/${file}`)}`}
                          className="block w-full"
                          muted
                          loop
                          playsInline
                          preload="metadata"
                          onMouseEnter={withAudio ? e => unmuteOnHover(e.currentTarget) : undefined}
                          onMouseLeave={withAudio ? e => muteOnLeave(e.currentTarget) : undefined}
                        />
                        {(isBest || isWorst) && (
                          <span
                            className={cn(
                              'absolute top-1.5 left-1.5 rounded px-1.5 py-0.5 text-[10px] font-semibold text-white',
                              isBest ? 'bg-emerald-600/90' : 'bg-red-600/90',
                            )}
                          >
                            {isBest ? 'Best' : 'Worst'}
                          </span>
                        )}
                        <span className="text-muted-foreground absolute right-1.5 bottom-1.5 rounded bg-black/60 px-1 text-[10px]">
                          seed {sample.seeds[seedIdx]}
                        </span>
                      </div>
                      {!readOnly && !row.skipped && (
                        <div className="flex gap-1.5">
                          <Button
                            variant={isBest ? 'default' : 'outline'}
                            size="sm"
                            className={cn('h-6 flex-1 text-[10px]', isBest && 'bg-emerald-600 hover:bg-emerald-500')}
                            onClick={() => pick(sample.index, 'best', seedIdx)}
                          >
                            <Crown className="mr-1 h-3 w-3" /> Best
                          </Button>
                          <Button
                            variant={isWorst ? 'default' : 'outline'}
                            size="sm"
                            className={cn('h-6 flex-1 text-[10px]', isWorst && 'bg-red-600 hover:bg-red-500')}
                            onClick={() => pick(sample.index, 'worst', seedIdx)}
                          >
                            <ThumbsDown className="mr-1 h-3 w-3" /> Worst
                          </Button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function initialChoices(round: DpoRound): Record<number, RowChoice> {
  const bySample: Record<number, RowChoice> = {};
  for (const sample of round.pending.samples) {
    bySample[sample.index] = { best: null, worst: null, skipped: false };
  }
  for (const choice of round.labels?.choices ?? []) {
    if (bySample[choice.index]) {
      bySample[choice.index] = {
        best: choice.best ?? null,
        worst: choice.worst ?? null,
        skipped: !!choice.skipped,
      };
    }
  }
  return bySample;
}

function isRowComplete(row: RowChoice | undefined): boolean {
  if (!row) return false;
  return row.skipped || (row.best != null && row.worst != null && row.best !== row.worst);
}
