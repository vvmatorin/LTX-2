'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiFetch, apiPost } from '@/lib/api';
import type { DpoChoice, DpoLabels, DpoRound, ProcessingJob } from '@/lib/types';

/**
 * Polls a training job's Live-DPO rounds. A round without labels is "pending":
 * the trainer is halted waiting for it, and the Runs page shows the labeling button.
 */
export function useDpoRounds(job: ProcessingJob | null) {
  const queryClient = useQueryClient();

  const jobId = job?.id ?? null;
  const dpoEnabled = !!job && job.type === 'training' && !!(job.config as { dpo?: { enabled?: boolean } }).dpo?.enabled;

  const query = useQuery<{ rounds: DpoRound[] }>({
    queryKey: ['dpo', jobId],
    queryFn: () => apiFetch(`/api/jobs/${jobId}/dpo`),
    enabled: dpoEnabled && jobId != null,
    refetchInterval: job?.status === 'running' ? 4000 : false,
    refetchIntervalInBackground: false,
  });

  const submit = useMutation({
    mutationFn: ({ step, choices }: { step: number; choices: DpoChoice[] }) =>
      apiPost<{ ok: boolean; labels: DpoLabels }>(`/api/jobs/${jobId}/dpo`, { step, choices }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dpo', jobId] });
    },
  });

  const rounds = query.data?.rounds ?? [];

  return {
    enabled: dpoEnabled,
    rounds,
    pendingRound: rounds.find(r => !r.labels) ?? null,
    submitLabels: submit.mutateAsync,
    isSubmitting: submit.isPending,
  };
}
