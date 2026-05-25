"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { ProcessingJob } from "@/lib/types";

export function useJobs() {
  const queryClient = useQueryClient();

  const query = useQuery<ProcessingJob[]>({
    queryKey: ["jobs"],
    queryFn: async () => {
      const res = await fetch("/api/jobs");
      return res.json();
    },
    refetchInterval: 2000,
  });

  const createJob = useMutation({
    mutationFn: async (job: { type: string; name: string; config: Record<string, unknown> }) => {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(job),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const stopJob = useMutation({
    mutationFn: async (id: number) => {
      const res = await fetch(`/api/jobs/${id}/stop`, { method: "POST" });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  return {
    jobs: query.data || [],
    isLoading: query.isLoading,
    createJob: createJob.mutateAsync,
    stopJob: stopJob.mutateAsync,
    refreshJobs: query.refetch,
    isRefreshingJobs: query.isFetching,
  };
}
