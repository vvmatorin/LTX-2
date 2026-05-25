"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiPost } from "@/lib/api";

export function useRuns() {
  const queryClient = useQueryClient();

  const createRun = useMutation({
    mutationFn: (run: {
      name: string;
      config: Record<string, unknown>;
      gpuMode: string;
      gpuIds: string;
      datasetName: string;
    }) => apiPost("/api/runs", run),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  return {
    createRun: createRun.mutateAsync,
    isCreating: createRun.isPending,
  };
}
