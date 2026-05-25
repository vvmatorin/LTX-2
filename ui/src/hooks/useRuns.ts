"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";

export function useRuns() {
  const queryClient = useQueryClient();

  const createRun = useMutation({
    mutationFn: async (run: {
      name: string;
      config: Record<string, unknown>;
      gpuMode: string;
      gpuIds: string;
      datasetName: string;
    }) => {
      const res = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(run),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  return {
    createRun: createRun.mutateAsync,
    isCreating: createRun.isPending,
  };
}
