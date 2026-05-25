"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { TrainingDataset } from "@/lib/types";
import { parseApiError } from "@/lib/utils";

export function useDatasets() {
  const queryClient = useQueryClient();

  const query = useQuery<TrainingDataset[]>({
    queryKey: ["datasets"],
    queryFn: async () => {
      const res = await fetch("/api/datasets");
      return res.json();
    },
  });

  const createDataset = useMutation({
    mutationFn: async (dataset: { name: string; path: string; buckets: unknown[] }) => {
      const res = await fetch("/api/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dataset),
      });
      if (!res.ok) throw new Error(await parseApiError(res));
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const deleteDataset = useMutation({
    mutationFn: async (id: number) => {
      const res = await fetch(`/api/datasets?id=${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await parseApiError(res));
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  return {
    datasets: query.data || [],
    isLoading: query.isLoading,
    createDataset: createDataset.mutateAsync,
    deleteDataset: deleteDataset.mutateAsync,
    refreshDatasets: query.refetch,
    isRefreshing: query.isFetching,
  };
}
