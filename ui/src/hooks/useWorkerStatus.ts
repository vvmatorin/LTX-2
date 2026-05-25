"use client";

import { useQuery } from "@tanstack/react-query";

interface WorkerStatus {
  alive: boolean;
  lastSeen: string | null;
}

export function useWorkerStatus() {
  const query = useQuery<WorkerStatus>({
    queryKey: ["worker-status"],
    queryFn: async () => {
      const res = await fetch("/api/worker/status");
      return res.json();
    },
    refetchInterval: 3000,
  });

  return {
    alive: query.data?.alive ?? false,
    lastSeen: query.data?.lastSeen ?? null,
    isLoading: query.isLoading,
  };
}
