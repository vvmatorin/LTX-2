"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { AppSettings } from "@/lib/types";

export function useSettings() {
  const queryClient = useQueryClient();

  const query = useQuery<AppSettings>({
    queryKey: ["settings"],
    queryFn: async () => {
      const res = await fetch("/api/settings");
      return res.json();
    },
  });

  const mutation = useMutation({
    mutationFn: async (settings: Partial<AppSettings>) => {
      const res = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });

  return {
    settings: query.data,
    isLoading: query.isLoading,
    saveSettings: mutation.mutateAsync,
    isSaving: mutation.isPending,
  };
}
