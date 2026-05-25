"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { SourceFolder } from "@/lib/types";

export function useFolders() {
  const queryClient = useQueryClient();

  const query = useQuery<SourceFolder[]>({
    queryKey: ["folders"],
    queryFn: async () => {
      const res = await fetch("/api/folders");
      return res.json();
    },
  });

  const addFolder = useMutation({
    mutationFn: async (folderPath: string) => {
      const res = await fetch("/api/folders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: folderPath }),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["folders"] });
    },
  });

  const removeFolder = useMutation({
    mutationFn: async (id: number) => {
      const res = await fetch(`/api/folders?id=${id}`, { method: "DELETE" });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["folders"] });
    },
  });

  return {
    folders: query.data || [],
    isLoading: query.isLoading,
    addFolder: addFolder.mutateAsync,
    removeFolder: removeFolder.mutateAsync,
  };
}
