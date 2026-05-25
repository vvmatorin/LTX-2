'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { SourceFolder } from '@/lib/types';
import { apiFetch, apiPost, apiDelete } from '@/lib/api';

export function useFolders() {
  const queryClient = useQueryClient();

  const query = useQuery<SourceFolder[]>({
    queryKey: ['folders'],
    queryFn: () => apiFetch<SourceFolder[]>('/api/folders'),
  });

  const addFolder = useMutation({
    mutationFn: (folderPath: string) => apiPost<SourceFolder>('/api/folders', { path: folderPath }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders'] });
    },
  });

  const removeFolder = useMutation({
    mutationFn: (id: number) => apiDelete<{ ok: boolean }>(`/api/folders?id=${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders'] });
    },
  });

  return {
    folders: query.data || [],
    isError: query.isError,
    error: query.error,
    addFolder: addFolder.mutateAsync,
    removeFolder: removeFolder.mutateAsync,
  };
}
