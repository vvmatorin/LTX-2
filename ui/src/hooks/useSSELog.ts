"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export function useSSELog(jobId: number | null) {
  const [lines, setLines] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const connect = useCallback(() => {
    if (!jobId) return;

    eventSourceRef.current?.close();

    const es = new EventSource(`/api/jobs/${jobId}/logs?mode=sse`);
    eventSourceRef.current = es;

    es.onopen = () => setConnected(true);

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data === "__DONE__") {
        es.close();
        setConnected(false);
        return;
      }
      if (typeof data === "string" && data.length > 0) {
        const newLines = data.split("\n").filter((l: string) => l.length > 0);
        setLines((prev) => [...prev, ...newLines]);
      }
    };

    es.onerror = () => {
      setConnected(false);
      es.close();
    };
  }, [jobId]);

  useEffect(() => {
    connect();
    return () => {
      eventSourceRef.current?.close();
    };
  }, [connect]);

  const reset = useCallback(() => {
    setLines([]);
    connect();
  }, [connect]);

  return { lines, connected, reset };
}
