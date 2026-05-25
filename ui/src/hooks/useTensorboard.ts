"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { ProcessingJob } from "@/lib/types";

interface TensorboardState {
  running: boolean;
  port: number;
  logDir: string | null;
  error: string | null;
}

export function useTensorboard(activeJob: ProcessingJob | null) {
  const [state, setState] = useState<TensorboardState>({
    running: false,
    port: 0,
    logDir: null,
    error: null,
  });
  const lastStartedLogDir = useRef<string | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await fetch("/api/tensorboard");
      const data = await res.json();
      setState((prev) => ({ ...prev, running: data.running, port: data.port, logDir: data.logDir }));
    } catch {
      // network error, keep last state
    }
  }, []);

  const start = useCallback(async (logDir: string) => {
    setState((prev) => ({ ...prev, error: null }));
    try {
      const res = await fetch("/api/tensorboard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ logDir }),
      });
      const data = await res.json();
      if (data.error) {
        setState((prev) => ({ ...prev, error: data.error }));
      } else {
        setState({ running: true, port: data.port, logDir: data.logDir, error: null });
        lastStartedLogDir.current = logDir;
      }
    } catch (err) {
      setState((prev) => ({ ...prev, error: err instanceof Error ? err.message : "Failed to start" }));
    }
  }, []);

  const stop = useCallback(async () => {
    try {
      await fetch("/api/tensorboard", { method: "DELETE" });
      setState({ running: false, port: 0, logDir: null, error: null });
      lastStartedLogDir.current = null;
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, [poll]);

  useEffect(() => {
    if (activeJob && activeJob.type === "training" && activeJob.status === "running") {
      const config = activeJob.config as Record<string, unknown>;
      const outputDir = (config.outputDir as string) || "";
      if (outputDir) {
        const logDir = `${outputDir}/tensorboard`;
        if (!state.running && lastStartedLogDir.current !== logDir) {
          start(logDir);
        }
      }
    } else if (!activeJob && state.running) {
      stop();
    }
  }, [activeJob, state.running, start, stop]);

  return { ...state, start, stop };
}
