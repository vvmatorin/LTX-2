"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { ProcessingJob } from "@/lib/types";

interface TensorboardState {
  running: boolean;
  ready: boolean;
  port: number;
  logDir: string | null;
  error: string | null;
}

const TB_READY_POLL_MS = 500;
const TB_READY_TIMEOUT_MS = 15_000;

export function useTensorboard(activeJob: ProcessingJob | null) {
  const [state, setState] = useState<TensorboardState>({
    running: false,
    ready: false,
    port: 0,
    logDir: null,
    error: null,
  });
  const lastStartedLogDir = useRef<string | null>(null);
  const readinessTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearReadinessProbe = useCallback(() => {
    if (readinessTimer.current) {
      clearTimeout(readinessTimer.current);
      readinessTimer.current = null;
    }
  }, []);

  const probeReadiness = useCallback(() => {
    clearReadinessProbe();
    const deadline = Date.now() + TB_READY_TIMEOUT_MS;

    const check = async () => {
      try {
        const res = await fetch("/tensorboard/", { method: "HEAD" });
        if (res.ok) {
          setState((prev) => ({ ...prev, ready: true }));
          return;
        }
      } catch {
        // not ready yet
      }
      if (Date.now() < deadline) {
        readinessTimer.current = setTimeout(check, TB_READY_POLL_MS);
      } else {
        setState((prev) => ({ ...prev, error: "TensorBoard is taking too long to respond" }));
      }
    };
    check();
  }, [clearReadinessProbe]);

  const wasRunning = useRef(false);

  const poll = useCallback(async () => {
    try {
      const res = await fetch("/api/tensorboard");
      const data = await res.json();
      const nowRunning = data.running as boolean;

      if (nowRunning && !wasRunning.current) {
        probeReadiness();
      }
      wasRunning.current = nowRunning;

      setState((prev) => {
        if (!nowRunning && prev.running) {
          return { ...prev, running: false, ready: false, port: data.port, logDir: data.logDir };
        }
        return { ...prev, running: nowRunning, port: data.port, logDir: data.logDir };
      });
    } catch {
      // network error, keep last state
    }
  }, [probeReadiness]);

  const start = useCallback(
    async (logDir: string) => {
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
          setState({
            running: true,
            ready: false,
            port: data.port,
            logDir: data.logDir,
            error: null,
          });
          probeReadiness();
          lastStartedLogDir.current = logDir;
        }
      } catch (err) {
        setState((prev) => ({
          ...prev,
          error: err instanceof Error ? err.message : "Failed to start",
        }));
      }
    },
    [probeReadiness],
  );

  const stop = useCallback(async () => {
    clearReadinessProbe();
    try {
      await fetch("/api/tensorboard", { method: "DELETE" });
      setState({ running: false, ready: false, port: 0, logDir: null, error: null });
      lastStartedLogDir.current = null;
    } catch {
      // ignore
    }
  }, [clearReadinessProbe]);

  useEffect(() => {
    poll();
    const interval = setInterval(poll, 3000);
    return () => {
      clearInterval(interval);
      clearReadinessProbe();
    };
  }, [poll, clearReadinessProbe]);

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
