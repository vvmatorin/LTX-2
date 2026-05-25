"use client";

import { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";
import { cn } from "@/lib/utils";

interface Props {
  jobId: number | null;
  className?: string;
}

const TERMINAL_THEME = {
  background: "#0d1220",
  foreground: "#e2e8f0",
  cursor: "#64748b",
  cursorAccent: "#0d1220",
  selectionBackground: "#334155",
  black: "#1e293b",
  red: "#f87171",
  green: "#34d399",
  yellow: "#fbbf24",
  blue: "#60a5fa",
  magenta: "#c084fc",
  cyan: "#22d3ee",
  white: "#e2e8f0",
  brightBlack: "#475569",
  brightRed: "#fca5a5",
  brightGreen: "#6ee7b7",
  brightYellow: "#fcd34d",
  brightBlue: "#93c5fd",
  brightMagenta: "#d8b4fe",
  brightCyan: "#67e8f9",
  brightWhite: "#f8fafc",
} as const;

export function LogViewer({ jobId, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);

  useEffect(() => {
    const host = containerRef.current;
    if (!host) return;

    const term = new Terminal({
      convertEol: true,
      cursorBlink: false,
      disableStdin: true,
      fontFamily:
        "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
      fontSize: 12,
      lineHeight: 1.25,
      scrollback: 10_000,
      theme: TERMINAL_THEME,
      allowProposedApi: true,
    });

    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(host);

    const safeFit = () => {
      try {
        fit.fit();
      } catch {
        /* container not visible yet */
      }
    };

    requestAnimationFrame(safeFit);

    const resizeObserver = new ResizeObserver(safeFit);
    resizeObserver.observe(host);

    termRef.current = term;
    fitRef.current = fit;

    return () => {
      resizeObserver.disconnect();
      term.dispose();
      termRef.current = null;
      fitRef.current = null;
    };
  }, []);

  useEffect(() => {
    const term = termRef.current;
    if (!term) return;

    term.clear();
    if (!jobId) return;

    const es = new EventSource(`/api/jobs/${jobId}/logs?mode=sse`);

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data === "__DONE__") {
        es.close();
        return;
      }
      if (typeof data === "string" && data.length > 0) {
        term.write(data);
      }
    };

    es.onerror = () => es.close();

    return () => es.close();
  }, [jobId]);

  return (
    <div
      className={cn(
        "surface-neo flex flex-col overflow-hidden rounded-2xl border border-border",
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-white/10 px-3 py-1.5">
        <span className="text-xs font-medium text-muted-foreground">Output</span>
      </div>
      <div
        ref={containerRef}
        className="flex-1 p-2"
        style={{ background: TERMINAL_THEME.background, height: 360 }}
      />
    </div>
  );
}
