"use client";

import { useRef, useEffect, useMemo } from "react";
import type { LossPoint } from "@/lib/logParsing";

interface Props {
  points: LossPoint[];
  className?: string;
}

function emaSmooth(values: number[], alpha: number): number[] {
  const out: number[] = [];
  let prev = values[0] ?? 0;
  for (const v of values) {
    prev = alpha * v + (1 - alpha) * prev;
    out.push(prev);
  }
  return out;
}

export function LossGraph({ points, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const smoothed = useMemo(() => {
    if (points.length < 2) return [];
    return emaSmooth(
      points.map((p) => p.loss),
      0.15,
    );
  }, [points]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length < 2) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;

    const pad = { top: 20, right: 12, bottom: 30, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const steps = points.map((p) => p.step);
    const losses = points.map((p) => p.loss);
    const minStep = steps[0];
    const maxStep = steps[steps.length - 1];
    const allValues = [...losses, ...smoothed];
    const minLoss = Math.min(...allValues) * 0.95;
    const maxLoss = Math.max(...allValues) * 1.05;

    const xScale = (v: number) => pad.left + ((v - minStep) / (maxStep - minStep || 1)) * plotW;
    const yScale = (v: number) => pad.top + (1 - (v - minLoss) / (maxLoss - minLoss || 1)) * plotH;

    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      const val = maxLoss - (i / 4) * (maxLoss - minLoss);
      ctx.fillText(val.toFixed(4), pad.left - 6, y + 3);
    }
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const x = pad.left + (plotW / 4) * i;
      const val = minStep + (i / 4) * (maxStep - minStep);
      ctx.fillText(Math.round(val).toString(), x, h - 8);
    }

    // Raw loss (faded)
    ctx.strokeStyle = "rgba(96,165,250,0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const x = xScale(steps[i]);
      const y = yScale(losses[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Smoothed loss
    if (smoothed.length > 0) {
      ctx.strokeStyle = "rgba(96,165,250,1)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < smoothed.length; i++) {
        const x = xScale(steps[i]);
        const y = yScale(smoothed[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }, [points, smoothed]);

  if (points.length < 2) {
    return (
      <div className={className}>
        <div className="surface-neo-inset flex h-full items-center justify-center rounded-2xl border border-border bg-background text-sm text-muted-foreground">
          Waiting for loss data...
        </div>
      </div>
    );
  }

  return (
    <div className={className}>
      <canvas ref={canvasRef} className="surface-neo-inset h-full w-full rounded-2xl border border-border bg-background" />
    </div>
  );
}
