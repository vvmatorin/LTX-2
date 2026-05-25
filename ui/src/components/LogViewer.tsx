"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { ArrowDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  lines: string[];
  className?: string;
}

export function LogViewer({ lines, className }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, autoScroll]);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 40;
    setAutoScroll(atBottom);
  };

  return (
    <div className={cn("surface-neo relative flex flex-col rounded-2xl border border-border bg-background", className)}>
      <div className="flex items-center justify-between border-b border-white/10 px-3 py-1.5">
        <span className="text-xs font-medium text-muted-foreground">Output</span>
        <span className="text-[10px] text-muted-foreground tabular-nums">
          {lines.length} lines
        </span>
      </div>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-auto p-3 font-mono text-[11px] leading-5"
        style={{ minHeight: 200, maxHeight: 400 }}
      >
        {lines.map((line, i) => (
          <div
            key={i}
            className={cn(
              "whitespace-pre-wrap break-all",
              line.includes("Error") || line.includes("error") || line.includes("CUDA out of memory")
                ? "text-red-400"
                : line.includes("Done") || line.includes("100.0%")
                  ? "text-emerald-400"
                  : "text-muted-foreground",
            )}
          >
            {line}
          </div>
        ))}
        {lines.length === 0 && (
          <div className="flex h-32 items-center justify-center text-muted-foreground">
            Waiting for output...
          </div>
        )}
      </div>
      {!autoScroll && (
        <Button
          size="sm"
          variant="secondary"
          className="absolute bottom-4 right-4 z-10 h-7 gap-1 text-xs shadow-lg"
          onClick={() => {
            setAutoScroll(true);
            scrollRef.current?.scrollTo({
              top: scrollRef.current.scrollHeight,
              behavior: "smooth",
            });
          }}
        >
          <ArrowDown className="h-3 w-3" />
          Scroll to bottom
        </Button>
      )}
    </div>
  );
}
