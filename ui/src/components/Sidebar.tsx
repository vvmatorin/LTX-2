"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Database,
  GraduationCap,
  Play,
  Settings,
  Layers,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/training", label: "Training", icon: GraduationCap },
  { href: "/runs", label: "Runs", icon: Play },
] as const;

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="relative z-10 flex h-full w-[15.5rem] flex-col rounded-r-3xl border-r border-sidebar-border bg-sidebar p-2 shadow-[12px_0_30px_oklch(0.08_0.03_258_/_0.52)]">
      <div className="surface-neo flex items-center gap-2.5 rounded-2xl px-4 py-4">
        <div className="ring-glow flex h-9 w-9 items-center justify-center rounded-xl bg-primary">
          <Layers className="h-4 w-4 text-primary-foreground" />
        </div>
        <div>
          <h1 className="title-gradient text-lg font-bold leading-none tracking-tight">LTX Trainer</h1>
          <p className="mt-1 text-[0.78rem] text-sidebar-foreground/78">Training UI</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-1 py-4">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-[0.95rem] font-medium transition-all duration-200",
                active
                  ? "ring-glow bg-sidebar-accent text-sidebar-accent-foreground shadow-[inset_0_1px_0_oklch(1_0_0_/_0.12),0_6px_18px_oklch(0.07_0.03_258_/_0.55)]"
                  : "text-sidebar-foreground/82 hover:bg-sidebar-accent/65 hover:text-sidebar-foreground",
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-sidebar-border px-1 py-3">
        <Link
          href="/settings"
          className={cn(
            "flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-[0.95rem] font-medium transition-all duration-200",
            pathname.startsWith("/settings")
              ? "ring-glow bg-sidebar-accent text-sidebar-accent-foreground shadow-[inset_0_1px_0_oklch(1_0_0_/_0.12),0_6px_18px_oklch(0.07_0.03_258_/_0.55)]"
              : "text-sidebar-foreground/82 hover:bg-sidebar-accent/65 hover:text-sidebar-foreground",
          )}
        >
          <Settings className="h-4 w-4 shrink-0" />
          Settings
        </Link>
      </div>
    </aside>
  );
}
