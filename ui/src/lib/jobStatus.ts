import {
  Clock,
  Loader2,
  CheckCircle2,
  XCircle,
  type LucideIcon,
} from "lucide-react";

export interface StatusStyle {
  icon: LucideIcon;
  label: string;
  class: string;
}

export const JOB_STATUS: Record<string, StatusStyle> = {
  queued: {
    icon: Clock,
    label: "Queued",
    class: "bg-yellow-500/10 text-yellow-400 border-yellow-500/30",
  },
  running: {
    icon: Loader2,
    label: "Running",
    class: "bg-blue-500/10 text-blue-400 border-blue-500/30",
  },
  completed: {
    icon: CheckCircle2,
    label: "Done",
    class: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
  },
  failed: {
    icon: XCircle,
    label: "Failed",
    class: "bg-red-500/10 text-red-400 border-red-500/30",
  },
  cancelled: {
    icon: XCircle,
    label: "Cancelled",
    class: "bg-muted text-muted-foreground border-border",
  },
  pending: {
    icon: Clock,
    label: "Pending",
    class: "bg-muted text-muted-foreground border-border",
  },
} as const;
