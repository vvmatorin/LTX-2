import { cn } from "@/lib/utils";

interface Props {
  children: React.ReactNode;
  className?: string;
}

export function EmptyState({ children, className }: Props) {
  return (
    <div
      className={cn(
        "rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground",
        className,
      )}
    >
      {children}
    </div>
  );
}
