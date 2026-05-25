"use client";

import type { SourceFolder } from "@/lib/types";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Folder, FileVideo, FileImage, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  folder: SourceFolder;
  selected: boolean;
  onSelect: () => void;
  onRemove: () => void;
}

export function SourceFolderCard({ folder, selected, onSelect, onRemove }: Props) {
  const Icon = folder.mediaType === "images" ? FileImage : FileVideo;

  return (
    <Card
      className={cn(
        "cursor-pointer transition-colors hover:border-primary/40",
        selected && "border-primary ring-1 ring-primary/30",
      )}
      onClick={onSelect}
    >
      <CardContent className="flex items-center gap-3 px-4 py-2.5">
        <div className="rounded-md bg-muted p-1.5">
          <Folder className="h-4 w-4 text-muted-foreground" />
        </div>

        <h3 className="min-w-0 flex-1 break-all text-sm font-medium font-mono">
          {folder.path}
        </h3>

        <Badge variant="secondary" className="shrink-0 text-[10px]">
          <Icon className="mr-1 h-3 w-3" />
          {folder.fileCount}
        </Badge>

        <ConfirmDialog
          title="Remove folder?"
          description={`This will remove "${folder.path}" from the list. Existing preprocessed data and files on disk will not be affected.`}
          confirmLabel="Remove"
          onConfirm={onRemove}
        >
          <Button
            size="sm"
            variant="ghost"
            className="h-7 w-7 p-0 text-destructive hover:text-destructive shrink-0"
            onClick={(e) => { e.stopPropagation(); }}
            title="Remove folder"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </ConfirmDialog>
      </CardContent>
    </Card>
  );
}
