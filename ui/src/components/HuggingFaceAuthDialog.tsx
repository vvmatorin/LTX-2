"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { LogIn, Loader2 } from "lucide-react";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  loading: boolean;
  error: string | null;
  onLogin: (token: string) => Promise<boolean>;
  onClearError: () => void;
}

export function HuggingFaceAuthDialog({
  open,
  onOpenChange,
  loading,
  error,
  onLogin,
  onClearError,
}: Props) {
  const [token, setToken] = useState("");

  const handleLogin = async () => {
    const success = await onLogin(token);
    if (success) {
      setToken("");
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>HuggingFace Login</DialogTitle>
          <DialogDescription>
            Enter your HuggingFace access token. Get one from{" "}
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noopener"
              className="underline text-primary"
            >
              huggingface.co/settings/tokens
            </a>
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 pt-2">
          <Input
            type="password"
            placeholder="hf_..."
            value={token}
            onChange={(e) => {
              setToken(e.target.value);
              onClearError();
            }}
            className="font-mono text-xs"
          />
          {error && <p className="text-xs text-destructive">{error}</p>}
          <Button
            className="w-full"
            disabled={!token || loading}
            onClick={handleLogin}
          >
            {loading ? (
              <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
            ) : (
              <LogIn className="mr-1.5 h-4 w-4" />
            )}
            {loading ? "Authenticating..." : "Login"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
