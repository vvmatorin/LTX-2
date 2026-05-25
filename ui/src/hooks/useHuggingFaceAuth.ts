"use client";

import { useState, useEffect, useCallback } from "react";

interface HuggingFaceAuthState {
  loggedIn: boolean;
  username: string | null;
  loading: boolean;
  error: string | null;
}

export function useHuggingFaceAuth() {
  const [state, setState] = useState<HuggingFaceAuthState>({
    loggedIn: false,
    username: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    fetch("/api/auth/huggingface")
      .then((r) => r.json())
      .then((data) => {
        if (data.loggedIn) {
          setState((prev) => ({
            ...prev,
            loggedIn: true,
            username: data.username ?? null,
          }));
        }
      })
      .catch(() => {});
  }, []);

  const login = useCallback(async (token: string) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const res = await fetch("/api/auth/huggingface", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      const data = await res.json();
      if (data.loggedIn) {
        setState({
          loggedIn: true,
          username: data.username ?? null,
          loading: false,
          error: null,
        });
        return true;
      }
      setState((prev) => ({
        ...prev,
        loading: false,
        error: data.error || "Login failed",
      }));
      return false;
    } catch {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: "Request failed",
      }));
      return false;
    }
  }, []);

  const logout = useCallback(async () => {
    await fetch("/api/auth/huggingface", { method: "DELETE" });
    setState({ loggedIn: false, username: null, loading: false, error: null });
  }, []);

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  return { ...state, login, logout, clearError };
}
