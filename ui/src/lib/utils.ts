import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Extracts a human-readable error message from a non-OK fetch Response.
 * JSON bodies with an `error` field are unwrapped; HTML or other content
 * is replaced with a generic "<status> <statusText>" string so raw HTML
 * is never surfaced in the UI.
 */
export async function parseApiError(res: Response): Promise<string> {
  const contentType = res.headers.get("content-type") ?? ""
  if (contentType.includes("application/json")) {
    try {
      const json = await res.json()
      if (typeof json?.error === "string") return json.error
      return JSON.stringify(json)
    } catch {
      // fall through
    }
  }
  return `Server error ${res.status}${res.statusText ? `: ${res.statusText}` : ""}`
}
