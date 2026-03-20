import { useCallback, useEffect, useRef, useState } from "react";
import type { HubConnection } from "@microsoft/signalr";
import { createOmniHubConnection, type OmniEventPayload } from "@/services/omniHub";

type Status = "idle" | "connecting" | "live" | "error";

export function useOmniHub(onEvent?: (ev: OmniEventPayload) => void) {
  const [status, setStatus] = useState<Status>("idle");
  const [lastError, setLastError] = useState<string | null>(null);
  const connRef = useRef<HubConnection | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(async () => {
    setLastError(null);
    setStatus("connecting");
    const conn = createOmniHubConnection();
    connRef.current = conn;

    conn.on("OmniEvent", (...args: unknown[]) => {
      const raw = args[0] as Record<string, unknown>;
      onEventRef.current?.({
        type: String(raw.type ?? ""),
        cameraId: String(raw.cameraId ?? ""),
        data: typeof raw.data === "string" ? raw.data : JSON.stringify(raw.data ?? {}),
        timestamp: String(raw.timestamp ?? ""),
      });
    });

    try {
      await conn.start();
      await conn.invoke("SubscribeAll");
      setStatus("live");
    } catch (e) {
      setStatus("error");
      setLastError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const disconnect = useCallback(async () => {
    const c = connRef.current;
    connRef.current = null;
    if (c) {
      try {
        await c.stop();
      } catch {
        /* ignore */
      }
    }
    setStatus("idle");
  }, []);

  useEffect(() => {
    return () => {
      void disconnect();
    };
  }, [disconnect]);

  return { status, lastError, connect, disconnect };
}
