import { useEffect, useState, useCallback } from "react";
import { startConnection, onOmniEvent, onConnectionStatus } from "@/services/signalr";
import type { OmniEvent } from "@/types/omni";

export type ConnectionStatus = "connected" | "disconnected" | "reconnecting";

export function useSignalR() {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [events, setEvents] = useState<OmniEvent[]>([]);

  useEffect(() => {
    const unsubStatus = onConnectionStatus(setStatus);
    const unsubEvent = onOmniEvent((event) => {
      setEvents((prev) => [event, ...prev].slice(0, 100));
    });
    void startConnection();
    return () => {
      unsubStatus();
      unsubEvent();
      // Do not stopConnection() here — React Strict Mode runs cleanup between mounts and
      // aborts the hub mid-handshake ("stopped before the hub handshake could complete").
      // Hub is a process-wide singleton; it can stay up for the SPA session.
    };
  }, []);

  const clearEvents = useCallback(() => setEvents([]), []);

  return { status, events, clearEvents };
}
