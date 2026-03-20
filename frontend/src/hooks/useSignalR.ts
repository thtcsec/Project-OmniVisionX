import { useEffect, useState, useCallback } from "react";
import { startConnection, stopConnection, onOmniEvent, onConnectionStatus } from "@/services/signalr";
import type { OmniEvent } from "@/types/omni";

export type ConnectionStatus = "connected" | "disconnected" | "reconnecting";

export function useSignalR() {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [events, setEvents] = useState<OmniEvent[]>([]);

  useEffect(() => {
    startConnection();
    const unsubEvent = onOmniEvent((event) => {
      setEvents((prev) => [event, ...prev].slice(0, 100));
    });
    const unsubStatus = onConnectionStatus(setStatus);
    return () => {
      unsubEvent();
      unsubStatus();
      stopConnection();
    };
  }, []);

  const clearEvents = useCallback(() => setEvents([]), []);

  return { status, events, clearEvents };
}
