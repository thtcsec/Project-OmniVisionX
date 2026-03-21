import * as signalR from "@microsoft/signalr";
import { normalizeOmniEvent } from "@/lib/normalizeOmniEvent";
import type { OmniEvent } from "@/types/omni";

/**
 * Hub URL: prefer **same-origin** `/hubs/omni` so Vite/nginx can proxy WebSockets.
 * Do NOT fall back to `OMNI_API_BASE_URL` — that points at another origin (e.g. :8080)
 * while the SPA runs on :5173/:3000 and breaks the SignalR WebSocket.
 * Set `OMNI_SIGNALR_URL` only for explicit cross-origin hubs (full URL including path).
 */
function buildHubUrl(): string {
  const path = import.meta.env.OMNI_SIGNALR_HUB_PATH ?? "/hubs/omni";
  const pathPart = path.startsWith("/") ? path : `/${path}`;
  const explicit = import.meta.env.OMNI_SIGNALR_URL?.trim();
  if (explicit) {
    const base = explicit.replace(/\/$/, "");
    if (base.includes("/hubs")) return base;
    return `${base}${pathPart}`;
  }
  return pathPart;
}

const HUB_URL = buildHubUrl();

let connection: signalR.HubConnection | null = null;
const listeners = new Set<(event: OmniEvent) => void>();
const statusListeners = new Set<(status: "connected" | "disconnected" | "reconnecting") => void>();
const camerasChangedListeners = new Set<() => void>();

/** Camera groups the UI asked to join — re-applied after SignalR reconnect. */
const joinedCameraIds = new Set<string>();

/** Single in-flight connect so concurrent `start()` never throws "not in Disconnected state". */
let connectInFlight: Promise<void> | null = null;

function notifyStatus(status: "connected" | "disconnected" | "reconnecting") {
  statusListeners.forEach((fn) => fn(status));
}

function getCurrentStatus(): "connected" | "disconnected" | "reconnecting" {
  const conn = connection;
  if (!conn) return "disconnected";
  if (conn.state === signalR.HubConnectionState.Connected) return "connected";
  if (conn.state === signalR.HubConnectionState.Reconnecting || conn.state === signalR.HubConnectionState.Connecting) return "reconnecting";
  return "disconnected";
}

async function rejoinCameraGroups() {
  const conn = connection;
  if (!conn || conn.state !== signalR.HubConnectionState.Connected) return;
  for (const id of joinedCameraIds) {
    try {
      await conn.invoke("JoinCameraGroup", id);
    } catch (e) {
      console.warn("JoinCameraGroup after reconnect failed:", id, e);
    }
  }
}

export function getConnection(): signalR.HubConnection {
  if (!connection) {
    connection = new signalR.HubConnectionBuilder()
      .withUrl(HUB_URL)
      .withAutomaticReconnect([0, 2000, 5000, 10000, 30000])
      .configureLogging(signalR.LogLevel.Warning)
      .build();

    connection.on("OmniEvent", (event: unknown) => {
      const normalized = normalizeOmniEvent(event);
      listeners.forEach((fn) => fn(normalized));
    });

    connection.on("CamerasChanged", () => {
      camerasChangedListeners.forEach((fn) => fn());
    });

    connection.onreconnecting(() => notifyStatus("reconnecting"));
    connection.onreconnected(async () => {
      notifyStatus("connected");
      try {
        await connection?.invoke("SubscribeAll");
        await rejoinCameraGroups();
      } catch (e) {
        console.error("SignalR post-reconnect subscribe failed:", e);
      }
    });
    connection.onclose(() => notifyStatus("disconnected"));
  }
  return connection;
}

/** Wait until hub is connected (starts hub if needed). Safe for concurrent callers. */
export async function ensureSignalRConnected(): Promise<void> {
  const conn = getConnection();
  if (conn.state === signalR.HubConnectionState.Connected) return;

  if (connectInFlight) {
    await connectInFlight;
    if (conn.state === signalR.HubConnectionState.Connected) return;
    return ensureSignalRConnected();
  }

  connectInFlight = (async () => {
    const c = getConnection();
    const deadline = Date.now() + 60_000;

    while (Date.now() < deadline) {
      if (c.state === signalR.HubConnectionState.Connected) {
        await c.invoke("SubscribeAll");
        return;
      }
      if (c.state === signalR.HubConnectionState.Disconnected) {
        await c.start();
        notifyStatus("connected");
        await c.invoke("SubscribeAll");
        return;
      }
      // Connecting | Reconnecting | Disconnecting — poll; never call start() twice
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error("SignalR: connection timeout");
  })();

  try {
    await connectInFlight;
  } finally {
    connectInFlight = null;
  }
}

export async function startConnection() {
  try {
    await ensureSignalRConnected();
  } catch (err) {
    console.error("SignalR connection failed:", err);
    notifyStatus("disconnected");
  }
}

export async function stopConnection() {
  joinedCameraIds.clear();
  if (connection && connection.state !== signalR.HubConnectionState.Disconnected) {
    await connection.stop();
  }
}

export async function joinCameraGroup(cameraId: string) {
  joinedCameraIds.add(cameraId);
  try {
    await ensureSignalRConnected();
    await getConnection().invoke("JoinCameraGroup", cameraId);
  } catch (e) {
    console.error("joinCameraGroup failed:", cameraId, e);
  }
}

export async function leaveCameraGroup(cameraId: string) {
  joinedCameraIds.delete(cameraId);
  const conn = getConnection();
  if (conn.state === signalR.HubConnectionState.Connected) {
    try {
      await conn.invoke("LeaveCameraGroup", cameraId);
    } catch (e) {
      console.warn("leaveCameraGroup:", e);
    }
  }
}

export function onOmniEvent(callback: (event: OmniEvent) => void) {
  listeners.add(callback);
  return () => {
    listeners.delete(callback);
  };
}

export function onConnectionStatus(callback: (status: "connected" | "disconnected" | "reconnecting") => void) {
  statusListeners.add(callback);
  callback(getCurrentStatus());
  return () => {
    statusListeners.delete(callback);
  };
}

/** Invalidate camera queries when API mutates list (multi-tab / other clients). */
export function onCamerasChanged(callback: () => void) {
  camerasChangedListeners.add(callback);
  return () => {
    camerasChangedListeners.delete(callback);
  };
}
