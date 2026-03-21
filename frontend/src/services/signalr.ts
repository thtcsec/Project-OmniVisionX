import * as signalR from "@microsoft/signalr";
import type { OmniEvent } from "@/types/omni";

// Same-origin: nginx (Docker) or Vite proxy forwards /hubs → omni-api. Override with OMNI_SIGNALR_URL if needed.
const SIGNALR_URL =
  import.meta.env.OMNI_SIGNALR_URL ?? import.meta.env.OMNI_API_BASE_URL ?? "";
const HUB_PATH = import.meta.env.OMNI_SIGNALR_HUB_PATH ?? "/hubs/omni";

let connection: signalR.HubConnection | null = null;
const listeners = new Set<(event: OmniEvent) => void>();
const statusListeners = new Set<(status: "connected" | "disconnected" | "reconnecting") => void>();

function notifyStatus(status: "connected" | "disconnected" | "reconnecting") {
  statusListeners.forEach((fn) => fn(status));
}

export function getConnection(): signalR.HubConnection {
  if (!connection) {
    connection = new signalR.HubConnectionBuilder()
      .withUrl(`${SIGNALR_URL}${HUB_PATH}`)
      .withAutomaticReconnect([0, 2000, 5000, 10000, 30000])
      .configureLogging(signalR.LogLevel.Warning)
      .build();

    connection.on("OmniEvent", (event: OmniEvent) => {
      listeners.forEach((fn) => fn(event));
    });

    connection.onreconnecting(() => notifyStatus("reconnecting"));
    connection.onreconnected(() => {
      notifyStatus("connected");
      connection?.invoke("SubscribeAll").catch(console.error);
    });
    connection.onclose(() => notifyStatus("disconnected"));
  }
  return connection;
}

export async function startConnection() {
  const conn = getConnection();
  if (conn.state === signalR.HubConnectionState.Disconnected) {
    try {
      await conn.start();
      notifyStatus("connected");
      await conn.invoke("SubscribeAll");
    } catch (err) {
      console.error("SignalR connection failed:", err);
      notifyStatus("disconnected");
    }
  }
}

export async function stopConnection() {
  if (connection && connection.state !== signalR.HubConnectionState.Disconnected) {
    await connection.stop();
  }
}

export async function joinCameraGroup(cameraId: string) {
  const conn = getConnection();
  if (conn.state === signalR.HubConnectionState.Connected) {
    await conn.invoke("JoinCameraGroup", cameraId);
  }
}

export async function leaveCameraGroup(cameraId: string) {
  const conn = getConnection();
  if (conn.state === signalR.HubConnectionState.Connected) {
    await conn.invoke("LeaveCameraGroup", cameraId);
  }
}

export function onOmniEvent(callback: (event: OmniEvent) => void) {
  listeners.add(callback);
  return () => { listeners.delete(callback); };
}

export function onConnectionStatus(callback: (status: "connected" | "disconnected" | "reconnecting") => void) {
  statusListeners.add(callback);
  return () => { statusListeners.delete(callback); };
}
