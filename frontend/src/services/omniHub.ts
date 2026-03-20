import { HubConnection, HubConnectionBuilder, LogLevel } from "@microsoft/signalr";
import { getSignalRHubUrl } from "./env";

export type OmniEventPayload = {
  type: string;
  cameraId: string;
  data: string;
  timestamp: string;
};

export function createOmniHubConnection(): HubConnection {
  const url = getSignalRHubUrl();
  return new HubConnectionBuilder()
    .withUrl(url, { withCredentials: false })
    .withAutomaticReconnect()
    .configureLogging(LogLevel.Warning)
    .build();
}
