/**
 * Runtime config. In Docker, nginx proxies `/api` and `/hubs` to `omni-api`, so
 * empty base uses same-origin relative URLs.
 */
export function getApiBaseUrl(): string {
  const v = import.meta.env.OMNI_API_BASE_URL;
  return (v ?? "").replace(/\/$/, "");
}

export function getSignalRHubPath(): string {
  return import.meta.env.OMNI_SIGNALR_HUB_PATH || "/hubs/omni";
}

/** Full URL for SignalR (WebSocket). */
export function getSignalRHubUrl(): string {
  const path = getSignalRHubPath().startsWith("/")
    ? getSignalRHubPath()
    : `/${getSignalRHubPath()}`;
  const base = getApiBaseUrl();
  if (base) return `${base}${path}`;
  if (typeof window !== "undefined") {
    return `${window.location.origin}${path}`;
  }
  return path;
}
