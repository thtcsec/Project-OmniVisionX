/**
 * MediaMTX defaults (see media-server/mediamtx.yml):
 * - HLS:  port 8888
 * - WebRTC / WHEP: port 8889
 * Do not use the HLS base URL for /whep — it will always fail.
 */

function trimBase(url: string) {
  return url.replace(/\/$/, "");
}

/** HLS + .m3u8 playlist base (default http://localhost:8888) */
export function getMediaHlsBase(): string {
  return trimBase(import.meta.env.OMNI_MEDIA_BASE_URL ?? "http://localhost:8888");
}

/** WHEP endpoint base (default same host as HLS but port 8889) */
export function getMediaWebRtcBase(): string {
  const explicit = import.meta.env.OMNI_MEDIA_WEBRTC_BASE_URL?.trim();
  if (explicit) return trimBase(explicit);
  const hls = getMediaHlsBase();
  try {
    const u = new URL(hls);
    u.port = "8889";
    return u.origin;
  } catch {
    return "http://localhost:8889";
  }
}

export function buildDefaultHlsUrl(cameraId: string): string {
  return `${getMediaHlsBase()}/${encodeURIComponent(cameraId)}/index.m3u8`;
}

export function buildDefaultWebRtcUrl(cameraId: string): string {
  return `${getMediaWebRtcBase()}/${encodeURIComponent(cameraId)}/whep`;
}

export function getCameraMediaKey(camera: { id: string; streamUrl?: string }): string {
  const raw = camera.streamUrl?.trim();
  if (!raw) return camera.id;

  try {
    const u = new URL(raw);
    const isRtsp = u.protocol === "rtsp:" || u.protocol === "rtsps:";
    if (!isRtsp) return camera.id;

    const host = u.hostname.toLowerCase();
    const isMediaMtxHost = host === "omni-mediamtx" || host === "localhost" || host === "127.0.0.1";
    if (!isMediaMtxHost) return camera.id;

    const port = u.port || (u.protocol === "rtsps:" ? "322" : "554");
    const isMediaMtxPort = port === "8554" || port === "18554";
    if (!isMediaMtxPort) return camera.id;

    const path = u.pathname.replace(/^\/+/, "").trim();
    if (!path) return camera.id;
    return path;
  } catch {
    return camera.id;
  }
}

export function buildCameraHlsUrl(camera: { id: string; streamUrl?: string }): string {
  const key = getCameraMediaKey(camera);
  return `${getMediaHlsBase()}/${encodeURIComponent(key)}/index.m3u8`;
}

export function buildCameraWebRtcUrl(camera: { id: string; streamUrl?: string }): string {
  const key = getCameraMediaKey(camera);
  return `${getMediaWebRtcBase()}/${encodeURIComponent(key)}/whep`;
}
