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

/**
 * When the stream URL already targets MediaMTX (publisher path), HLS must use that path
 * segment (e.g. cam-cam1), not the DB camera id — the API skips re-registering those URLs.
 * Otherwise use camera id (matches API MediaMTX path registration for external RTSP).
 */
export function getCameraMediaKey(camera: { id: string; streamUrl?: string }): string {
  const raw = camera.streamUrl?.trim();
  if (!raw) return camera.id;
  try {
    const u = new URL(raw);
    if (u.protocol !== "rtsp:" && u.protocol !== "rtsps:") return camera.id;
    const host = u.hostname.toLowerCase();
    if (host !== "omni-mediamtx" && host !== "localhost" && host !== "127.0.0.1") return camera.id;
    let port = u.port ? parseInt(u.port, 10) : NaN;
    if (Number.isNaN(port)) {
      port = u.protocol === "rtsps:" ? 322 : 554;
    }
    if (port !== 8554 && port !== 18554) return camera.id;
    const seg = u.pathname
      .replace(/^\/+/, "")
      .split("/")
      .filter(Boolean)[0];
    return seg || camera.id;
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
