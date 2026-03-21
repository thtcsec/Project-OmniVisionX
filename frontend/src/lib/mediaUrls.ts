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
