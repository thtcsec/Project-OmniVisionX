import type { OmniEvent } from "@/types/omni";

export type TrackOverlay = {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  confidence: number;
  lastSeen: number;
  kind: OmniEvent["type"];
  /** Inference frame size (bbox pixel space). When set, overlay must use these instead of video.videoWidth/Height. */
  sourceFrameW?: number;
  sourceFrameH?: number;
};

/** Parse hub `data` whether it arrives as JSON string or pre-parsed object. */
export function parseOmniEventPayload(data: OmniEvent["data"]): Record<string, unknown> | null {
  if (data == null) return null;
  if (typeof data === "string") {
    try {
      return JSON.parse(data) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  if (typeof data === "object" && !Array.isArray(data)) {
    return data as Record<string, unknown>;
  }
  return null;
}

function parseBboxParts(d: Record<string, unknown>): [number, number, number, number] | null {
  const raw = d.bbox;
  if (typeof raw === "string") {
    const parts = raw.split(",").map((x) => parseFloat(x.trim()));
    if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) return null;
    return [parts[0], parts[1], parts[2], parts[3]];
  }
  if (Array.isArray(raw) && raw.length === 4) {
    const parts = raw.map((x) => (typeof x === "number" ? x : parseFloat(String(x))));
    if (parts.some((n) => Number.isNaN(n))) return null;
    return [parts[0], parts[1], parts[2], parts[3]];
  }
  return null;
}

export function parseBboxFromOmniEvent(event: OmniEvent): Omit<TrackOverlay, "lastSeen"> | null {
  try {
    const d = parseOmniEventPayload(event.data);
    if (!d) return null;
    const coords = parseBboxParts(d);
    if (!coords) return null;
    const [x1, y1, x2, y2] = coords;
    const id = String(
      d.globalTrackId ?? d.track_id ?? `${event.type}-${event.timestamp}-${d.label ?? "obj"}`,
    );
    const label =
      (typeof d.plateText === "string" && d.plateText) ||
      (typeof d.label === "string" && d.label) ||
      (typeof d.faceIdentity === "string" && d.faceIdentity) ||
      event.type;
    const confidence = parseFloat(String(d.confidence ?? "0")) || 0;
    const fwRaw = d.frameWidth ?? d.FrameWidth ?? d.frame_width;
    const fhRaw = d.frameHeight ?? d.FrameHeight ?? d.frame_height;
    const sourceFrameW = typeof fwRaw === "number" ? fwRaw : parseFloat(String(fwRaw ?? ""));
    const sourceFrameH = typeof fhRaw === "number" ? fhRaw : parseFloat(String(fhRaw ?? ""));
    const overlay: Omit<TrackOverlay, "lastSeen"> = {
      id,
      x1,
      y1,
      x2,
      y2,
      label,
      confidence,
      kind: event.type,
    };
    if (Number.isFinite(sourceFrameW) && sourceFrameW > 0) overlay.sourceFrameW = sourceFrameW;
    if (Number.isFinite(sourceFrameH) && sourceFrameH > 0) overlay.sourceFrameH = sourceFrameH;
    return overlay;
  } catch {
    return null;
  }
}
