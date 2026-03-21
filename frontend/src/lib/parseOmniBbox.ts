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
};

export function parseBboxFromOmniEvent(event: OmniEvent): Omit<TrackOverlay, "lastSeen"> | null {
  try {
    const d = JSON.parse(event.data) as Record<string, unknown>;
    const bboxStr = d.bbox;
    if (typeof bboxStr !== "string") return null;
    const parts = bboxStr.split(",").map((x) => parseFloat(x.trim()));
    if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) return null;
    const [x1, y1, x2, y2] = parts;
    const id = String(
      d.globalTrackId ?? d.track_id ?? `${event.type}-${event.timestamp}-${d.label ?? "obj"}`,
    );
    const label =
      (typeof d.plateText === "string" && d.plateText) ||
      (typeof d.label === "string" && d.label) ||
      (typeof d.faceIdentity === "string" && d.faceIdentity) ||
      event.type;
    const confidence = parseFloat(String(d.confidence ?? "0")) || 0;
    return { id, x1, y1, x2, y2, label, confidence, kind: event.type };
  } catch {
    return null;
  }
}
