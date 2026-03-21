import type { OmniEvent } from "@/types/omni";

const EVENT_TYPES = new Set<OmniEvent["type"]>(["detection", "vehicle", "human", "plate"]);

/**
 * Hub / JSON serializers sometimes emit PascalCase or alternate keys; normalize before overlay logic.
 */
export function normalizeOmniEvent(raw: unknown): OmniEvent {
  const e = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const typeRaw = String(e.type ?? e.Type ?? "detection").toLowerCase();
  const type: OmniEvent["type"] = EVENT_TYPES.has(typeRaw as OmniEvent["type"])
    ? (typeRaw as OmniEvent["type"])
    : "detection";

  const cameraId = String(e.cameraId ?? e.CameraId ?? e.camera_id ?? "").trim();

  const rawData = e.data ?? e.Data;
  let data: OmniEvent["data"];
  if (typeof rawData === "string") {
    data = rawData;
  } else if (rawData && typeof rawData === "object") {
    data = rawData as Record<string, unknown>;
  } else {
    data = "{}";
  }

  const timestamp = String(e.timestamp ?? e.Timestamp ?? new Date().toISOString());

  return { type, cameraId, data, timestamp };
}
