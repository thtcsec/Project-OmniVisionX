import type { Camera, DashboardStats, Detection, PlateResult, SimulatorCamera, SimulatorVideo } from "@/types/omni";

/** Same-origin in prod + Vite proxy in dev; override with OMNI_API_BASE_URL if needed */
const API_BASE = (import.meta.env.OMNI_API_BASE_URL as string | undefined) ?? "";

const SIM_BASE = import.meta.env.OMNI_SIMULATOR_BASE_URL ?? "http://localhost:8554";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const hasBody = options?.body !== undefined && options?.body !== null;
  const defaultHeaders: HeadersInit = hasBody ? { "Content-Type": "application/json" } : {};
  const res = await fetch(url, {
    headers: {
      ...defaultHeaders,
      ...(options?.headers ?? {}),
    },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      if (j?.error) detail = String(j.error);
      else if (j?.title) detail = String(j.title);
    } catch {
      /* ignore */
    }
    throw new Error(`API ${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

function simulatorVideosFromResponse(payload: unknown): SimulatorVideo[] {
  const raw = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object" && Array.isArray((payload as { videos?: unknown[] }).videos)
      ? (payload as { videos: unknown[] }).videos
      : [];

  return raw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
    .map((item) => {
      const filename = typeof item.filename === "string" ? item.filename : "";
      const baseName = filename ? filename.replace(/\.[^.]+$/, "") : (typeof item.name === "string" ? item.name : "video");
      const path = typeof item.path === "string" ? item.path : "";
      return {
        id: typeof item.id === "string" ? item.id : filename || baseName,
        name: typeof item.name === "string" ? item.name : baseName,
        filename: filename || (path ? path.split("/").pop() ?? baseName : baseName),
        path: path || undefined,
        duration: typeof item.duration === "number" ? item.duration : undefined,
      };
    });
}

function simulatorCamerasFromResponse(payload: unknown): SimulatorCamera[] {
  const raw = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object" && Array.isArray((payload as { cameras?: unknown[] }).cameras)
      ? (payload as { cameras: unknown[] }).cameras
      : [];

  return raw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
    .map((item) => {
      const id = typeof item.id === "string"
        ? item.id
        : typeof item.camera_id === "string"
          ? item.camera_id
          : "unknown-camera";
      const rtspUrl = typeof item.rtspUrl === "string"
        ? item.rtspUrl
        : typeof item.rtsp_url === "string"
          ? item.rtsp_url
          : "";
      const status = item.status === "running" ? "running" : "stopped";
      return { id, rtspUrl, status };
    });
}

/** Map .NET API (camelCase) → UI Camera */
export function normalizeCamera(raw: Record<string, unknown>): Camera {
  const id = String(raw.id ?? "");
  const name = String(raw.name ?? "");
  const statusRaw = typeof raw.status === "string" ? raw.status.toLowerCase() : "";
  const status: "online" | "offline" = statusRaw === "online" ? "online" : "offline";
  const features = {
    objectDetection: Boolean(raw.enableObjectDetection ?? (raw.features as Record<string, unknown>)?.objectDetection),
    plateRecognition: Boolean(raw.enablePlateOcr ?? (raw.features as Record<string, unknown>)?.plateRecognition),
    faceDetection: Boolean(raw.enableFaceRecognition ?? (raw.features as Record<string, unknown>)?.faceDetection),
  };
  return {
    id,
    name,
    status,
    streamUrl: typeof raw.streamUrl === "string" ? raw.streamUrl : undefined,
    hlsUrl: typeof raw.hlsUrl === "string" ? raw.hlsUrl : undefined,
    webrtcUrl: typeof raw.webrtcUrl === "string" ? raw.webrtcUrl : undefined,
    features,
    createdAt: typeof raw.createdAt === "string" ? raw.createdAt : undefined,
    updatedAt: typeof raw.updatedAt === "string" ? raw.updatedAt : undefined,
  };
}

function cameraToApiBody(values: { name: string; streamUrl: string }) {
  const hasStream = values.streamUrl.trim().length > 0;
  return {
    name: values.name,
    streamUrl: values.streamUrl,
    // RTSP / stream presence drives availability — no manual online switch in UI
    status: hasStream ? "online" : "offline",
    enableObjectDetection: true,
    enablePlateOcr: true,
    enableFaceRecognition: false,
  };
}

// Cameras
export const fetchCameras = async () => {
  const list = await request<Record<string, unknown>[]>(`${API_BASE}/api/Cameras`);
  return Array.isArray(list) ? list.map(normalizeCamera) : [];
};

export const fetchCamera = async (id: string) => {
  const raw = await request<Record<string, unknown>>(`${API_BASE}/api/Cameras/${encodeURIComponent(id)}`);
  return normalizeCamera(raw);
};

export const createCamera = (values: Parameters<typeof cameraToApiBody>[0]) =>
  request<Record<string, unknown>>(`${API_BASE}/api/Cameras`, {
    method: "POST",
    body: JSON.stringify(cameraToApiBody(values)),
  }).then(normalizeCamera);

export const updateCamera = (id: string, values: Parameters<typeof cameraToApiBody>[0]) =>
  request<void>(`${API_BASE}/api/Cameras/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(cameraToApiBody(values)),
  });

export const deleteCamera = (id: string) =>
  request<void>(`${API_BASE}/api/Cameras/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });

/** Proxied from omni-object via omni-api — works when Redis→SignalR is broken. */
export type LatestDetectionsSnapshot = {
  items: Record<
    string,
    {
      timestamp?: number;
      frame_width?: number;
      frame_height?: number;
      detections?: Array<{
        track_id: number;
        class_name?: string;
        class?: string;
        confidence?: number;
        bbox: [number, number, number, number];
        plate_text?: string;
      }>;
    }
  >;
  error?: string;
  detail?: string;
};

export const fetchLatestDetectionsSnapshot = (cameraIds: string) =>
  request<LatestDetectionsSnapshot>(
    `${API_BASE}/api/live/detections/latest?cameraIds=${encodeURIComponent(cameraIds)}`,
  );

export type LiveIngestStatus = {
  reachable: boolean;
  healthOk?: boolean;
  ingestCameraIds?: string[];
  detail?: string;
};

/** omni-object capture pool — explains missing bbox when empty or camera id not listed */
export const fetchLiveIngestStatus = () =>
  request<LiveIngestStatus>(`${API_BASE}/api/live/ingest-status`);

// Detections
export const fetchDetections = (params?: { cameraId?: string; from?: string; to?: string }) => {
  const q = new URLSearchParams();
  if (params?.cameraId) q.set("cameraId", params.cameraId);
  if (params?.from) q.set("from", params.from);
  if (params?.to) q.set("to", params.to);
  return request<Detection[]>(`${API_BASE}/api/Detections?${q}`);
};

// Plates
export const fetchPlates = (search?: string) => {
  const q = new URLSearchParams();
  if (search) q.set("plateText", search);
  const qs = q.toString();
  return request<PlateResult[]>(`${API_BASE}/api/Plates${qs ? `?${qs}` : ""}`);
};

// Dashboard stats
export const fetchDashboardStats = () => request<DashboardStats>(`${API_BASE}/api/Stats/dashboard`);

// Simulator
export const fetchSimulatorVideos = async () => simulatorVideosFromResponse(await request<unknown>(`${SIM_BASE}/simulator/videos`));
export const fetchSimulatorCameras = async () => simulatorCamerasFromResponse(await request<unknown>(`${SIM_BASE}/simulator/cameras`));
export const rescanSimulatorVideos = async (opts?: { autoStart?: boolean }) =>
  simulatorVideosFromResponse(
    await request<unknown>(`${SIM_BASE}/simulator/videos/rescan?auto_start=${opts?.autoStart ? "true" : "false"}`, { method: "POST" }),
  );
export const startSimulatorCamera = (cameraId: string, videoPath: string) =>
  request<Record<string, unknown>>(`${SIM_BASE}/simulator/cameras/${encodeURIComponent(cameraId)}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_path: videoPath,
      loop: true,
      fps: 15,
      transcode_h264: false,
    }),
  });
export const stopSimulatorCamera = (id: string) =>
  request<SimulatorCamera>(`${SIM_BASE}/simulator/cameras/${id}/stop`, { method: "POST" });

export type IntegrationEnvVar = {
  group: string;
  label: string;
  key: string;
  isSecret: boolean;
  isSet: boolean;
  value?: string | null;
};

export const fetchIntegrationEnvVars = () =>
  request<IntegrationEnvVar[]>(`${API_BASE}/api/settings/integrations/env`);

export const updateIntegrationEnvVars = (updates: Array<{ key: string; value: string }>) =>
  request<void>(`${API_BASE}/api/settings/integrations/env`, {
    method: "PUT",
    body: JSON.stringify({ updates }),
  });

export type IntegrationsStatus = {
  agora: { configured: boolean };
  elevenlabs: { configured: boolean };
  valsea: { configured: boolean };
  openai: { configured: boolean };
  exa: { configured: boolean };
  qwen: { configured: boolean };
  dify: { configured: boolean };
};

function readConfigured(raw: unknown, key: string): boolean {
  if (!raw || typeof raw !== "object") return false;
  const block = (raw as Record<string, unknown>)[key];
  if (!block || typeof block !== "object") return false;
  return Boolean((block as { configured?: unknown }).configured);
}

/** Ensures every integration key exists — avoids crashes if API or cached bundles omit fields. */
export function normalizeIntegrationsStatus(raw: unknown): IntegrationsStatus {
  const o = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  return {
    agora: { configured: readConfigured(o, "agora") },
    elevenlabs: { configured: readConfigured(o, "elevenlabs") },
    valsea: { configured: readConfigured(o, "valsea") },
    openai: { configured: readConfigured(o, "openai") },
    exa: { configured: readConfigured(o, "exa") },
    qwen: { configured: readConfigured(o, "qwen") },
    dify: { configured: readConfigured(o, "dify") },
  };
}

export async function fetchIntegrationsStatus(): Promise<IntegrationsStatus> {
  const raw = await request<unknown>(`${API_BASE}/api/integrations/status`);
  return normalizeIntegrationsStatus(raw);
}

export const chatWithCamera = (payload: {
  message: string;
  cameraId?: string;
  useExaGrounding?: boolean;
  /** When false, backend skips omni-object JPEG (text-only). Default: vision when model supports it. */
  includeFrameImage?: boolean;
}) =>
  request<{ provider: string; model: string; reply: string; exaUsed?: boolean; visionUsed?: boolean }>(
    `${API_BASE}/api/integrations/chat`,
    {
      method: "POST",
      body: JSON.stringify({
        message: payload.message,
        cameraId: payload.cameraId,
        useExaGrounding: payload.useExaGrounding,
        includeFrameImage: payload.includeFrameImage,
      }),
    },
  );

export const speakText = (payload: { text: string; voiceId?: string; modelId?: string }) =>
  fetch(`${API_BASE}/api/integrations/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(async (r) => {
    if (!r.ok) throw new Error(await r.text());
    return r.blob();
  });

export type AgoraTokenResult = {
  token: string;
  appId: string;
  channel: string;
  uid: number;
  expireSeconds: number;
};

export const fetchAgoraToken = (channelName: string, uid = 0, role: "subscriber" | "publisher" = "subscriber") =>
  request<AgoraTokenResult>(`${API_BASE}/api/agora/token`, {
    method: "POST",
    body: JSON.stringify({ channelName, uid, role, expireSeconds: 3600 }),
  });

export const fetchAgoraStatus = () =>
  request<{ configured: boolean; appId: string | null }>(`${API_BASE}/api/agora/status`);
