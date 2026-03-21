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
      return {
        id: typeof item.id === "string" ? item.id : filename || baseName,
        name: typeof item.name === "string" ? item.name : baseName,
        filename: filename || (typeof item.path === "string" ? item.path.split("/").pop() ?? baseName : baseName),
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

function cameraToApiBody(values: {
  name: string;
  streamUrl: string;
  status: "online" | "offline";
  enableObjectDetection: boolean;
  enablePlateOcr: boolean;
  enableFaceRecognition: boolean;
}) {
  return {
    name: values.name,
    streamUrl: values.streamUrl,
    status: values.status,
    enableObjectDetection: values.enableObjectDetection,
    enablePlateOcr: values.enablePlateOcr,
    enableFaceRecognition: values.enableFaceRecognition,
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
  const q = search ? `?search=${encodeURIComponent(search)}` : "";
  return request<PlateResult[]>(`${API_BASE}/api/Plates${q}`);
};

// Dashboard stats
export const fetchDashboardStats = () => request<DashboardStats>(`${API_BASE}/api/Stats/dashboard`);

// Simulator
export const fetchSimulatorVideos = async () => simulatorVideosFromResponse(await request<unknown>(`${SIM_BASE}/simulator/videos`));
export const fetchSimulatorCameras = async () => simulatorCamerasFromResponse(await request<unknown>(`${SIM_BASE}/simulator/cameras`));
export const startSimulatorCamera = (id: string) =>
  request<SimulatorCamera>(`${SIM_BASE}/simulator/cameras/${id}/start`, { method: "POST" });
export const stopSimulatorCamera = (id: string) =>
  request<SimulatorCamera>(`${SIM_BASE}/simulator/cameras/${id}/stop`, { method: "POST" });
