import type { Camera, DashboardStats, Detection, PlateResult, SimulatorCamera, SimulatorVideo } from "@/types/omni";

const API_BASE = import.meta.env.OMNI_API_BASE_URL ?? "http://localhost:8080";
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
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

// Cameras
export const fetchCameras = () => request<Camera[]>(`${API_BASE}/api/Cameras`);
export const fetchCamera = (id: string) => request<Camera>(`${API_BASE}/api/Cameras/${id}`);

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
export const fetchSimulatorVideos = () => request<SimulatorVideo[]>(`${SIM_BASE}/simulator/videos`);
export const fetchSimulatorCameras = () => request<SimulatorCamera[]>(`${SIM_BASE}/simulator/cameras`);
export const startSimulatorCamera = (id: string) =>
  request<SimulatorCamera>(`${SIM_BASE}/simulator/cameras/${id}/start`, { method: "POST" });
export const stopSimulatorCamera = (id: string) =>
  request<SimulatorCamera>(`${SIM_BASE}/simulator/cameras/${id}/stop`, { method: "POST" });
