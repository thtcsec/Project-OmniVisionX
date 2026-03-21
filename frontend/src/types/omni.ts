// Camera
export interface Camera {
  id: string;
  name: string;
  status: "online" | "offline";
  streamUrl?: string;
  hlsUrl?: string;
  webrtcUrl?: string;
  features: {
    objectDetection?: boolean;
    plateRecognition?: boolean;
    faceDetection?: boolean;
  };
  createdAt?: string;
  updatedAt?: string;
}

// Detection event from SignalR
export interface OmniEvent {
  type: "detection" | "vehicle" | "human" | "plate";
  cameraId: string;
  /** Hub usually sends a JSON string; some transports may deserialize to object. */
  data: string | Record<string, unknown>;
  timestamp: string;
}

export interface Detection {
  id: string;
  cameraId: string;
  type: string;
  confidence: number;
  bbox?: { x: number; y: number; w: number; h: number };
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface PlateResult {
  id: string;
  cameraId: string;
  cameraName?: string;
  plateText: string;
  confidence: number;
  timestamp: string;
  vehicleType?: string;
  color?: string | null;
  plateImageUrl?: string | null;
  frameImageUrl?: string | null;
}

// Simulator types
export interface SimulatorVideo {
  id: string;
  name: string;
  filename: string;
  /** Absolute path inside the simulator container (for POST /start). */
  path?: string;
  duration?: number;
}

export interface SimulatorCamera {
  id: string;
  videoId?: string;
  rtspUrl: string;
  status: "running" | "stopped";
}

// Dashboard stats
export interface DashboardStats {
  camerasOnline: number;
  camerasTotal: number;
  detectionsToday: number;
  platesDetected: number;
  activeAlerts: number;
}
