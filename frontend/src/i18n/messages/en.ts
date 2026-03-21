export const en = {
  nav: {
    group: "Navigation",
    dashboard: "Dashboard",
    cameras: "Cameras",
    plates: "Plate search",
    simulator: "Simulator",
    live: "Live preview",
    about: "About",
    settings: "Settings",
    subtitle: "Traffic analytics",
  },
  settings: {
    title: "Settings",
    description: "Theme and display language.",
    appearance: "Theme",
    light: "Light",
    dark: "Dark",
    system: "System",
    appliedPrefix: "Active:",
    darkLabel: "Dark",
    lightLabel: "Light",
    followSystem: "(follows system)",
    language: "Language",
    english: "English",
    vietnamese: "Vietnamese",
  },
  live: {
    title: "Live preview",
    subtitle: "Stream with optional real-time detection overlays (SignalR).",
    selectCamera: "Camera",
    pickCamera: "Select a camera",
    showTracks: "Show tracking overlays",
    tracksHint:
      "Boxes come from Redis → API (detections / vehicles / humans). Scaled with object-contain; resolution assumes detector frame matches stream.",
    noStream: "No HLS/WebRTC URL for this camera. Configure stream in Cameras or MediaMTX.",
    waitingEvents: "Waiting for detection events…",
    activeBoxes: "Active boxes",
    joinSignalR: "Connect to SignalR to receive overlays.",
    streamUnavailable: "Stream unavailable",
    noStreamConfigured: "No stream configured",
    troubleshootTitle: "Seeing “Stream unavailable” or ERR_CONNECTION_REFUSED on :8888?",
    troubleshootP1:
      "This page builds HLS/WebRTC URLs on port 8888 (MediaMTX). If no server listens there, the browser cannot load the playlist or WHEP endpoint.",
    troubleshootP2:
      "After MediaMTX is up, the path name must match your camera ID (e.g. /1/index.m3u8) and the path’s source must be your RTSP URL (configure via MediaMTX API or omni-object relay).",
  },
  about: {
    tagline:
      "AI vision stack for traffic and scene understanding — event-driven on Redis Streams, .NET 9 API, SignalR realtime.",
    project: "Project",
    projectDesc:
      "Built for hackathon demo — camera → detections → plates / people pipeline.",
    developedBy: "Developed by",
    github: "Source on GitHub",
  },
  cameras: {
    hackathonNote: "CRUD cameras (hackathon — no auth). Use JWT / API keys in production.",
  },
} as const;

export type MessageTree = typeof en;
