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
    description: "Appearance, language, and optional API integration keys.",
    tabGeneral: "General",
    tabIntegrations: "Integrations",
    integrationsTitle: "Sponsor integrations",
    integrationsHint: "API keys and endpoints (server-side .env).",
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
    save: "Save",
    saved: "Saved",
    saveFailed: "Save failed",
    saving: "Saving…",
    loading: "Loading…",
    envDisabled: "Env editor is disabled on the API.",
    envFooter:
      "Changes apply to the API process. Other containers may still need a restart to pick up new values.",
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
    troubleshootTitle: "Seeing “Stream unavailable” or ERR_CONNECTION_REFUSED on :8888 / :8889?",
    troubleshootP1:
      "HLS uses port 8888; WebRTC (WHEP) uses 8889 (MediaMTX defaults). If nothing listens on those ports, the browser cannot load the playlist or WHEP endpoint.",
    troubleshootP2:
      "Path name must match your camera ID. Saving an online camera with an RTSP URL registers it in MediaMTX automatically (API + Docker). Or use omni-object relay when detection is enabled.",
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
