

## OmniVision — Traffic & Vision Analytics SPA

### Design
Light theme, sidebar navigation, generous whitespace, card-based layouts. Clean and modern "minimal ops" style using shadcn/ui components.

### Layout
- **Sidebar** (collapsible) with nav links: Dashboard, Cameras, Simulator, Plate Search
- **Header** with sidebar trigger and app title "OmniVision"

### Pages

**1. Dashboard (`/`)**
- Stats cards: cameras online, total detections today, plates detected, active alerts
- Recent events list (latest ~10 from SignalR or API)
- Connection status indicator (SignalR connected/disconnected with toast on drop)

**2. Camera List (`/cameras`)**
- Card grid showing each camera: name, status badge (online/offline), feature flags (object/plate/face as small badges)
- Click card → camera detail

**3. Camera Detail (`/cameras/:id`)**
- Live video player: WebRTC first, HLS fallback (using hls.js)
- Real-time event panel fed by SignalR (`JoinCameraGroup`)
- History tab with date/time range filter and events table

**4. Simulator (`/simulator`)** *(dev tool)*
- Available videos list from simulator API
- Camera streams table with start/stop controls
- Shows RTSP URL for each stream

**5. Plate Search (`/plates`)**
- Search input for plate number
- Results table with timestamp, camera, plate text, confidence
- CSV export button

### Data & Connectivity
- API service layer using `OMNI_*` env vars from `import.meta.env`
- SignalR connection manager: connect on app load, `SubscribeAll()`, parse `OmniEvent` by type, reconnect with exponential backoff, toast on disconnect
- React Query for REST calls with loading/error states
- Ship `.env.example` documenting all `OMNI_*` variables

### Video Playback
- Custom hook `useCameraStream` that attempts WebRTC (WHEP) first, falls back to HLS via hls.js
- Graceful error states if neither protocol connects

### Vite Config
- `envPrefix: ['OMNI_']` — no `VITE_` prefix used

### Key Files
- `src/services/api.ts` — REST client
- `src/services/signalr.ts` — SignalR connection + event hooks
- `src/hooks/useCameraStream.ts` — WebRTC/HLS player logic
- `src/components/AppSidebar.tsx` — navigation sidebar
- `src/components/EventFeed.tsx` — real-time event list component
- `src/components/VideoPlayer.tsx` — video component with protocol fallback

