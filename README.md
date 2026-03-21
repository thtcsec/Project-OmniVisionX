# Project OmniVisionX — AI Vision Platform

<p align="center">
  <img src=".image/project-omnivisionx-white.png" alt="OmniVisionX" width="320" />
</p>

**Project OmniVisionX** is a modular AI vision stack aimed at traffic and scene understanding. It uses an **event-driven** architecture built on **Redis Streams**, with a **.NET 9** API and **SignalR** for real-time updates.

## Architecture

```
Project OmniVisionX/
├── backend/
│   ├── apis/                 # .NET 9 Web API + SignalR
│   │   ├── Controllers/      # REST endpoints
│   │   ├── Models/
│   │   ├── Data/             # EF Core DbContext
│   │   ├── Hubs/             # SignalR OmniHub
│   │   └── Services/         # Background workers (e.g. Redis consumer)
│   │
│   └── services/
│       ├── omni-simulator/   # Mock camera: loop video → RTSP
│       ├── omni-object/      # Object detection (YOLO + tracking)
│       ├── omni-vehicle/     # License plate + vehicle pipeline
│       ├── omni-human/       # Face + person (stream consumer)
│       ├── omni-fusion/      # Spatial–temporal fusion
│       └── …
│
├── shared/                   # Shared Python utilities
├── redis/                    # Redis configuration
├── media-server/             # MediaMTX (RTSP → HLS/WebRTC) when used
├── models/yolo/              # YOLO weights (mounted into detectors; not committed)
├── frontend/                 # Web UI (Vite + React)
└── docker-compose.yml        # Stack orchestration
```

## Services (typical ports)

| Service         | Port  | Description                          |
|-----------------|-------|--------------------------------------|
| omni-db         | 5432  | PostgreSQL + pgvector                |
| omni-bus        | 6379  | Redis (event bus)                    |
| omni-api        | 8080  | .NET API + SignalR                   |
| omni-simulator  | 8554  | Mock camera (RTSP + control API)     |
| omni-object     | 8555  | YOLO / detection service             |
| omni-vehicle    | 8001  | LPR + vehicle pipeline               |
| omni-human      | 8002  | Face / person recognition            |
| omni-ui         | 3000  | Frontend (when run via compose)      |
| omni-minio      | 9000 / 9001 | S3 API / web console (see `.env`) |

Exact ports and profiles may vary; see `docker-compose.yml`. Configure MinIO credentials and bucket names via `.env` (`MINIO_*`, `OMNI_S3_*`).

## Quick start

```powershell
# Core infrastructure
docker compose -f docker-compose.yml up -d omni-db omni-bus omni-api

# Mock camera (drop videos under ./data/videos as configured)
docker compose -f docker-compose.yml up -d omni-simulator

# GPU-backed AI workers (when defined in compose)
docker compose -f docker-compose.yml --profile gpu up -d
```

Copy `.env.example` to `.env` and adjust secrets and URLs before production use.

### Web UI (Vite)

```powershell
cd frontend
npm install
npm run dev
```

With `omni-api` on port 8080, the dev server proxies `/api`, `/hubs`, and `/health`. See [frontend/README.md](frontend/README.md) and [docs/ADAPT_FROM_LEGACY.md](docs/ADAPT_FROM_LEGACY.md) for porting ideas from the legacy repo-root `frontend/`.

## Object storage (MinIO)

Start MinIO with Compose (`omni-minio`). Set `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, and optional `MINIO_API_PORT` / `MINIO_CONSOLE_PORT` in `.env`. Application services should use `OMNI_S3_*` variables (endpoint, bucket, keys) to upload thumbnails, exports, or model artifacts—**never** commit real secrets.

## Redis streams

| Stream           | Purpose                    |
|------------------|----------------------------|
| `omni:detections`| Object detection events    |
| `omni:vehicles`  | Plate + vehicle metadata   |
| `omni:humans`    | Face / person events       |

Fusion (`omni-fusion`) consumes the streams above and posts linked results to the **REST API** (`omni-api`), not to a separate `omni:fusion` stream.

## Mock camera (`omni-simulator`)

Place video files under the mounted videos directory (e.g. `./data/videos/`). Typical RTSP URLs:

```text
video.mp4   → rtsp://localhost:8554/cam-video
highway.mp4 → rtsp://localhost:8554/cam-highway
```

Control API (FastAPI on the simulator port):

- `GET /simulator/videos` — list available files  
- `GET /simulator/cameras` — list active streams  
- `POST /simulator/cameras/{id}/start` — start a stream  
- `POST /simulator/cameras/{id}/stop` — stop a stream  

Browsers do not play raw RTSP; use **HLS** or **WebRTC** via a relay such as **MediaMTX** for live preview in the web UI.

## Cameras (CRUD)

The web UI under **Cameras** supports **create / read / update / delete** via `GET|POST|PUT|DELETE /api/Cameras`.  
Stream URL accepts **rtsp(s)://** or **http(s)://** (e.g. HLS later). After mutations, clients subscribed to SignalR receive **`CamerasChanged`** so lists refresh across tabs.

## Live preview (`/live`)

Spa route **Live preview** plays the camera HLS/WebRTC URL (same convention as camera detail + `OMNI_MEDIA_BASE_URL`) and optionally draws **bounding boxes** from **`OmniEvent`** (Redis → `OmniEventConsumerService` → SignalR). Toggle **Show tracking overlays** to enable/disable. UI supports **English / Vietnamese** and persists locale in `localStorage` (`omnivisionx-locale`).

**If you see “Stream unavailable” / `ERR_CONNECTION_REFUSED` on ports 8888 or 8889:** the UI defaults to HLS `http://localhost:8888/<cameraId>/index.m3u8` and WHEP `http://localhost:8889/<cameraId>/whep` (MediaMTX). Start the relay:

```powershell
docker compose up -d omni-mediamtx
```

Optional env: `OMNI_MEDIA_WEBRTC_BASE_URL` if WebRTC is not on the same host with port 8889.

The **API** registers MediaMTX paths automatically when you **save** a camera as **online** with an **RTSP** URL (`MediaMtx__ApiBaseUrl` in Docker). Path name = camera **ID** (same as the HLS URL). `omni-object` also registers paths when detection is enabled and relay is on.

## Security note (hackathon vs production)

**Hackathon build:** API and CRUD are **open** (no JWT / roles) so judges can try flows quickly.  
**Production:** add authentication (e.g. ASP.NET Identity + JWT or API keys), HTTPS, rate limiting, and never expose RTSP credentials in client-side logs.

## Roadmap / TODO

- [ ] Harden `omni-object` (YOLO pipeline, publishing to Redis)  
- [ ] Harden `omni-vehicle` (Vietnamese LPR path)  
- [ ] Harden `omni-human` (e.g. InsightFace integration)  
- [ ] Harden `omni-fusion`  
- [ ] Integrations: Agora, ElevenLabs, Qwen, Manus AI (see `.env.example`)  
- [ ] Real-time operations dashboard  

## Remote repository

Primary repo: [github.com/thtcsec/Project-OmniVisionX](https://github.com/thtcsec/Project-OmniVisionX) — update remote if needed:  
`git remote set-url origin https://github.com/thtcsec/Project-OmniVisionX.git`

## Docker — sau khi sửa code, chạy gì?

`docker restart` **không** lấy code mới: container vẫn chạy **image cũ** đã build trước đó.

| Tình huống | Lệnh gợi ý |
|------------|------------|
| Đổi **API** (.NET) | `docker compose build omni-api && docker compose up -d omni-api` |
| Đổi **UI** (React) | `docker compose build omni-ui && docker compose up -d omni-ui` |
| Đổi **cả hai** | `docker compose build omni-api omni-ui && docker compose up -d omni-api omni-ui` |
| Muốn build lại sạch (ít cache) | `docker compose build --no-cache omni-api omni-ui && docker compose up -d omni-api omni-ui` |

**Nhanh nhất khi dev UI:** chạy API trong Docker, còn frontend chạy local:

```bash
cd frontend && npm install && npm run dev
```

(Vite proxy `/api` và `/hubs` → `127.0.0.1:8080` — không cần rebuild `omni-ui` mỗi lần sửa React.)

## Hackathon

Submitted as part of **LotusHack × HackHarvard × GenAI Fund Vietnam**.
