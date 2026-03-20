# OmniVision UI

Vite + React + TypeScript. Public env vars use the **`OMNI_`** prefix (`envPrefix` in `vite.config.ts`).

## Local dev

1. Start `omni-api` (e.g. Docker or `dotnet run` on port **8080**).
2. Copy `.env.example` → `.env` and adjust if needed.
3. `npm install` then `npm run dev` → [http://localhost:5173](http://localhost:5173) (proxies `/api`, `/hubs`, `/health` to `OMNI_DEV_API_PROXY_TARGET`).

## Docker

`Dockerfile` runs `npm run build` then serves `dist/` with **nginx**, proxying `/api`, `/hubs`, and `/health` to **`omni-api:8080`** (same Compose network as `docker-compose.yml`).

## Docs

- [docs/FRONTEND_PROMPT.md](../docs/FRONTEND_PROMPT.md) — product constraints and hub contract.
- [docs/ADAPT_FROM_LEGACY.md](../docs/ADAPT_FROM_LEGACY.md) — what to reuse from repo-root `frontend/` / `web-cms/`.
