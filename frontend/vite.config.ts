import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

/** Public env vars use `OMNI_` prefix (see `.env.example`). */
export default defineConfig(({ mode }) => {
  const root = __dirname;
  const env = loadEnv(mode, root, "OMNI_");

  const apiTarget =
    env.OMNI_DEV_API_PROXY_TARGET || "http://localhost:8080";

  return {
    root,
    envDir: root,
    envPrefix: ["OMNI_"],
    server: {
      host: "::",
      port: 5173,
      proxy: {
        "/api": { target: apiTarget, changeOrigin: true },
        "/hubs": { target: apiTarget, changeOrigin: true, ws: true },
        "/health": { target: apiTarget, changeOrigin: true },
      },
    },
    plugins: [react()],
    resolve: {
      alias: { "@": path.resolve(root, "./src") },
    },
    build: {
      chunkSizeWarningLimit: 900,
    },
  };
});
