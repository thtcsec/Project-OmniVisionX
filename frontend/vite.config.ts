import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  envPrefix: ["OMNI_"],
  server: {
    host: "::",
    port: 5173,
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": { target: "http://127.0.0.1:8080", changeOrigin: true },
      "/hubs": { target: "http://127.0.0.1:8080", ws: true, changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8080", changeOrigin: true },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
