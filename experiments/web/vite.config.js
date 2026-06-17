import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build into experiments/web/dist (served by the FastAPI app in production).
// In dev (`npm run dev`), proxy /api to the FastAPI control plane on :8011.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8011" },
  },
});
