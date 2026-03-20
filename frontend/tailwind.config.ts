import type { Config } from "tailwindcss";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["system-ui", "Segoe UI", "sans-serif"],
      },
      colors: {
        omni: {
          bg: "#0c0f14",
          surface: "#141922",
          border: "#252b38",
          accent: "#3b82f6",
          muted: "#8b93a7",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
