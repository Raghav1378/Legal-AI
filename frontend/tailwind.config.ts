import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#1a1a18",
        surface: "#212121",
        raised: "#242422",
        hover: "#2b2b28",
        press: "#31312d",
        ink: "#e8e6e3",
        "ink-muted": "#a3a09a",
        "ink-faint": "#6f6b65",
        line: "rgba(255, 255, 255, 0.08)",
        "line-strong": "rgba(255, 255, 255, 0.12)",
        accent: "#c15f3c",
        "accent-hover": "#d97757",
        "accent-soft": "rgba(193, 95, 60, 0.14)",
        "accent-ink": "#e8b49a",
      },
    },
  },
  plugins: [],
} satisfies Config;
