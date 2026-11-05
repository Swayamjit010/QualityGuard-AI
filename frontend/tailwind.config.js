/** Tailwind is used with a `tw-` prefix and without preflight so it can coexist with the existing
 *  hand-written design system in src/styles.css (which owns the theme variables, light/dark). */
export default {
  prefix: "tw-",
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  corePlugins: { preflight: false },
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        qg: {
          ink: "var(--text)",
          muted: "var(--muted)",
          card: "var(--panel)", panel2: "var(--panel-2)",
          line: "var(--border)",
          brand: "var(--primary)",
        },
        verdict: { normal: "var(--pass)", defective: "var(--fail)", manual: "var(--review)" },
      },
      fontFamily: { mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
    },
  },
  plugins: [],
};
