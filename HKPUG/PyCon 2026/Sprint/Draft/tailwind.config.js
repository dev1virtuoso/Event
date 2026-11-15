/** Tailwind build config for the Clef-Flash web UI.
 *
 *  The palette mirrors the pygame constants in Draft/image_clef.py so the
 *  browser version looks like the same tool:
 *    BG_COLOR      (18, 18, 24)    -> #121218
 *    PANEL_COLOR   (28, 29, 38)    -> #1C1D26
 *    ACCENT_COLOR  (80, 200, 120)  -> #50C878
 *    TEXT_COLOR    (235, 235, 240) -> #EBEBF0
 *    MUTED_COLOR   (140, 145, 160) -> #8C91A0
 *    WARN_COLOR    (230, 160, 60)  -> #E6A03C
 *    BORDER_COLOR  (60, 65, 85)    -> #3C4155
 *    HOVER_COLOR   (110, 220, 150) -> #6EDC96
 */
module.exports = {
  content: ["./image_clef_web.py"],
  theme: {
    extend: {
      colors: {
        bg: "#121218",
        panel: "#1C1D26",
        panelalt: "#16171F",
        accent: "#50C878",
        hover: "#6EDC96",
        text: "#EBEBF0",
        muted: "#8C91A0",
        warn: "#E6A03C",
        edge: "#3C4155",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["Menlo", "Consolas", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
