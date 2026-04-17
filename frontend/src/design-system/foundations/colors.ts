/**
 * Amphoreus design system color tokens.
 *
 * Primary palette uses stone (warm neutral) for a sophisticated, professional feel.
 * Accent colors are used sparingly for status and emphasis.
 */

export const colors = {
  primary: {
    900: "#1c1917",
    800: "#292524",
    700: "#44403c",
    600: "#57534e",
    500: "#78716c",
    400: "#a8a29e",
    300: "#d6d3d1",
    200: "#e7e5e4",
    100: "#f5f5f4",
    50: "#fafaf9",
  },
  accent: {
    blue: "#3b82f6",
    green: "#22c55e",
    amber: "#f59e0b",
    red: "#ef4444",
    cyan: "#06b6d4",
  },
  semantic: {
    success: "#22c55e",
    warning: "#f59e0b",
    error: "#ef4444",
    info: "#3b82f6",
  },
} as const;
