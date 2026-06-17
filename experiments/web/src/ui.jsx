// shared UI helpers (kept here so views never import App.jsx -> no circular imports)
export const go = (h) => { window.location.hash = h; };

export function StatusTag({ status }) {
  return <span className={`tag ${status || ""}`}>{status || "—"}</span>;
}

export const BOTS = [
  "random", "greedy",
  "stockfish-fast", "stockfish-1320", "stockfish-1800", "stockfish-2400", "stockfish-full",
  "maia-1100", "maia-1500", "maia-1900",
];
