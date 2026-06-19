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

// The opponent ladder offered for optimization experiments (rated rungs only).
export const OPT_OPPONENTS = [
  "maia-1100", "maia-1500", "maia-1900",
  "stockfish-1320", "stockfish-1800", "stockfish-2400",
];

export function KindBadge({ kind }) {
  const opt = kind === "optimization";
  return (
    <span className="tag" style={opt ? { color: "var(--warn)", borderColor: "var(--warn)" } : { color: "var(--accent)", borderColor: "var(--accent)" }}>
      {opt ? "optimization" : "single"}
    </span>
  );
}
