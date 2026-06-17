// A small dependency-free SVG line chart: y over x, clickable points.
export default function Chart({ points, color = "#4493f8", ylabel = "", domain = null, onPick }) {
  const W = 560, H = 240, pad = 44;
  const valid = points.filter((p) => p.y !== null && p.y !== undefined);
  if (!valid.length) return <div className="muted small">no data yet</div>;
  const xs = points.map((p) => p.x);
  const ys = valid.map((p) => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs, xmin + 1);
  let ymin = domain ? domain[0] : Math.min(...ys);
  let ymax = domain ? domain[1] : Math.max(...ys);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const sx = (x) => pad + (W - 2 * pad) * (x - xmin) / (xmax - xmin || 1);
  const sy = (y) => H - pad - (H - 2 * pad) * (y - ymin) / (ymax - ymin || 1);
  const path = valid.map((p, i) => (i ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.y).toFixed(1)).join(" ");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => ymin + t * (ymax - ymin));

  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad} x2={W - pad} y1={sy(t)} y2={sy(t)} stroke="#2b3440" strokeWidth="1" />
          <text x={pad - 6} y={sy(t) + 4} fill="#8b949e" fontSize="10" textAnchor="end">
            {t.toFixed(t > 5 ? 0 : 2)}
          </text>
        </g>
      ))}
      <text x={10} y={14} fill="#8b949e" fontSize="11">{ylabel}</text>
      <text x={W - pad} y={H - 8} fill="#8b949e" fontSize="10" textAnchor="end">simulate #</text>
      <path d={path} fill="none" stroke={color} strokeWidth="2" />
      {valid.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="4" fill={color}
          style={{ cursor: onPick ? "pointer" : "default" }}
          onClick={() => onPick && onPick(p.x)}>
          <title>{`#${p.x}: ${p.y}`}</title>
        </circle>
      ))}
    </svg>
  );
}
