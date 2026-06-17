import { useEffect, useState } from "react";
import { api, usePoll, fmt } from "./api.js";
import { go } from "./ui.jsx";

const COLORS = ["#4493f8", "#3fb950", "#d29922", "#db61a2", "#a371f7", "#f85149", "#56d4dd"];

function MultiChart({ seriesMap, metric }) {
  const W = 760, H = 320, pad = 46;
  const ids = Object.keys(seriesMap);
  const all = ids.flatMap((id) => seriesMap[id]).filter((p) => p.y !== null && p.y !== undefined);
  if (!all.length) return <div className="muted small">select runs with data to compare</div>;
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  const xmax = Math.max(...xs, 1);
  let ymin = metric === "score" ? 0 : Math.min(...ys);
  let ymax = metric === "score" ? 1 : Math.max(...ys);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const sx = (x) => pad + (W - 2 * pad) * x / (xmax || 1);
  const sy = (y) => H - pad - (H - 2 * pad) * (y - ymin) / (ymax - ymin || 1);
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
        const v = ymin + t * (ymax - ymin);
        return (
          <g key={i}>
            <line x1={pad} x2={W - pad} y1={sy(v)} y2={sy(v)} stroke="#2b3440" />
            <text x={pad - 6} y={sy(v) + 4} fill="#8b949e" fontSize="10" textAnchor="end">{v.toFixed(metric === "score" ? 2 : 0)}</text>
          </g>
        );
      })}
      <text x={W - pad} y={H - 8} fill="#8b949e" fontSize="10" textAnchor="end">simulate #</text>
      {ids.map((id, idx) => {
        const pts = seriesMap[id].filter((p) => p.y !== null && p.y !== undefined);
        const d = pts.map((p, i) => (i ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.y).toFixed(1)).join(" ");
        return <path key={id} d={d} fill="none" stroke={COLORS[idx % COLORS.length]} strokeWidth="2" />;
      })}
    </svg>
  );
}

export default function Compare() {
  const { data: runs } = usePoll(api.runs, 5000);
  const [sel, setSel] = useState([]);
  const [metric, setMetric] = useState("score");
  const [series, setSeries] = useState({});

  const toggle = (id) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  useEffect(() => {
    let alive = true;
    Promise.all(sel.map((id) =>
      api.run(id).then((d) => [id, (d.simulates || []).map((s) => ({ x: s.idx, y: s.summary?.[metric === "score" ? "score" : "est_elo"] ?? null }))]).catch(() => [id, []])
    )).then((pairs) => { if (alive) setSeries(Object.fromEntries(pairs)); });
    return () => { alive = false; };
  }, [sel.join(","), metric]); // eslint-disable-line

  return (
    <div className="wrap">
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Compare runs</h2>
        <div className="row">
          <button className={metric === "score" ? "primary" : ""} onClick={() => setMetric("score")}>score</button>
          <button className={metric === "elo" ? "primary" : ""} onClick={() => setMetric("elo")}>est. Elo</button>
        </div>
      </div>
      <div className="row" style={{ alignItems: "flex-start", gap: 20, flexWrap: "wrap" }}>
        <div style={{ minWidth: 240 }}>
          {(runs || []).map((r, idx) => {
            const on = sel.includes(r.run_id);
            const color = COLORS[sel.indexOf(r.run_id) % COLORS.length];
            return (
              <div key={r.run_id} className="card" style={{ marginBottom: 8, cursor: "pointer", borderColor: on ? color : undefined }} onClick={() => toggle(r.run_id)}>
                <div className="spread">
                  <strong className="small">{r.label || r.run_id}</strong>
                  {on && <span style={{ width: 12, height: 12, background: color, borderRadius: 3 }} />}
                </div>
                <div className="small muted mono">{r.metaparam} · {r.backend?.model || r.backend?.name} · {r.num_simulates} sims · score {fmt(r.latest_score)}</div>
              </div>
            );
          })}
        </div>
        <div className="card" style={{ flex: 1, minWidth: 400 }}>
          <MultiChart seriesMap={series} metric={metric} />
          <div className="row" style={{ flexWrap: "wrap", gap: 12, marginTop: 8 }}>
            {sel.map((id, idx) => (
              <span key={id} className="small mono row" style={{ gap: 6 }}>
                <span style={{ width: 10, height: 10, background: COLORS[idx % COLORS.length], borderRadius: 2, display: "inline-block" }} />
                {(runs || []).find((r) => r.run_id === id)?.label || id}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
