import { useEffect, useMemo, useState } from "react";
import { api, usePoll, fmt } from "./api.js";
import { go, StatusTag } from "./ui.jsx";
import { Trajectory } from "./RunDetail.jsx";
import Board from "./Board.jsx";

// ---------- small helpers ----------
const AX = "#2b3440", MUTED = "#8b949e";
const C_ACCEPT = "#3fb950", C_REJECT = "#f85149", C_FRONT = "#d29922", C_LINE = "#4493f8";

function lerp(a, b, t) { return a + (b - a) * t; }
// blue (low) -> green (high) ramp for est_elo coloring
function eloColor(v, lo, hi) {
  if (v === null || v === undefined || hi <= lo) return MUTED;
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const r = Math.round(lerp(68, 63, t)), g = Math.round(lerp(147, 185, t)), b = Math.round(lerp(248, 80, t));
  return `rgb(${r},${g},${b})`;
}
function extent(vals, fallback = [0, 1]) {
  const v = vals.filter((x) => x !== null && x !== undefined && !Number.isNaN(x));
  if (!v.length) return fallback;
  let lo = Math.min(...v), hi = Math.max(...v);
  if (lo === hi) { lo -= 1; hi += 1; }
  return [lo, hi];
}
function ChartCard({ title, hint, children }) {
  return (
    <div className="card" style={{ minWidth: 320 }}>
      <div className="small muted">{title}</div>
      {hint && <div className="small muted" style={{ opacity: 0.7, marginBottom: 4 }}>{hint}</div>}
      {children}
    </div>
  );
}
function Empty({ msg = "no data yet" }) { return <div className="muted small" style={{ padding: "12px 0" }}>{msg}</div>; }

// generic axes frame + y ticks
function Frame({ W, H, pad, ymin, ymax, xlabel, ylabel, fmtY = (v) => v.toFixed(0) }) {
  const sy = (y) => H - pad - (H - 2 * pad) * (y - ymin) / (ymax - ymin || 1);
  return (
    <>
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
        const v = ymin + t * (ymax - ymin);
        return (
          <g key={i}>
            <line x1={pad} x2={W - pad} y1={sy(v)} y2={sy(v)} stroke={AX} />
            <text x={pad - 6} y={sy(v) + 4} fill={MUTED} fontSize="10" textAnchor="end">{fmtY(v)}</text>
          </g>
        );
      })}
      {ylabel && <text x={10} y={14} fill={MUTED} fontSize="11">{ylabel}</text>}
      {xlabel && <text x={W - pad} y={H - 6} fill={MUTED} fontSize="10" textAnchor="end">{xlabel}</text>}
    </>
  );
}

// ---------- 1. Elo vs step scatter + best-so-far ----------
function EloScatter({ pop, lo, hi, onPick }) {
  const W = 560, H = 260, pad = 46;
  const pts = pop.filter((p) => p.est_elo !== null && p.est_elo !== undefined);
  if (!pts.length) return <Empty />;
  const [ymin, ymax] = extent(pts.map((p) => p.est_elo));
  const xs = pts.map((p) => p.step ?? 0);
  const xmin = Math.min(...xs), xmax = Math.max(...xs, xmin + 1);
  const sx = (x) => pad + (W - 2 * pad) * (x - xmin) / (xmax - xmin || 1);
  const sy = (y) => H - pad - (H - 2 * pad) * (y - ymin) / (ymax - ymin || 1);
  const sorted = [...pts].sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
  let best = -Infinity;
  const bestLine = sorted.map((p) => { best = Math.max(best, p.est_elo); return { x: p.step ?? 0, y: best }; });
  const path = bestLine.map((p, i) => (i ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.y).toFixed(1)).join(" ");
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      <Frame W={W} H={H} pad={pad} ymin={ymin} ymax={ymax} xlabel="step" ylabel="est. Elo" />
      <path d={path} fill="none" stroke={C_LINE} strokeWidth="2" strokeDasharray="4 3" />
      {sorted.map((p, i) => {
        const cx = sx(p.step ?? 0), cy = sy(p.est_elo);
        const front = p.on_pareto_front, acc = p.accepted;
        return (
          <g key={p.candidate_id || i} style={{ cursor: "pointer" }} onClick={() => onPick && onPick(p.candidate_id)}>
            {front && <circle cx={cx} cy={cy} r="7" fill="none" stroke={C_FRONT} strokeWidth="2" />}
            <circle cx={cx} cy={cy} r="4.5"
              fill={acc ? eloColor(p.est_elo, lo, hi) : "none"}
              stroke={acc ? eloColor(p.est_elo, lo, hi) : C_REJECT} strokeWidth="1.5">
              <title>{`${p.candidate_id} · gen ${p.gen} · elo ${fmt(p.est_elo)} · ${acc ? "accepted" : "rejected"}${front ? " · frontier" : ""}`}</title>
            </circle>
          </g>
        );
      })}
    </svg>
  );
}

// ---------- 2. Per-generation convergence (best & mean elo) ----------
function byGen(pop, key) {
  const m = {};
  for (const p of pop) {
    const v = p[key];
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    (m[p.gen] = m[p.gen] || []).push(v);
  }
  return m;
}
function GenConvergence({ pop }) {
  const W = 560, H = 240, pad = 46;
  const groups = byGen(pop, "est_elo");
  const gens = Object.keys(groups).map(Number).sort((a, b) => a - b);
  if (!gens.length) return <Empty />;
  const bests = gens.map((g) => Math.max(...groups[g]));
  const means = gens.map((g) => groups[g].reduce((s, v) => s + v, 0) / groups[g].length);
  const [ymin, ymax] = extent([...bests, ...means]);
  const xmin = gens[0], xmax = gens[gens.length - 1];
  const sx = (g) => pad + (W - 2 * pad) * (g - xmin) / (xmax - xmin || 1);
  const sy = (y) => H - pad - (H - 2 * pad) * (y - ymin) / (ymax - ymin || 1);
  const line = (arr) => arr.map((y, i) => (i ? "L" : "M") + sx(gens[i]).toFixed(1) + " " + sy(y).toFixed(1)).join(" ");
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      <Frame W={W} H={H} pad={pad} ymin={ymin} ymax={ymax} xlabel="generation" ylabel="est. Elo" />
      <path d={line(bests)} fill="none" stroke={C_ACCEPT} strokeWidth="2" />
      <path d={line(means)} fill="none" stroke={C_LINE} strokeWidth="2" strokeDasharray="4 3" />
      {gens.map((g, i) => <circle key={"b" + g} cx={sx(g)} cy={sy(bests[i])} r="3.5" fill={C_ACCEPT} />)}
      {gens.map((g, i) => <circle key={"m" + g} cx={sx(g)} cy={sy(means[i])} r="3" fill={C_LINE} />)}
      <text x={pad} y={16} fill={C_ACCEPT} fontSize="10">— best</text>
      <text x={pad + 50} y={16} fill={C_LINE} fontSize="10">-- mean</text>
    </svg>
  );
}

// ---------- 3. Score spread / diversity per generation (min–max band + mean) ----------
function ScoreSpread({ pop }) {
  const W = 560, H = 240, pad = 46;
  const groups = byGen(pop, "scalar");
  const gens = Object.keys(groups).map(Number).sort((a, b) => a - b);
  if (!gens.length) return <Empty />;
  const stat = gens.map((g) => {
    const v = groups[g];
    const mean = v.reduce((s, x) => s + x, 0) / v.length;
    return { g, min: Math.min(...v), max: Math.max(...v), mean };
  });
  const [ymin, ymax] = extent(stat.flatMap((s) => [s.min, s.max]));
  const xmin = gens[0], xmax = gens[gens.length - 1];
  const sx = (g) => pad + (W - 2 * pad) * (g - xmin) / (xmax - xmin || 1);
  const sy = (y) => H - pad - (H - 2 * pad) * (y - ymin) / (ymax - ymin || 1);
  const band = stat.map((s) => `${sx(s.g).toFixed(1)} ${sy(s.max).toFixed(1)}`).join(" L ") +
    " L " + [...stat].reverse().map((s) => `${sx(s.g).toFixed(1)} ${sy(s.min).toFixed(1)}`).join(" L ");
  const meanLine = stat.map((s, i) => (i ? "L" : "M") + sx(s.g).toFixed(1) + " " + sy(s.mean).toFixed(1)).join(" ");
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      <Frame W={W} H={H} pad={pad} ymin={ymin} ymax={ymax} xlabel="generation" ylabel="scalar" fmtY={(v) => v.toFixed(2)} />
      <path d={"M " + band + " Z"} fill="rgba(68,147,248,0.18)" stroke="none" />
      <path d={meanLine} fill="none" stroke={C_LINE} strokeWidth="2" />
      {stat.map((s) => <circle key={s.g} cx={sx(s.g)} cy={sy(s.mean)} r="3" fill={C_LINE} />)}
      <text x={pad} y={16} fill={MUTED} fontSize="10">band = min–max · line = mean</text>
    </svg>
  );
}

// ---------- 4. Pareto frontier scatter (selectable axes) ----------
function axisOptions(pop) {
  const opp = new Set(), vec = new Set();
  for (const p of pop) {
    Object.keys(p.per_instance || {}).forEach((k) => opp.add(k));
    Object.keys(p.vector || {}).forEach((k) => vec.add(k));
  }
  const out = [];
  [...opp].forEach((k) => out.push({ key: "opp:" + k, label: "opp · " + k }));
  [...vec].forEach((k) => out.push({ key: "vec:" + k, label: "vec · " + k }));
  out.push({ key: "scalar", label: "scalar" }, { key: "est_elo", label: "est. Elo" });
  return out;
}
function axisVal(p, key) {
  if (key === "scalar") return p.scalar;
  if (key === "est_elo") return p.est_elo;
  if (key.startsWith("opp:")) return (p.per_instance || {})[key.slice(4)];
  if (key.startsWith("vec:")) return (p.vector || {})[key.slice(4)];
  return undefined;
}
function ParetoScatter({ pop, lo, hi, onPick }) {
  const opts = useMemo(() => axisOptions(pop), [pop]);
  const [xk, setXk] = useState(null), [yk, setYk] = useState(null);
  const xKey = xk || opts[0]?.key, yKey = yk || opts[1]?.key || opts[0]?.key;
  const W = 560, H = 260, pad = 50;
  if (!pop.length || !opts.length) return <Empty />;
  const pts = pop.map((p) => ({ p, x: axisVal(p, xKey), y: axisVal(p, yKey) }))
    .filter((d) => d.x !== null && d.x !== undefined && d.y !== null && d.y !== undefined);
  const [xmin, xmax] = extent(pts.map((d) => d.x));
  const [ymin, ymax] = extent(pts.map((d) => d.y));
  const sx = (x) => pad + (W - 2 * pad) * (x - xmin) / (xmax - xmin || 1);
  const sy = (y) => H - pad - (H - 2 * pad) * (y - ymin) / (ymax - ymin || 1);
  return (
    <div>
      <div className="row" style={{ gap: 8, marginBottom: 6 }}>
        <select value={xKey} onChange={(e) => setXk(e.target.value)} style={{ width: "auto" }}>
          {opts.map((o) => <option key={o.key} value={o.key}>x: {o.label}</option>)}
        </select>
        <select value={yKey} onChange={(e) => setYk(e.target.value)} style={{ width: "auto" }}>
          {opts.map((o) => <option key={o.key} value={o.key}>y: {o.label}</option>)}
        </select>
      </div>
      {pts.length === 0 ? <Empty /> : (
        <svg width={W} height={H} style={{ maxWidth: "100%" }}>
          <Frame W={W} H={H} pad={pad} ymin={ymin} ymax={ymax} fmtY={(v) => v.toFixed(2)} />
          {pts.map((d, i) => {
            const front = d.p.on_pareto_front;
            return (
              <g key={d.p.candidate_id || i} style={{ cursor: "pointer" }} onClick={() => onPick && onPick(d.p.candidate_id)}>
                {front && <circle cx={sx(d.x)} cy={sy(d.y)} r="7" fill="none" stroke={C_FRONT} strokeWidth="2" />}
                <circle cx={sx(d.x)} cy={sy(d.y)} r="4.5" fill={eloColor(d.p.est_elo, lo, hi)}>
                  <title>{`${d.p.candidate_id}${front ? " · frontier" : ""}`}</title>
                </circle>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

// ---------- 5. Per-opponent win-rate heatmap for best candidate ----------
function WinRateHeatmap({ best }) {
  const pi = best?.per_instance || {};
  const keys = Object.keys(pi);
  if (!keys.length) return <Empty msg="no per-opponent scores" />;
  return (
    <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
      {keys.map((k) => {
        const v = pi[k];
        const t = Math.max(0, Math.min(1, v ?? 0));
        const bg = `rgba(63,185,80,${0.15 + 0.75 * t})`;
        return (
          <div key={k} style={{ background: bg, border: "1px solid var(--border)", borderRadius: 8, padding: "10px 12px", minWidth: 110, textAlign: "center" }}>
            <div className="small mono">{k}</div>
            <div style={{ fontWeight: 700, fontSize: 18 }}>{fmt(v)}</div>
          </div>
        );
      })}
    </div>
  );
}

// ---------- 6. Acceptance & proposer effectiveness ----------
function AcceptanceBars({ pop }) {
  const W = 560, H = 240, pad = 46;
  const m = {};
  for (const p of pop) {
    const k = p.method || "?";
    (m[k] = m[k] || { acc: 0, rej: 0 });
    if (p.accepted) m[k].acc++; else m[k].rej++;
  }
  const methods = Object.keys(m);
  if (!methods.length) return <Empty />;
  const maxN = Math.max(1, ...methods.map((k) => m[k].acc + m[k].rej));
  const ymax = maxN;
  const sy = (y) => H - pad - (H - 2 * pad) * y / (ymax || 1);
  const groupW = (W - 2 * pad) / methods.length;
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      <Frame W={W} H={H} pad={pad} ymin={0} ymax={ymax} ylabel="count" />
      {methods.map((k, i) => {
        const gx = pad + i * groupW + groupW * 0.18;
        const bw = groupW * 0.28;
        const acc = m[k].acc, rej = m[k].rej;
        return (
          <g key={k}>
            <rect x={gx} y={sy(acc)} width={bw} height={H - pad - sy(acc)} fill={C_ACCEPT}><title>{`${k}: ${acc} accepted`}</title></rect>
            <rect x={gx + bw + 4} y={sy(rej)} width={bw} height={H - pad - sy(rej)} fill={C_REJECT}><title>{`${k}: ${rej} rejected`}</title></rect>
            <text x={gx + bw} y={H - pad + 12} fill={MUTED} fontSize="10" textAnchor="middle">{k}</text>
          </g>
        );
      })}
      <text x={pad} y={16} fill={C_ACCEPT} fontSize="10">■ accepted</text>
      <text x={pad + 70} y={16} fill={C_REJECT} fontSize="10">■ rejected</text>
    </svg>
  );
}

// ---------- 7. Metric-vector radar ----------
function Radar({ candidate }) {
  const vec = candidate?.vector || {};
  const keys = Object.keys(vec);
  if (!keys.length) return <Empty msg="select a candidate with a metric vector" />;
  const W = 320, H = 320, cx = W / 2, cy = H / 2, R = 110;
  const [lo, hi] = extent(keys.map((k) => vec[k]), [0, 1]);
  const norm = (v) => (hi === lo ? 0.5 : (v - lo) / (hi - lo));
  const pt = (i, r) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / keys.length;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  };
  const poly = keys.map((k, i) => pt(i, R * (0.1 + 0.9 * norm(vec[k]))).map((n) => n.toFixed(1)).join(",")).join(" ");
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      {[0.33, 0.66, 1].map((t, i) => (
        <polygon key={i} points={keys.map((_, j) => pt(j, R * t).map((n) => n.toFixed(1)).join(",")).join(" ")}
          fill="none" stroke={AX} />
      ))}
      {keys.map((k, i) => {
        const [ex, ey] = pt(i, R);
        const [lx, ly] = pt(i, R + 16);
        return (
          <g key={k}>
            <line x1={cx} y1={cy} x2={ex} y2={ey} stroke={AX} />
            <text x={lx} y={ly} fill={MUTED} fontSize="9" textAnchor="middle">{k.length > 10 ? k.slice(0, 9) + "…" : k}</text>
          </g>
        );
      })}
      <polygon points={poly} fill="rgba(68,147,248,0.25)" stroke={C_LINE} strokeWidth="2" />
    </svg>
  );
}

// ---------- 8. Genealogy tree (layered by gen) ----------
function GenealogyTree({ pop, lo, hi, selected, onPick }) {
  if (!pop.length) return <Empty />;
  const layers = {};
  for (const p of pop) (layers[p.gen ?? 0] = layers[p.gen ?? 0] || []).push(p);
  const gens = Object.keys(layers).map(Number).sort((a, b) => a - b);
  const colW = 150, rowH = 56, pad = 30;
  const W = Math.max(360, pad * 2 + gens.length * colW);
  const maxRows = Math.max(...gens.map((g) => layers[g].length));
  const H = pad * 2 + maxRows * rowH + 20;
  const pos = {};
  gens.forEach((g, gi) => {
    layers[g].forEach((p, ri) => {
      pos[p.candidate_id] = { x: pad + gi * colW + 30, y: pad + 20 + ri * rowH };
    });
  });
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      {gens.map((g, gi) => (
        <text key={"g" + g} x={pad + gi * colW + 30} y={pad} fill={MUTED} fontSize="10" textAnchor="middle">gen {g}</text>
      ))}
      {pop.map((p) => {
        const a = pos[p.parent_id], b = pos[p.candidate_id];
        if (!a || !b) return null;
        return <line key={"e" + p.candidate_id} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={AX} strokeWidth="1.5" />;
      })}
      {pop.map((p) => {
        const c = pos[p.candidate_id];
        if (!c) return null;
        const sel = p.candidate_id === selected;
        return (
          <g key={p.candidate_id} style={{ cursor: "pointer" }} onClick={() => onPick && onPick(p.candidate_id)}>
            <circle cx={c.x} cy={c.y} r="9"
              fill={eloColor(p.est_elo, lo, hi)}
              stroke={sel ? "#fff" : (p.on_pareto_front ? C_FRONT : AX)}
              strokeWidth={sel ? 3 : (p.on_pareto_front ? 2.5 : 1)}>
              <title>{`${p.candidate_id} · gen ${p.gen} · elo ${fmt(p.est_elo)}`}</title>
            </circle>
            <text x={c.x + 14} y={c.y + 4} fill={MUTED} fontSize="9">{fmt(p.est_elo)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------- candidate list (shared by trajectories + games tabs) ----------
function CandidateList({ rows, selected, onPick }) {
  if (!rows.length) return <Empty msg="no candidates yet" />;
  return (
    <div style={{ minWidth: 240 }}>
      {rows.map((c) => (
        <div key={c.candidate_id} className="card click"
          style={{ marginBottom: 8, borderColor: c.candidate_id === selected ? "var(--accent)" : undefined }}
          onClick={() => onPick(c.candidate_id)}>
          <div className="spread">
            <strong className="mono small">{c.candidate_id}</strong>
            <div className="row" style={{ gap: 4 }}>
              {c.on_pareto_front && <span className="tag" style={{ color: C_FRONT, borderColor: C_FRONT }}>frontier</span>}
              <span className="tag" style={c.accepted ? { color: C_ACCEPT, borderColor: C_ACCEPT } : { color: C_REJECT, borderColor: C_REJECT }}>
                {c.accepted ? "accepted" : "rejected"}
              </span>
            </div>
          </div>
          <div className="small muted mono" style={{ marginTop: 4 }}>
            gen {fmt(c.gen)} · {c.method || "?"} · elo {fmt(c.est_elo)} · scalar {fmt(c.scalar)}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------- Trajectories tab ----------
function TrajectoriesTab({ runId, rows, selected, onPick }) {
  const { data: cand, error } = usePoll(
    () => (selected ? api.candidate(runId, selected) : Promise.resolve(null)),
    0, [runId, selected]);
  return (
    <div className="row" style={{ alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
      <CandidateList rows={rows} selected={selected} onPick={onPick} />
      <div style={{ flex: 1, minWidth: 360 }}>
        {!selected && <Empty msg="pick a candidate to inspect its genome + transcript" />}
        {selected && error && <div className="tag error">{error}</div>}
        {selected && !cand && !error && <div className="muted small">loading…</div>}
        {cand && (
          <div>
            <div className="small muted" style={{ marginBottom: 8 }}>
              {cand.candidate_id} · {cand.method || "?"} · scalar {fmt(cand.scalar)}
              {cand.parent_id && <> · derived from <span className="mono">{cand.parent_id}</span></>}
            </div>
            <div className="small muted">system_prompt</div>
            <pre className="code" style={{ maxHeight: 220 }}>{cand.genome?.system_prompt || "(none)"}</pre>
            <div className="small muted" style={{ marginTop: 8 }}>verifier</div>
            <pre className="code" style={{ maxHeight: 220 }}>{cand.genome?.verifier || "(none)"}</pre>
            <div className="small muted" style={{ margin: "12px 0 6px" }}>transcript</div>
            <Trajectory messages={cand.trajectory} />
          </div>
        )}
      </div>
    </div>
  );
}

// ---------- Games tab (candidate game replay on Board) ----------
function GamesTab({ runId, rows, selected, onPick }) {
  const { data: games } = usePoll(
    () => (selected ? api.candidateGames(runId, selected) : Promise.resolve(null)),
    0, [runId, selected]);
  const list = games || [];
  const sims = list.filter((s) => (s.sampled_games || []).length);
  const [simIdx, setSimIdx] = useState(null);
  const [gameId, setGameId] = useState(null);
  const [game, setGame] = useState(null);
  const [ply, setPly] = useState(0);

  const sim = sims.length ? (sims.find((s) => s.idx === simIdx) || sims[sims.length - 1]) : null;
  const simKey = sim?.idx;
  useEffect(() => { setSimIdx(null); setGameId(null); setGame(null); setPly(0); }, [selected]);
  useEffect(() => { if (sim) setGameId(sim.sampled_games[0]); }, [simKey]); // eslint-disable-line
  useEffect(() => {
    if (!sim || !gameId) return;
    setGame(null); setPly(0);
    api.candidateGame(runId, selected, sim.idx, gameId).then(setGame).catch(() => setGame(null));
  }, [runId, selected, simKey, gameId]); // eslint-disable-line

  const randomGame = () => {
    if (!sims.length) return;
    const s = sims[Math.floor(Math.random() * sims.length)];
    setSimIdx(s.idx);
    setGameId(s.sampled_games[Math.floor(Math.random() * s.sampled_games.length)]);
  };

  const moves = game?.moves || [];
  const played = moves.filter((m) => m.uci);
  const cur = played[ply];
  const fen = cur ? cur.fen_after : (moves[0]?.fen_before || null);

  return (
    <div className="row" style={{ alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
      <CandidateList rows={rows} selected={selected} onPick={onPick} />
      <div style={{ flex: 1, minWidth: 360 }}>
        {!selected && <Empty msg="pick a candidate to replay its games" />}
        {selected && !sims.length && <Empty msg="no sampled games for this candidate" />}
        {selected && sims.length > 0 && (
          <>
            <div className="row" style={{ flexWrap: "wrap", gap: 10, marginBottom: 10 }}>
              <div className="field" style={{ margin: 0 }}><label>simulate #</label>
                <select value={sim.idx} onChange={(e) => setSimIdx(Number(e.target.value))}>
                  {sims.map((s) => <option key={s.idx} value={s.idx}>#{s.idx} (score {fmt(s.summary?.score)})</option>)}
                </select></div>
              <div className="field" style={{ margin: 0 }}><label>game</label>
                <select value={gameId ?? sim.sampled_games[0]} onChange={(e) => setGameId(e.target.value)}>
                  {sim.sampled_games.map((g) => <option key={g} value={g}>game {g}</option>)}
                </select></div>
              <button style={{ alignSelf: "flex-end" }} onClick={randomGame}>🎲 Random game</button>
            </div>
            {game && (
              <div className="row" style={{ alignItems: "flex-start", gap: 18, flexWrap: "wrap" }}>
                <div>
                  <Board fen={fen} lastUci={cur?.uci} />
                  <div className="row" style={{ marginTop: 8 }}>
                    <button onClick={() => setPly(0)}>⏮</button>
                    <button onClick={() => setPly((p) => Math.max(0, p - 1))}>◀</button>
                    <input type="range" min="0" max={Math.max(0, played.length - 1)} value={ply} onChange={(e) => setPly(Number(e.target.value))} />
                    <button onClick={() => setPly((p) => Math.min(played.length - 1, p + 1))}>▶</button>
                    <button onClick={() => setPly(played.length - 1)}>⏭</button>
                  </div>
                  <div className="small muted" style={{ marginTop: 6 }}>
                    ply {ply + 1}/{played.length} · {cur ? `${cur.actor} ${cur.san || cur.uci}` : "start"}
                  </div>
                </div>
                <div style={{ minWidth: 220 }}>
                  <table className="kv"><tbody>
                    <tr><td>policy</td><td>{game.policy?.color}</td></tr>
                    <tr><td>opponent</td><td>{game.opponent?.name}</td></tr>
                    <tr><td>result</td><td>{game.result}</td></tr>
                    <tr><td>winner</td><td>{game.winner || "—"}</td></tr>
                    <tr><td>termination</td><td>{game.termination}</td></tr>
                    <tr><td>plies</td><td>{game.num_ply}</td></tr>
                  </tbody></table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ---------- main ----------
export default function OptimizationDetail({ id }) {
  const { data, error } = usePoll(() => api.run(id), 2000, [id]);
  const [tab, setTab] = useState("progress");
  const [selected, setSelected] = useState(null);

  const pop = data?.population || [];
  const candidates = data?.candidates || [];
  // rows for lists: prefer candidates; fall back to population (which carries the same base fields)
  const rows = candidates.length ? candidates : pop;

  const [eloLo, eloHi] = useMemo(() => extent(pop.map((p) => p.est_elo)), [pop]);
  const best = useMemo(() => {
    const withElo = pop.filter((p) => p.est_elo !== null && p.est_elo !== undefined);
    if (!withElo.length) return null;
    return withElo.reduce((a, b) => (b.est_elo > a.est_elo ? b : a));
  }, [pop]);
  const selCand = useMemo(() => pop.find((p) => p.candidate_id === selected) || null, [pop, selected]);

  const pickInTrajectory = (cid) => { setSelected(cid); setTab("trajectories"); };

  if (error) return <div className="wrap"><a onClick={() => go("#/")}>← runs</a><div className="tag error" style={{ marginTop: 12 }}>{error}</div></div>;
  if (!data) return <div className="wrap muted">loading…</div>;

  const run = data.run || {};
  const cfg = run.config || {};
  const opt = cfg.optimization || cfg || {};
  const be = opt.backend || cfg.backend || {};

  return (
    <div className="wrap">
      <a onClick={() => go("#/")}>← runs</a>
      <div className="spread" style={{ margin: "10px 0 4px" }}>
        <h2 style={{ margin: 0 }}>{cfg.label || run.label || id}</h2>
        <div className="row">
          <span className="tag" style={{ color: "var(--warn)", borderColor: "var(--warn)" }}>optimization</span>
          <StatusTag status={data.live ? "running" : run.status} />
          {data.live && <button onClick={() => api.stopRun(id)}>Stop</button>}
        </div>
      </div>
      <div className="small muted mono" style={{ marginBottom: 8 }}>
        recipe {opt.recipe || "—"} · {be.name || "—"} {be.model || ""} ·
        {" "}{pop.length} evaluated · {candidates.length || pop.length} candidates
        {best && <> · best <span style={{ color: "var(--good)" }}>{best.candidate_id} (elo {fmt(best.est_elo)} · scalar {fmt(best.scalar)})</span></>}
      </div>
      {run.error && <div className="tag error" style={{ marginBottom: 12 }}>{run.error}</div>}

      <div className="tabs">
        {["progress", "trajectories", "games"].map((t) =>
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {tab === "progress" && (
        <>
          {pop.length === 0 && <Empty msg="no population yet — charts will populate as candidates are evaluated" />}
          <div className="row" style={{ alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
            <ChartCard title="Elo vs step (population)" hint="filled=accepted · hollow=rejected · gold ring=frontier · click → trajectory">
              <EloScatter pop={pop} lo={eloLo} hi={eloHi} onPick={pickInTrajectory} />
            </ChartCard>
            <ChartCard title="Per-generation convergence" hint="best (solid) & mean (dashed) est. Elo">
              <GenConvergence pop={pop} />
            </ChartCard>
            <ChartCard title="Score spread / diversity per generation">
              <ScoreSpread pop={pop} />
            </ChartCard>
            <ChartCard title="Pareto frontier" hint="pick two axes · gold ring = on frontier · click → trajectory">
              <ParetoScatter pop={pop} lo={eloLo} hi={eloHi} onPick={pickInTrajectory} />
            </ChartCard>
            <ChartCard title={`Per-opponent win-rate — best (${best?.candidate_id || "—"})`}>
              <WinRateHeatmap best={best} />
            </ChartCard>
            <ChartCard title="Acceptance & proposer effectiveness" hint="accepted vs rejected by method">
              <AcceptanceBars pop={pop} />
            </ChartCard>
            <ChartCard title={`Metric-vector radar — ${selCand?.candidate_id || best?.candidate_id || "—"}`} hint="click any point/node to select">
              <Radar candidate={selCand || best} />
            </ChartCard>
            <ChartCard title="Genealogy tree" hint="color = est. Elo · gold outline = frontier · click to select">
              <GenealogyTree pop={pop} lo={eloLo} hi={eloHi} selected={selected} onPick={setSelected} />
            </ChartCard>
          </div>
        </>
      )}

      {tab === "trajectories" && (
        <TrajectoriesTab runId={id} rows={rows} selected={selected} onPick={setSelected} />
      )}
      {tab === "games" && (
        <GamesTab runId={id} rows={rows} selected={selected} onPick={setSelected} />
      )}
    </div>
  );
}
