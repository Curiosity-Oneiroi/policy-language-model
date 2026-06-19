import { useEffect, useState } from "react";
import { api, usePoll, fmt } from "./api.js";
import { go, StatusTag } from "./ui.jsx";
import Chart from "./Chart.jsx";
import Board from "./Board.jsx";

export function codeOf(msg) {
  const tc = (msg.tool_calls || [])[0];
  if (!tc) return null;
  let args = tc.function?.arguments;
  if (typeof args === "string") { try { args = JSON.parse(args); } catch { return args; } }
  return args?.code ?? null;
}

export function Trajectory({ messages }) {
  if (!messages?.length) return <div className="muted">No transcript yet.</div>;
  return (
    <div>
      {messages.map((m, i) => {
        const code = m.role === "assistant" ? codeOf(m) : null;
        return (
          <div key={i} className={`msg ${m.role}`}>
            <div className="head">{m.role}{m.tool_status ? " · " + m.tool_status : ""}</div>
            {m.reasoning ? <div className="body muted small">{m.reasoning}</div> : null}
            {m.content ? <div className="body">{typeof m.content === "string" ? m.content : JSON.stringify(m.content)}</div> : null}
            {code ? <div className="body codeblock">{code}</div> : null}
          </div>
        );
      })}
    </div>
  );
}

function Policies({ simulates }) {
  // Derive the genealogy from what was simulated: each policy NAME -> its score history +
  // latest source. policyzero is the canonical one; other names are forks/candidates.
  const [sel, setSel] = useState(null);
  const byName = {};
  for (const s of simulates) {
    const n = s.policy_name || "?";
    (byName[n] = byName[n] || { name: n, history: [], source: null });
    byName[n].history.push({ x: s.idx, y: s.summary?.score });
    if (s.policy_source) byName[n].source = s.policy_source;
  }
  const names = Object.keys(byName).sort((a, b) => (a === "policyzero" ? -1 : b === "policyzero" ? 1 : a.localeCompare(b)));
  if (!names.length) return <div className="muted">No policies simulated yet.</div>;
  const cur = sel && byName[sel] ? sel : names[0];
  const p = byName[cur];
  const best = Math.max(...p.history.map((h) => h.y ?? -1));
  return (
    <div className="row" style={{ alignItems: "flex-start", gap: 16 }}>
      <div style={{ minWidth: 200 }}>
        {names.map((n) => (
          <div key={n} className={"card click"} style={{ marginBottom: 8, borderColor: n === cur ? "var(--accent)" : undefined }} onClick={() => setSel(n)}>
            <div className="spread"><strong className="mono">{n}</strong>{n === "policyzero" ? <span className="tag">canonical</span> : null}</div>
            <div className="small muted">{byName[n].history.length} tests · best {fmt(Math.max(...byName[n].history.map((h) => h.y ?? -1)))}</div>
          </div>
        ))}
      </div>
      <div style={{ flex: 1 }}>
        <div className="small muted" style={{ marginBottom: 6 }}>{cur} · best score {fmt(best)} · source at last test</div>
        <pre className="code">{p.source || "(source not captured)"}</pre>
      </div>
    </div>
  );
}

function GameReplay({ runId, simulates }) {
  const sims = simulates.filter((s) => (s.sampled_games || []).length);
  const [simIdx, setSimIdx] = useState(null);
  const [gameId, setGameId] = useState(null);
  const [game, setGame] = useState(null);
  const [ply, setPly] = useState(0);
  const sim = sims.length ? (sims.find((s) => s.idx === simIdx) || sims[sims.length - 1]) : null;
  const simKey = sim?.idx;
  useEffect(() => { if (sim) setGameId(sim.sampled_games[0]); }, [simKey]); // eslint-disable-line
  useEffect(() => {
    if (!sim || !gameId) return;
    setGame(null); setPly(0);
    api.game(runId, sim.idx, gameId).then(setGame).catch(() => setGame(null));
  }, [runId, simKey, gameId]); // eslint-disable-line
  if (!sims.length) return <div className="muted">No sampled games yet.</div>;

  const moves = game?.moves || [];
  const played = moves.filter((m) => m.uci);
  const cur = played[ply];
  const fen = cur ? cur.fen_after : (moves[0]?.fen_before || null);
  return (
    <div>
      <div className="row" style={{ flexWrap: "wrap", gap: 10, marginBottom: 10 }}>
        <div className="field" style={{ margin: 0 }}><label>simulate #</label>
          <select value={sim.idx} onChange={(e) => setSimIdx(Number(e.target.value))}>
            {sims.map((s) => <option key={s.idx} value={s.idx}>#{s.idx} (score {fmt(s.summary?.score)})</option>)}
          </select></div>
        <div className="field" style={{ margin: 0 }}><label>game</label>
          <select value={gameId ?? sim.sampled_games[0]} onChange={(e) => setGameId(e.target.value)}>
            {sim.sampled_games.map((g) => <option key={g} value={g}>game {g}</option>)}
          </select></div>
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
            <div className="small muted" style={{ margin: "10px 0 4px" }}>policyzero source at simulate #{sim.idx}</div>
            <pre className="code" style={{ maxHeight: 240 }}>{sim.policy_source || "(not captured)"}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default function RunDetail({ id }) {
  const { data, error } = usePoll(() => api.run(id), 2000, [id]);
  const [tab, setTab] = useState("progress");
  if (error) return <div className="wrap"><a onClick={() => go("#/")}>← runs</a><div className="tag error" style={{ marginTop: 12 }}>{error}</div></div>;
  if (!data) return <div className="wrap muted">loading…</div>;

  const run = data.run || {};
  const cfg = run.config || {};
  const sims = data.simulates || [];
  const scorePts = sims.map((s) => ({ x: s.idx, y: s.summary?.score ?? null }));
  const eloPts = sims.map((s) => ({ x: s.idx, y: s.summary?.est_elo ?? null }));

  return (
    <div className="wrap">
      <a onClick={() => go("#/")}>← runs</a>
      <div className="spread" style={{ margin: "10px 0 4px" }}>
        <h2 style={{ margin: 0 }}>{cfg.label || id}</h2>
        <div className="row">
          <StatusTag status={data.live ? "running" : run.status} />
          {data.live && <button onClick={() => api.stopRun(id)}>Stop</button>}
        </div>
      </div>
      <div className="small muted mono" style={{ marginBottom: 12 }}>
        {Path(cfg.metaparam_dir)} · {cfg.backend?.name} {cfg.backend?.model || ""} · seed {cfg.seed} ·
        opponents [{(cfg.simulate_config?.opponents || []).join(", ")}] · clock {cfg.simulate_config?.clock} ·
        {sims.length} simulates
      </div>
      {run.error && <div className="tag error" style={{ marginBottom: 12 }}>{run.error}</div>}

      <div className="tabs">
        {["progress", "trajectory", "policies", "games"].map((t) =>
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {tab === "progress" && (
        <div className="row" style={{ alignItems: "flex-start", gap: 24, flexWrap: "wrap" }}>
          <div className="card"><div className="small muted">score vs simulate</div>
            <Chart points={scorePts} ylabel="score" domain={[0, 1]} color="#3fb950" onPick={() => setTab("games")} /></div>
          <div className="card"><div className="small muted">est. Elo vs simulate</div>
            <Chart points={eloPts} ylabel="elo" color="#4493f8" onPick={() => setTab("games")} /></div>
          {data.result && (
            <div className="card" style={{ minWidth: 260 }}><div className="small muted">final</div>
              <pre className="code" style={{ maxHeight: 200 }}>{JSON.stringify(data.result, null, 2)}</pre></div>
          )}
        </div>
      )}
      {tab === "trajectory" && <Trajectory messages={data.trajectory?.messages} />}
      {tab === "policies" && <Policies simulates={sims} />}
      {tab === "games" && <GameReplay runId={id} simulates={sims} />}
    </div>
  );
}

function Path(p) { return p ? String(p).split("/").pop() : "—"; }
