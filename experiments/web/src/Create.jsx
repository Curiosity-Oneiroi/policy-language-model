import { useEffect, useState } from "react";
import { api } from "./api.js";
import { go, BOTS } from "./ui.jsx";

export default function Create() {
  const [mps, setMps] = useState([]);
  const [backends, setBackends] = useState([]);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({
    label: "", metaparam: "", backendName: "", model: "", base_url: "",
    opponents: ["random", "greedy"], clock: "per_move", per_move_s: 1.0,
    game_clock_s: 60, max_moves: 300, evaluate: false, on_illegal: "forfeit",
    seed: 0, max_turns: 100, return_budget: 5, task: "",
  });
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  useEffect(() => { api.metaparams().then(setMps).catch(() => {}); api.backends().then(setBackends).catch(() => {}); }, []);
  useEffect(() => { if (mps.length && !form.metaparam) set("metaparam", mps[0].name); }, [mps]); // eslint-disable-line
  useEffect(() => {
    if (backends.length && !form.backendName) {
      const a = backends.find((b) => b.available) || backends[0];
      setForm((f) => ({ ...f, backendName: a.name, model: a.default_model || "" }));
    }
  }, [backends]); // eslint-disable-line

  const backend = backends.find((b) => b.name === form.backendName);
  const toggleOpp = (b) =>
    set("opponents", form.opponents.includes(b) ? form.opponents.filter((x) => x !== b) : [...form.opponents, b]);

  const submit = async () => {
    setErr(null); setBusy(true);
    try {
      const r = await api.createRun({
        metaparam: form.metaparam,
        backend: { name: form.backendName, model: form.model || null, base_url: form.base_url || null },
        simulate_config: {
          opponents: form.opponents, clock: form.clock,
          per_move_s: Number(form.per_move_s), game_clock_s: Number(form.game_clock_s),
          max_moves: Number(form.max_moves), evaluate: form.evaluate, on_illegal: form.on_illegal,
        },
        seed: Number(form.seed), max_turns: Number(form.max_turns),
        return_budget: Number(form.return_budget), task: form.task || null, label: form.label || null,
      });
      go("#/run/" + r.run_id);
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  const hr = <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "14px 0" }} />;
  return (
    <div className="wrap" style={{ maxWidth: 780 }}>
      <h2>New experiment</h2>
      {err && <div className="tag error" style={{ marginBottom: 12 }}>{err}</div>}
      <div className="card">
        <div className="field"><label>Label</label>
          <input value={form.label} onChange={(e) => set("label", e.target.value)} placeholder="e.g. chess-baseline-seed0" /></div>
        <div className="field"><label>Metaparam set</label>
          <select value={form.metaparam} onChange={(e) => set("metaparam", e.target.value)}>
            {mps.map((m) => <option key={m.name} value={m.name}>{m.name} ({m.mutable_policies.join(", ") || "no policies"})</option>)}
          </select></div>
        <div className="row">
          <div className="field" style={{ flex: 1 }}><label>Backend</label>
            <select value={form.backendName} onChange={(e) => { const b = backends.find((x) => x.name === e.target.value); set("backendName", e.target.value); set("model", b?.default_model || ""); }}>
              {backends.map((b) => <option key={b.name} value={b.name} disabled={!b.available}>{b.name}{b.available ? "" : " (no key)"}</option>)}
            </select></div>
          <div className="field" style={{ flex: 1 }}><label>Model</label>
            <input value={form.model} onChange={(e) => set("model", e.target.value)} /></div>
        </div>
        {backend?.supports_base_url &&
          <div className="field"><label>Base URL (optional)</label>
            <input value={form.base_url} onChange={(e) => set("base_url", e.target.value)} /></div>}

        {hr}
        <div className="muted small" style={{ marginBottom: 8 }}>
          Evaluation conditions — FIXED for this experiment (the policy can't change them)
        </div>
        <div className="field"><label>Opponents (drawn uniformly per game)</label>
          <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
            {BOTS.map((b) => (
              <span key={b} className={"tag " + (form.opponents.includes(b) ? "running" : "")}
                style={{ cursor: "pointer" }} onClick={() => toggleOpp(b)}>{b}</span>
            ))}
          </div></div>
        <div className="row">
          <div className="field" style={{ flex: 1 }}><label>Clock</label>
            <select value={form.clock} onChange={(e) => set("clock", e.target.value)}>
              <option value="per_move">per_move</option><option value="cumulative">cumulative</option><option value="off">off</option>
            </select></div>
          <div className="field" style={{ flex: 1 }}><label>per_move_s</label>
            <input type="number" step="0.1" value={form.per_move_s} onChange={(e) => set("per_move_s", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>game_clock_s</label>
            <input type="number" value={form.game_clock_s} onChange={(e) => set("game_clock_s", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>max_moves</label>
            <input type="number" value={form.max_moves} onChange={(e) => set("max_moves", e.target.value)} /></div>
        </div>
        <label className="row small" style={{ gap: 6 }}>
          <input type="checkbox" style={{ width: "auto" }} checked={form.evaluate} onChange={(e) => set("evaluate", e.target.checked)} />
          evaluate (Stockfish cpl / blunder annotations — needs a Stockfish binary)
        </label>

        {hr}
        <div className="row">
          <div className="field" style={{ flex: 1 }}><label>seed</label>
            <input type="number" value={form.seed} onChange={(e) => set("seed", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>max_turns</label>
            <input type="number" value={form.max_turns} onChange={(e) => set("max_turns", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>return_budget</label>
            <input type="number" value={form.return_budget} onChange={(e) => set("return_budget", e.target.value)} /></div>
        </div>
        <div className="field"><label>Task / kickoff message (optional)</label>
          <textarea rows={3} value={form.task} onChange={(e) => set("task", e.target.value)}
            placeholder="Improve policyzero to beat the configured opponents. Use simulate(...) to evaluate." /></div>
        <button className="primary" onClick={submit} disabled={busy || !form.metaparam || !form.backendName}>
          {busy ? "Launching…" : "Launch experiment"}
        </button>
      </div>
    </div>
  );
}
