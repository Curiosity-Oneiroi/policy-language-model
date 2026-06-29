import { useEffect, useState } from "react";
import { api } from "./api.js";
import { go, BOTS, OPT_OPPONENTS } from "./ui.jsx";

const HR = <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "14px 0" }} />;

// Pick a sensible default backend (prefer an available one, prefer vLLM).
function defaultBackend(backends) {
  return (
    backends.find((b) => b.available && /fireworks/i.test(b.name)) ||
    backends.find((b) => b.available) ||
    backends[0] ||
    null
  );
}

function SingleForm({ backends }) {
  const [mps, setMps] = useState([]);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({
    label: "", metaparam: "", backendName: "", model: "", base_url: "",
    opponents: ["random", "greedy"], clock: "per_move", per_move_s: 1.0,
    game_clock_s: 60, max_moves: 300, evaluate: false, on_illegal: "forfeit",
    seed: 0, max_turns: 100, return_budget: 5, task: "",
  });
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  useEffect(() => { api.metaparams().then(setMps).catch(() => {}); }, []);
  useEffect(() => { if (mps.length && !form.metaparam) set("metaparam", mps[0].name); }, [mps]); // eslint-disable-line
  useEffect(() => {
    if (backends.length && !form.backendName) {
      const a = defaultBackend(backends);
      if (a) setForm((f) => ({ ...f, backendName: a.name, model: a.default_model || "" }));
    }
  }, [backends]); // eslint-disable-line

  const backend = backends.find((b) => b.name === form.backendName);
  const toggleOpp = (b) =>
    set("opponents", form.opponents.includes(b) ? form.opponents.filter((x) => x !== b) : [...form.opponents, b]);

  const submit = async () => {
    setErr(null); setBusy(true);
    try {
      const r = await api.createRun({
        kind: "single",
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

  return (
    <>
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

        {HR}
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

        {HR}
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
    </>
  );
}

// A backend/model/base_url block reused for both the policy backend and the reflection model.
function BackendBlock({ title, backends, value, onChange }) {
  const backend = backends.find((b) => b.name === value.name);
  const set = (k, v) => onChange({ ...value, [k]: v });
  return (
    <div className="row" style={{ flexWrap: "wrap", alignItems: "flex-start" }}>
      <div className="field" style={{ flex: 1, minWidth: 180 }}><label>{title} backend</label>
        <select value={value.name} onChange={(e) => { const b = backends.find((x) => x.name === e.target.value); set("name", e.target.value); onChange({ ...value, name: e.target.value, model: b?.default_model || value.model }); }}>
          {backends.map((b) => <option key={b.name} value={b.name} disabled={!b.available}>{b.name}{b.available ? "" : " (no key)"}</option>)}
        </select></div>
      <div className="field" style={{ flex: 1, minWidth: 180 }}><label>{title} model</label>
        <input value={value.model} onChange={(e) => set("model", e.target.value)} /></div>
      {backend?.supports_base_url &&
        <div className="field" style={{ flex: 1, minWidth: 180 }}><label>{title} base_url</label>
          <input value={value.base_url} onChange={(e) => set("base_url", e.target.value)} /></div>}
    </div>
  );
}

const OPT_DEFAULTS = {
  model: "accounts/fireworks/models/qwen3p7-plus",
  base_url: "https://api.fireworks.ai/inference/v1",
};

function OptForm({ backends }) {
  const [recipes, setRecipes] = useState([]);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({
    label: "", recipe: "E4",
    backend: { name: "", model: OPT_DEFAULTS.model, base_url: OPT_DEFAULTS.base_url },
    reflection: { name: "", model: OPT_DEFAULTS.model, base_url: OPT_DEFAULTS.base_url },
    opponents: ["maia-1500", "stockfish-1320", "maia-1100"], clock: "per_move", per_move_s: 2,
    game_clock_s: 100, max_moves: 120, evaluate: true, games_per_opponent: 3,
    generations: 5, max_metric_calls: 30, population_size: 10, max_turns: 15, return_budget: 5,
  });
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  useEffect(() => { api.recipes().then(setRecipes).catch(() => {}); }, []);
  useEffect(() => {
    if (recipes.length && !recipes.find((r) => r.id === form.recipe)) set("recipe", recipes[0].id);
  }, [recipes]); // eslint-disable-line
  useEffect(() => {
    if (backends.length && !form.backend.name) {
      const a = defaultBackend(backends);
      if (!a) return;
      const sb = !!a.supports_base_url;
      const blk = { name: a.name, model: OPT_DEFAULTS.model || a.default_model, base_url: sb ? OPT_DEFAULTS.base_url : "" };
      setForm((f) => ({ ...f, backend: { ...blk }, reflection: { ...blk } }));
    }
  }, [backends]); // eslint-disable-line

  const toggleOpp = (b) =>
    set("opponents", form.opponents.includes(b) ? form.opponents.filter((x) => x !== b) : [...form.opponents, b]);

  const submit = async () => {
    setErr(null); setBusy(true);
    try {
      const clean = (b) => ({ name: b.name, model: b.model || null, base_url: b.base_url || null });
      const r = await api.createRun({
        kind: "optimization",
        label: form.label || null,
        optimization: {
          recipe: form.recipe,
          backend: clean(form.backend),
          reflection: clean(form.reflection),
          simulate_config: {
            opponents: form.opponents, clock: form.clock,
            per_move_s: Number(form.per_move_s), game_clock_s: Number(form.game_clock_s),
            max_moves: Number(form.max_moves), evaluate: form.evaluate,
            games_per_opponent: Number(form.games_per_opponent),
          },
          max_turns: Number(form.max_turns),
          return_budget: Number(form.return_budget),
          generations: Number(form.generations),
          max_metric_calls: Number(form.max_metric_calls),
          population_size: Number(form.population_size),
        },
      });
      go("#/opt/" + r.run_id);
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  const recipe = recipes.find((r) => r.id === form.recipe);
  return (
    <>
      {err && <div className="tag error" style={{ marginBottom: 12 }}>{err}</div>}
      <div className="card">
        <div className="field"><label>Label</label>
          <input value={form.label} onChange={(e) => set("label", e.target.value)} placeholder="e.g. e4-optimize-pop3" /></div>
        <div className="field"><label>Recipe</label>
          <select value={form.recipe} onChange={(e) => set("recipe", e.target.value)} title={recipe?.summary || ""}>
            {recipes.length === 0 && <option value={form.recipe}>{form.recipe}</option>}
            {recipes.map((r) => <option key={r.id} value={r.id} title={r.summary}>{r.id} — {r.title}</option>)}
          </select>
          {recipe?.summary && <div className="small muted" style={{ marginTop: 4 }}>{recipe.summary}</div>}
        </div>

        {HR}
        <div className="muted small" style={{ marginBottom: 8 }}>Models</div>
        <BackendBlock title="Policy" backends={backends} value={form.backend} onChange={(v) => set("backend", v)} />
        <BackendBlock title="Reflection" backends={backends} value={form.reflection} onChange={(v) => set("reflection", v)} />

        {HR}
        <div className="muted small" style={{ marginBottom: 8 }}>Opponent ladder & evaluation</div>
        <div className="field"><label>Opponents</label>
          <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
            {OPT_OPPONENTS.map((b) => (
              <span key={b} className={"tag " + (form.opponents.includes(b) ? "running" : "")}
                style={{ cursor: "pointer" }} onClick={() => toggleOpp(b)}>{b}</span>
            ))}
          </div></div>
        <div className="row">
          <div className="field" style={{ flex: 1 }}><label>Clock</label>
            <select value={form.clock} onChange={(e) => set("clock", e.target.value)}>
              <option value="per_move">per_move</option><option value="cumulative">cumulative</option>
            </select></div>
          <div className="field" style={{ flex: 1 }}><label>per_move_s</label>
            <input type="number" step="0.5" value={form.per_move_s} onChange={(e) => set("per_move_s", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>game_clock_s</label>
            <input type="number" value={form.game_clock_s} onChange={(e) => set("game_clock_s", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>max_moves</label>
            <input type="number" value={form.max_moves} onChange={(e) => set("max_moves", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>games / opponent</label>
            <input type="number" value={form.games_per_opponent} onChange={(e) => set("games_per_opponent", e.target.value)} /></div>
        </div>
        <label className="row small" style={{ gap: 6 }}>
          <input type="checkbox" style={{ width: "auto" }} checked={form.evaluate} onChange={(e) => set("evaluate", e.target.checked)} />
          evaluate (Stockfish cpl / blunder annotations)
        </label>

        {HR}
        <div className="muted small" style={{ marginBottom: 8 }}>Search budget</div>
        <div className="row">
          <div className="field" style={{ flex: 1 }}><label>generations</label>
            <input type="number" value={form.generations} onChange={(e) => set("generations", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>population_size</label>
            <input type="number" value={form.population_size} onChange={(e) => set("population_size", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>max_metric_calls</label>
            <input type="number" value={form.max_metric_calls} onChange={(e) => set("max_metric_calls", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>max_turns</label>
            <input type="number" value={form.max_turns} onChange={(e) => set("max_turns", e.target.value)} /></div>
          <div className="field" style={{ flex: 1 }}><label>return_budget</label>
            <input type="number" value={form.return_budget} onChange={(e) => set("return_budget", e.target.value)} /></div>
        </div>
        <button className="primary" onClick={submit} disabled={busy || !form.recipe || !form.backend.name}>
          {busy ? "Launching…" : "Launch optimization"}
        </button>
      </div>
    </>
  );
}

export default function Create() {
  const [backends, setBackends] = useState([]);
  const [kind, setKind] = useState("single");
  useEffect(() => { api.backends().then(setBackends).catch(() => {}); }, []);

  return (
    <div className="wrap" style={{ maxWidth: 820 }}>
      <h2>New experiment</h2>
      <div className="tabs" style={{ marginBottom: 16 }}>
        <button className={kind === "single" ? "active" : ""} onClick={() => setKind("single")}>Single experiment</button>
        <button className={kind === "optimization" ? "active" : ""} onClick={() => setKind("optimization")}>Optimization experiment</button>
      </div>
      {kind === "single"
        ? <SingleForm backends={backends} />
        : <OptForm backends={backends} />}
    </div>
  );
}
