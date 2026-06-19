import { useEffect, useState } from "react";
import { api, usePoll, fmt } from "./api.js";
import { go, StatusTag, KindBadge } from "./ui.jsx";
import Create from "./Create.jsx";
import RunDetail from "./RunDetail.jsx";
import OptimizationDetail from "./OptimizationDetail.jsx";
import Compare from "./Compare.jsx";

function useHashRoute() {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const h = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", h);
    return () => window.removeEventListener("hashchange", h);
  }, []);
  return hash;
}

function ExperimentsList() {
  const { data: runs, error } = usePoll(api.runs, 2000);
  return (
    <div className="wrap">
      <div className="spread" style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Experiments</h2>
        <div className="row">
          <button onClick={() => go("#/compare")}>Compare</button>
          <button className="primary" onClick={() => go("#/new")}>+ New experiment</button>
        </div>
      </div>
      {error && <div className="tag error">{error}</div>}
      {runs && runs.length === 0 && <div className="muted">No runs yet — create one.</div>}
      <div className="grid">
        {(runs || []).map((r) => {
          const opt = r.kind === "optimization";
          const dest = opt ? "#/opt/" + r.run_id : "#/run/" + r.run_id;
          return (
            <div key={r.run_id} className="card click" onClick={() => go(dest)}>
              <div className="spread">
                <strong>{r.label || r.run_id}</strong>
                <div className="row" style={{ gap: 6 }}>
                  <KindBadge kind={r.kind} />
                  <StatusTag status={r.live ? "running" : r.status} />
                </div>
              </div>
              <div className="small muted mono" style={{ margin: "6px 0" }}>
                {opt ? (r.recipe || "—") : r.metaparam} · {r.backend?.name} · {r.backend?.model || ""}
              </div>
              {opt ? (
                <table className="kv"><tbody>
                  <tr><td>candidates</td><td>{fmt(r.num_candidates)}</td></tr>
                  <tr><td>best scalar</td><td>{fmt(r.best_scalar)}</td></tr>
                  <tr><td>best Elo</td><td>{fmt(r.best_elo)}</td></tr>
                </tbody></table>
              ) : (
                <table className="kv"><tbody>
                  <tr><td>simulates</td><td>{r.num_simulates}</td></tr>
                  <tr><td>latest score</td><td>{fmt(r.latest_score)}</td></tr>
                  <tr><td>est. Elo</td><td>{fmt(r.latest_elo)}</td></tr>
                </tbody></table>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Settings() {
  const { data, refresh } = usePoll(api.settings, 2000);
  const [n, setN] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (data && n === "") setN(String(data.num_workspaces ?? 0)); }, [data]); // eslint-disable-line
  const save = async () => {
    setBusy(true);
    try { await api.setSettings(parseInt(n || "0", 10)); refresh(); } finally { setBusy(false); }
  };
  const pool = data?.pool;
  return (
    <div className="wrap">
      <h2>Settings</h2>
      <div className="card" style={{ maxWidth: 520 }}>
        <div className="field">
          <label>Number of workspaces (the kernel-venv pool)</label>
          <div className="row">
            <input type="number" min="0" value={n} onChange={(e) => setN(e.target.value)} />
            <button className="primary" onClick={save} disabled={busy}>Apply</button>
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>
            Raising provisions new venvs (game + deps) in the background. Lowering deletes the
            highest-numbered FREE workspaces (allocated ones are pruned when they finish).
          </div>
        </div>
      </div>
      {pool && (
        <div style={{ marginTop: 16 }}>
          <div className="muted small">target {pool.target} · {pool.count} present</div>
          <div className="grid" style={{ marginTop: 8 }}>
            {pool.workspaces.map((w) => (
              <div key={w.number} className="card">
                <div className="spread">
                  <strong className="mono">#{String(w.number).padStart(4, "0")}</strong>
                  <StatusTag status={w.status} />
                </div>
                {w.run_id && <div className="small muted mono" style={{ marginTop: 4 }}>{w.run_id}</div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const hash = useHashRoute();
  let view;
  if (hash.startsWith("#/new")) view = <Create />;
  else if (hash.startsWith("#/opt/")) view = <OptimizationDetail id={decodeURIComponent(hash.slice(6))} />;
  else if (hash.startsWith("#/run/")) view = <RunDetail id={decodeURIComponent(hash.slice(6))} />;
  else if (hash.startsWith("#/settings")) view = <Settings />;
  else if (hash.startsWith("#/compare")) view = <Compare />;
  else view = <ExperimentsList />;

  const tab = (h, label) => {
    const active = h === "#/" ? (hash === "#/" || hash === "") : hash.startsWith(h);
    return <a className={active ? "active" : ""} onClick={() => go(h)}>{label}</a>;
  };
  return (
    <>
      <div className="topbar">
        <span className="brand" style={{ cursor: "pointer" }} onClick={() => go("#/")}>♟ PLM EXPERIMENTS</span>
        <nav className="row">
          {tab("#/", "Runs")}
          {tab("#/compare", "Compare")}
          {tab("#/settings", "Settings")}
        </nav>
      </div>
      {view}
    </>
  );
}
