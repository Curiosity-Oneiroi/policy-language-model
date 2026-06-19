import { useEffect, useRef, useState } from "react";

export function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return String(Math.round(v * 1000) / 1000);
  return String(v);
}

async function req(path, opts) {
  const r = await fetch("/api" + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return r.json();
}

export const api = {
  settings: () => req("/settings"),
  setSettings: (num_workspaces) =>
    req("/settings", { method: "POST", body: JSON.stringify({ num_workspaces }) }),
  workspaces: () => req("/workspaces"),
  metaparams: () => req("/metaparams"),
  backends: () => req("/backends"),
  runs: () => req("/runs"),
  run: (id) => req("/runs/" + id),
  createRun: (spec) => req("/runs", { method: "POST", body: JSON.stringify(spec) }),
  stopRun: (id) => req("/runs/" + id + "/stop", { method: "POST" }),
  game: (id, simIdx, gameId) => req(`/runs/${id}/games/${simIdx}/${gameId}`),
  recipes: () => req("/recipes"),
  candidates: (runId) => req(`/runs/${runId}/candidates`),
  candidate: (runId, cid) => req(`/runs/${runId}/candidates/${cid}`),
  candidateGames: (runId, cid) => req(`/runs/${runId}/candidates/${cid}/games`),
  candidateGame: (runId, cid, simIdx, gameId) =>
    req(`/runs/${runId}/candidates/${cid}/games/${simIdx}/${gameId}`),
};

// Poll a fetcher every `ms` (and immediately). Returns {data, error, refresh}.
export function usePoll(fetcher, ms, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const fref = useRef(fetcher);
  fref.current = fetcher;
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    const go = () => fref.current().then(
      (d) => alive && (setData(d), setError(null)),
      (e) => alive && setError(e.message),
    );
    go();
    if (!ms) return () => { alive = false; };
    const h = setInterval(go, ms);
    return () => { alive = false; clearInterval(h); };
  }, [ms, tick, ...deps]); // eslint-disable-line
  return { data, error, refresh: () => setTick((t) => t + 1) };
}
