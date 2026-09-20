"use client";

import {
  Activity,
  ArrowRight,
  Beaker,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleDollarSign,
  Command,
  Dna,
  FlaskConical,
  Gauge,
  GitBranch,
  Hexagon,
  LoaderCircle,
  Menu,
  MoreHorizontal,
  Pause,
  Play,
  Plus,
  Rocket,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  Trophy,
  X,
  Zap,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_NEOFORM_API ?? "http://127.0.0.1:8787";

type Status =
  | "draft"
  | "running"
  | "paused"
  | "completed"
  | "cancelled"
  | "failed";

type Candidate = {
  id: string;
  generation: number;
  ordinal: number;
  parent_id: string | null;
  genotype: {
    learning_rate: number;
    temperature: number;
    max_tokens: number;
    top_p: number;
    loss_function: string;
    advantage_clip: number;
    optimizer_mode: string;
    curriculum_seed: number;
  };
  status: string;
  score: number | null;
  state_checkpoint: string | null;
  sampler_checkpoint: string | null;
  metrics: null | {
    scores: Record<string, number>;
    training_loss: number;
    latency_ms: number;
    usage: { train_tokens: number; sample_tokens: number; prompt_tokens: number };
    examples: Array<{ prompt: string; response: string; reward: number }>;
  };
  error: string | null;
};

type Evolution = {
  id: string;
  name: string;
  status: Status;
  champion_id: string | null;
  spent: number;
  reserved: number;
  created_at: string;
  spec: {
    base_model: string;
    max_usd: number;
    search: { population: number; generations: number; survivors: number };
    benchmarks: Array<{ name: string; weight: number; examples: number }>;
  };
  candidates?: Candidate[];
};

const defaultSpec = {
  name: "Reasoning frontier",
  base_model: "Qwen/Qwen3.5-4B",
  seed: 2026,
  max_usd: 10,
  checkpoint_ttl_days: 7,
  search: {
    population: 4,
    generations: 3,
    survivors: 2,
    rank: 32,
    steps_per_candidate: 25,
    batch_size: 8,
    max_sequence_tokens: 1024,
    max_concurrency: 2,
  },
  mutations: {
    learning_rate_min: 0.00002,
    learning_rate_max: 0.0002,
    temperatures: [0, 0.6, 0.8, 1],
    max_tokens: [32, 64, 128, 256],
    top_p: [0.9, 1],
    loss_functions: ["cross_entropy"],
    advantage_clips: [1],
    optimizer_modes: ["resume", "reset"],
    parent_selection: "rank",
  },
  benchmarks: [
    { name: "arithmetic_exact_match", weight: 1, examples: 4 },
  ],
};

const benchmarkPresets = {
  arithmetic_exact_match: {
    label: "Arithmetic exact match",
    benchmarks: [{ name: "arithmetic_exact_match", weight: 1, examples: 4 }],
  },
  gsm8k_exact_match: {
    label: "GSM8K-style reasoning",
    benchmarks: [{ name: "gsm8k_exact_match", weight: 1, examples: 8 }],
  },
  instruction_following: {
    label: "Instruction following",
    benchmarks: [{ name: "instruction_following", weight: 1, examples: 8 }],
  },
  mixed: {
    label: "Weighted mix",
    benchmarks: [
      { name: "gsm8k_exact_match", weight: 0.6, examples: 8 },
      { name: "arithmetic_exact_match", weight: 0.2, examples: 4 },
      { name: "instruction_following", weight: 0.2, examples: 6 },
    ],
  },
} as const;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail ?? "Request failed");
  }
  return response.json();
}

function formatScore(score: number | null) {
  return score == null ? "—" : `${(score * 100).toFixed(1)}%`;
}

function shortId(id: string) {
  return id.replace("cand_", "C-").slice(0, 13).toUpperCase();
}

function StatusDot({ status }: { status: string }) {
  return <span className={`status-dot status-${status}`} aria-label={status} />;
}

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <span>NEOFORM</span>
      <em>LAB</em>
    </div>
  );
}

export function Lab() {
  const [evolutions, setEvolutions] = useState<Evolution[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [evolution, setEvolution] = useState<Evolution | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<string | null>(null);
  const [launchOpen, setLaunchOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] = useState<"checking" | "ready" | "missing">("checking");
  const [message, setMessage] = useState("");
  const [reply, setReply] = useState("");

  const loadList = useCallback(async () => {
    try {
      const result = await request<{ data: Evolution[] }>("/api/v1/evolutions");
      setEvolutions(result.data);
      setSelectedId((current) => current ?? result.data[0]?.id ?? null);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Cannot reach NEOFORM");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadEvolution = useCallback(async (id: string) => {
    try {
      const result = await request<Evolution>(`/api/v1/evolutions/${id}`);
      setEvolution(result);
      setSelectedCandidate((current) => {
        if (current && result.candidates?.some((candidate) => candidate.id === current)) {
          return current;
        }
        return result.champion_id ?? result.candidates?.at(-1)?.id ?? null;
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Evolution unavailable");
    }
  }, []);

  useEffect(() => {
    void loadList();
    request<{ ok: boolean }>("/api/v1/doctor")
      .then(() => setConnection("ready"))
      .catch(() => setConnection("missing"));
  }, [loadList]);

  useEffect(() => {
    if (!selectedId) return;
    void loadEvolution(selectedId);
    const interval = window.setInterval(() => void loadEvolution(selectedId), 2000);
    return () => window.clearInterval(interval);
  }, [loadEvolution, selectedId]);

  const candidates = useMemo(() => evolution?.candidates ?? [], [evolution?.candidates]);
  const activeCandidate = candidates.find((candidate) => candidate.id === selectedCandidate) ?? null;
  const generations = useMemo(() => {
    const grouped = new Map<number, Candidate[]>();
    for (const candidate of candidates) {
      grouped.set(candidate.generation, [...(grouped.get(candidate.generation) ?? []), candidate]);
    }
    return [...grouped.entries()].sort(([a], [b]) => a - b);
  }, [candidates]);
  const complete = candidates.filter((candidate) => candidate.status === "complete").length;
  const best = Math.max(0, ...candidates.map((candidate) => candidate.score ?? 0));
  const progress = evolution
    ? Math.min(100, (candidates.length / (evolution.spec.search.population * evolution.spec.search.generations)) * 100)
    : 0;

  async function launch(spec: typeof defaultSpec) {
    setError(null);
    try {
      const created = await request<Evolution>("/api/v1/evolutions", {
        method: "POST",
        body: JSON.stringify(spec),
      });
      await request(`/api/v1/evolutions/${created.id}/start`, { method: "POST" });
      setLaunchOpen(false);
      setSelectedId(created.id);
      await loadList();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Launch failed");
    }
  }

  async function action(name: "pause" | "resume") {
    if (!evolution) return;
    await request(`/api/v1/evolutions/${evolution.id}/${name}`, { method: "POST" });
    await loadEvolution(evolution.id);
  }

  async function promote() {
    if (!evolution || !activeCandidate) return;
    await request(`/api/v1/evolutions/${evolution.id}/promote`, {
      method: "POST",
      body: JSON.stringify({ candidate_id: activeCandidate.id }),
    });
    await loadEvolution(evolution.id);
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    if (!evolution || !message.trim()) return;
    setReply("Thinking…");
    try {
      const result = await request<{
        choices: Array<{ message: { content: string } }>;
      }>("/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({
          model: `neoform://evolutions/${evolution.id}/champion`,
          messages: [{ role: "user", content: message }],
        }),
      });
      setReply(result.choices[0]?.message.content ?? "No response");
    } catch (reason) {
      setReply(reason instanceof Error ? reason.message : "Inference failed");
    }
  }

  return (
    <main className="shell">
      <aside className="rail">
        <Brand />
        <nav aria-label="Primary navigation">
          <button className="nav-item active"><GitBranch size={17} /> Evolutions</button>
          <button className="nav-item"><Dna size={17} /> Datasets</button>
          <button className="nav-item"><Gauge size={17} /> Evaluations</button>
          <button className="nav-item"><Hexagon size={17} /> Checkpoints</button>
        </nav>
        <div className="rail-section">
          <span className="rail-label">Experiments</span>
          {evolutions.map((item) => (
            <button
              className={`experiment-link ${selectedId === item.id ? "selected" : ""}`}
              key={item.id}
              onClick={() => setSelectedId(item.id)}
            >
              <StatusDot status={item.status} />
              <span>{item.name}</span>
            </button>
          ))}
        </div>
        <div className="rail-footer">
          <div className={`connection connection-${connection}`}>
            <span /><div><strong>{connection === "ready" ? "Tinker connected" : connection === "checking" ? "Checking Tinker" : "Tinker setup needed"}</strong><small>{connection === "ready" ? "Live API" : "Run neoform doctor"}</small></div>
          </div>
          <button className="icon-button" aria-label="Settings"><Settings size={17} /></button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <button className="mobile-menu" aria-label="Open menu"><Menu size={18} /></button>
          <div className="crumbs"><span>Evolution lab</span><ChevronRight size={14} /><strong>{evolution?.name ?? "New program"}</strong></div>
          <div className="top-actions">
            <div className="command"><Command size={14} /><span>K</span></div>
            <button className="secondary-button"><TerminalSquare size={15} /> API</button>
            <button className="primary-button" onClick={() => setLaunchOpen(true)}><Plus size={16} /> New evolution</button>
          </div>
        </header>

        {error && <div className="error-banner"><ShieldCheck size={16} /><span>{error}</span><button onClick={() => setError(null)} aria-label="Dismiss"><X size={15} /></button></div>}

        {loading ? (
          <div className="center-state"><LoaderCircle className="spin" /><span>Opening the lab</span></div>
        ) : !evolution ? (
          <EmptyLab connection={connection} onLaunch={() => setLaunchOpen(true)} />
        ) : (
          <>
            <div className="program-header">
              <div>
                <div className="eyeline"><span className={`state state-${evolution.status}`}><StatusDot status={evolution.status} />{evolution.status}</span><span>{evolution.id}</span></div>
                <h1>{evolution.name}</h1>
                <p>{evolution.spec.base_model} · LoRA evolutionary search</p>
              </div>
              <div className="program-actions">
                {evolution.status === "running" ? (
                  <button className="secondary-button" onClick={() => void action("pause")}><Pause size={15} /> Pause</button>
                ) : evolution.status === "paused" ? (
                  <button className="secondary-button" onClick={() => void action("resume")}><Play size={15} /> Resume</button>
                ) : null}
                <button className="icon-button" aria-label="More actions"><MoreHorizontal size={18} /></button>
              </div>
            </div>

            <section className="metrics" aria-label="Evolution metrics">
              <Metric icon={<Trophy />} label="Best fitness" value={formatScore(best)} detail="Weighted benchmark score" tone="lime" />
              <Metric icon={<FlaskConical />} label="Candidates" value={`${complete}/${evolution.spec.search.population * evolution.spec.search.generations}`} detail={`${evolution.spec.search.generations} generations planned`} tone="violet" />
              <Metric icon={<CircleDollarSign />} label="Live spend" value={`$${evolution.spent.toFixed(3)}`} detail={`$${evolution.spec.max_usd.toFixed(0)} hard ceiling`} tone="amber" />
              <Metric icon={<Activity />} label="Program" value={`${Math.round(progress)}%`} detail={evolution.status === "running" ? "Training is active" : evolution.status} tone="cyan" />
            </section>

            <div className="lab-grid">
              <section className="lineage-panel panel">
                <div className="panel-head">
                  <div><span className="section-kicker">PHYLOGENETIC MAP</span><h2>Checkpoint lineage</h2></div>
                  <div className="legend"><span><i className="legend-live" /> Active</span><span><i className="legend-best" /> Survivor</span><span><i /> Complete</span></div>
                </div>
                <div className="lineage-scroll">
                  {generations.length === 0 ? (
                    <div className="lineage-wait"><LoaderCircle className="spin" size={20} /><span>Awaiting the first candidate</span></div>
                  ) : (
                    <div className="generation-grid" style={{ gridTemplateColumns: `repeat(${generations.length}, minmax(190px, 1fr))` }}>
                      {generations.map(([generation, items]) => (
                        <div className="generation" key={generation}>
                          <div className="generation-title"><span>G{generation.toString().padStart(2, "0")}</span><small>{items.filter((item) => item.status === "complete").length}/{items.length} viable</small></div>
                          <div className="candidate-stack">
                            {items.map((candidate) => {
                              const isChampion = evolution.champion_id === candidate.id;
                              const isSelected = selectedCandidate === candidate.id;
                              return (
                                <button
                                  key={candidate.id}
                                  className={`candidate-card ${isSelected ? "selected" : ""} ${isChampion ? "champion" : ""}`}
                                  onClick={() => setSelectedCandidate(candidate.id)}
                                >
                                  <span className="candidate-line" aria-hidden="true" />
                                  <div className="candidate-top"><StatusDot status={candidate.status} /><span>{shortId(candidate.id)}</span>{isChampion && <Trophy size={13} />}</div>
                                  <strong>{formatScore(candidate.score)}</strong>
                                  <div className="candidate-meta"><span>{candidate.genotype.loss_function.toUpperCase()}</span><span>LR {candidate.genotype.learning_rate.toExponential(1)}</span></div>
                                  {candidate.status === "training" && <div className="scan-line" />}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
              </section>

              <CandidateInspector
                candidate={activeCandidate}
                champion={evolution.champion_id === activeCandidate?.id}
                onPromote={() => void promote()}
              />
            </div>

            <section className="lower-grid">
              <BenchmarkPanel candidate={activeCandidate} benchmarks={evolution.spec.benchmarks} />
              <section className="notes-panel panel">
                <div className="panel-head compact"><div><span className="section-kicker">LAB NOTES</span><h2>Runtime intelligence</h2></div><Sparkles size={17} /></div>
                <div className="note"><span className="note-icon cyan"><Zap size={15} /></span><div><strong>Concurrency is inside bounds</strong><p>Two parallel candidates respect the configured Tinker capacity.</p></div></div>
                <div className="note"><span className="note-icon lime"><ShieldCheck size={15} /></span><div><strong>Budget reservation active</strong><p>No candidate starts unless its worst-case token cost fits.</p></div></div>
                <div className="note"><span className="note-icon violet"><BrainCircuit size={15} /></span><div><strong>Selection remains reproducible</strong><p>Mutation seed 2026 and benchmark weights are locked.</p></div></div>
              </section>
              <section className="playground panel">
                <div className="panel-head compact"><div><span className="section-kicker">CHAMPION ALIAS</span><h2>Inference probe</h2></div><span className="alias-status"><span />{evolution.champion_id ? "Ready" : "Awaiting promotion"}</span></div>
                <form onSubmit={(event) => void sendMessage(event)}>
                  <textarea value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Test the promoted model…" disabled={!evolution.champion_id} />
                  <button aria-label="Send prompt" disabled={!evolution.champion_id || !message.trim()}><Send size={15} /></button>
                </form>
                {reply && <div className="model-reply">{reply}</div>}
              </section>
            </section>
          </>
        )}
      </section>

      {launchOpen && <LaunchModal connection={connection} onClose={() => setLaunchOpen(false)} onLaunch={launch} />}
    </main>
  );
}

function Metric({ icon, label, value, detail, tone }: { icon: React.ReactNode; label: string; value: string; detail: string; tone: string }) {
  return <article className="metric"><span className={`metric-icon ${tone}`}>{icon}</span><div><small>{label}</small><strong>{value}</strong><p>{detail}</p></div></article>;
}

function EmptyLab({ connection, onLaunch }: { connection: string; onLaunch: () => void }) {
  return (
    <div className="empty-lab">
      <div className="empty-orbit"><span /><span /><span /><div><Dna size={32} /></div></div>
      <span className="section-kicker">OPEN-SOURCE POST-TRAINING</span>
      <h1>Evolution starts with a single checkpoint.</h1>
      <p>Branch LoRA candidates, measure real fitness, and promote only the model that earns it.</p>
      <button className="primary-button large" onClick={onLaunch} disabled={connection !== "ready"}><Rocket size={17} /> Configure first evolution <ArrowRight size={16} /></button>
      {connection !== "ready" && <small className="setup-hint">Set TINKER_API_KEY and run <code>neoform doctor</code> first.</small>}
    </div>
  );
}

function CandidateInspector({ candidate, champion, onPromote }: { candidate: Candidate | null; champion: boolean; onPromote: () => void }) {
  if (!candidate) return <aside className="inspector panel empty-inspector"><Beaker size={25} /><p>Select a candidate to inspect its genotype and evidence.</p></aside>;
  return (
    <aside className="inspector panel">
      <div className="inspector-head"><div><span className="section-kicker">SPECIMEN</span><h2>{shortId(candidate.id)}</h2></div><span className={`state state-${candidate.status}`}><StatusDot status={candidate.status} />{candidate.status}</span></div>
      <div className="fitness-ring" style={{ "--fitness": `${(candidate.score ?? 0) * 360}deg` } as React.CSSProperties}><div><strong>{formatScore(candidate.score)}</strong><small>FITNESS</small></div></div>
      <div className="genotype"><DataRow label="Loss" value={candidate.genotype.loss_function.toUpperCase()} /><DataRow label="Learning rate" value={candidate.genotype.learning_rate.toExponential(2)} /><DataRow label="Temperature" value={candidate.genotype.temperature.toFixed(1)} /><DataRow label="Max tokens" value={String(candidate.genotype.max_tokens ?? 64)} /><DataRow label="Optimizer" value={candidate.genotype.optimizer_mode} /></div>
      {candidate.error && <p className="candidate-error">{candidate.error}</p>}
      <button className="promote-button" disabled={candidate.status !== "complete" || champion} onClick={onPromote}>{champion ? <><Check size={15} /> Current champion</> : <><Trophy size={15} /> Promote checkpoint</>}</button>
    </aside>
  );
}

function DataRow({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function BenchmarkPanel({ candidate, benchmarks }: { candidate: Candidate | null; benchmarks: Evolution["spec"]["benchmarks"] }) {
  return (
    <section className="benchmark-panel panel">
      <div className="panel-head compact"><div><span className="section-kicker">EVALUATION VECTOR</span><h2>Benchmark fitness</h2></div><span className="mono">WEIGHTED</span></div>
      <div className="bars">
        {benchmarks.map((benchmark) => {
          const score = candidate?.metrics?.scores[benchmark.name] ?? 0;
          return <div className="bar-row" key={benchmark.name}><div><strong>{benchmark.name}</strong><span>{Math.round(benchmark.weight * 100)}% weight</span></div><div className="bar-track"><span style={{ width: `${score * 100}%` }} /></div><b>{formatScore(score || null)}</b></div>;
        })}
      </div>
    </section>
  );
}

function LaunchModal({ connection, onClose, onLaunch }: { connection: string; onClose: () => void; onLaunch: (spec: typeof defaultSpec) => Promise<void> }) {
  const [name, setName] = useState(defaultSpec.name);
  const [budget, setBudget] = useState(defaultSpec.max_usd);
  const [benchmark, setBenchmark] = useState<keyof typeof benchmarkPresets>("arithmetic_exact_match");
  const [submitting, setSubmitting] = useState(false);
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="launch-modal" role="dialog" aria-modal="true" aria-labelledby="launch-title">
        <button className="modal-close" onClick={onClose} aria-label="Close"><X size={17} /></button>
        <div className="modal-symbol"><Dna size={22} /></div>
        <span className="section-kicker">NEW EVOLUTION</span><h2 id="launch-title">Design the search space</h2><p>NEOFORM reserves worst-case cost before any request reaches Tinker.</p>
        <label>Program name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label>Base model<div className="select-like"><span>Qwen/Qwen3.5-4B</span><small>Hybrid · Vision · 4B</small></div></label>
        <label htmlFor="benchmark-picker">Benchmark<select id="benchmark-picker" value={benchmark} onChange={(event) => setBenchmark(event.target.value as keyof typeof benchmarkPresets)}>{Object.entries(benchmarkPresets).map(([key, preset]) => <option key={key} value={key}>{preset.label}</option>)}</select></label>
        <div className="field-grid"><label>Population<div className="static-field">4 candidates</div></label><label>Generations<div className="static-field">3 cycles</div></label></div>
        <label>Hard budget ceiling<div className="budget-input"><span>$</span><input type="number" min="1" max="100000" value={budget} onChange={(event) => setBudget(Number(event.target.value))} /></div></label>
        <div className="launch-summary"><div><ShieldCheck size={16} /><span>Manual champion promotion</span></div><div><CircleDollarSign size={16} /><span>Conservative token reservation</span></div></div>
        <button className="primary-button launch-button" disabled={submitting || connection !== "ready" || !name.trim()} onClick={async () => { setSubmitting(true); await onLaunch({ ...defaultSpec, name, max_usd: budget, benchmarks: [...benchmarkPresets[benchmark].benchmarks] } as typeof defaultSpec).finally(() => setSubmitting(false)); }}>{submitting ? <LoaderCircle className="spin" size={16} /> : <Rocket size={16} />} Launch live evolution</button>
      </section>
    </div>
  );
}
