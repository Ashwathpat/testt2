/* Design reminder: warm ivory instrument-panel UI, terracotta signal accents, compact telemetry, asymmetric three-zone dashboard. */
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Archive,
  ArrowUpRight,
  AudioLines,
  Bot,
  BrainCircuit,
  Check,
  ChevronDown,
  HelpCircle,
  Clock3,
  Database,
  Gauge,
  Layers3,
  Mic,
  Mic2,
  Minimize2,
  Paperclip,
  Pause,
  Play,
  RotateCcw,
  Send,
  ShieldCheck,
  Sparkles,
  Trash2,
  Waves,
  X,
  Zap,
} from "lucide-react";


const presets = [
  { icon: "✦", label: "How does Voice RAG work?", query: "Explain the Voice RAG pipeline in plain language." },
  { icon: "⌁", label: "Compare chunking strategies", query: "Compare fixed-size chunking with semantic paragraph chunking." },
  { icon: "◈", label: "Test a grounded answer", query: "What are the guardrails for an off-topic question?" },
  { icon: "↯", label: "Inspect latency", query: "Where does most of the latency come from in this pipeline?" },
];

const initialMessages = [
  { id: 1, query: "What does the grounding check protect against?", answer: "It verifies that the response stays anchored to retrieved source passages before synthesis. If the evidence is weak or off-topic, the assistant asks for clarification instead of inventing an answer.", time: "09:42", source: "Guardrails handbook · p. 12" },
  { id: 2, query: "How should I tune fixed-size chunks?", answer: "Start with 128-token chunks for low-latency voice turns, then add a small overlap when sentence boundaries are frequently split. Validate with recall and time-to-first-token together.", time: "09:37", source: "RAG tuning guide · p. 08" },
];

const metrics = [
  { icon: Mic2, label: "STT Audio", value: "18 ms", tone: "terracotta" },
  { icon: Database, label: "Vector Search (Qdrant)", value: "23 ms", tone: "blue" },
  { icon: ShieldCheck, label: "Grounding Check", value: "< 10 ms", tone: "amber" },
  { icon: BrainCircuit, label: "LLM Token Synthesis", value: "42 ms", tone: "teal" },
];

function Logo() {
  return <div className="brand-lockup"><div className="brand-mark"><Mic2 size={20} aria-hidden="true" /></div><div><strong>Voice RAG</strong><span>Assistant / Workbench</span></div></div>;
}

function Header({ onClear }) {
  return <header className="topbar"><Logo /><div className="topbar-center"><span className="live-dot" /> <span>WORKSPACE / GOA 2026</span><span className="slash">/</span><span className="muted">TASK 02</span></div><div className="topbar-actions"><div className="status-pill"><span className="pulse-dot" /> PIPELINE READY</div><button className="icon-button" aria-label="Help"><HelpCircle size={17} /></button><button className="avatar" aria-label="Profile">A</button></div></header>;
}

function HistoryRail({ messages, onClear }) {
  return <aside className="history-rail panel"><div className="panel-heading"><div><span className="eyebrow"><Archive size={13} /> SESSION LOG</span><h2>Chat history</h2></div><button className="text-button" onClick={onClear}>{messages.length ? "Clear" : "Reset"}</button></div>{messages.length === 0 ? <div className="empty-history"><div className="empty-glyph"><AudioLines size={24} /></div><p>No past voice queries yet.</p><span>Submit a query to start building your chat history.</span></div> : <div className="history-list">{messages.map((message) => <button className="history-item" key={message.id}><span className="history-time">{message.time}</span><strong>{message.query}</strong><span>{message.source}</span><ArrowUpRight size={14} /></button>)}</div>}<div className="rail-footer"><div className="rail-footer-icon"><Layers3 size={15} /></div><div><span>MEMORY INDEX</span><strong>2,481 source chunks</strong></div></div></aside>;
}

function LatencyScope() {
  const ticks = Array.from({ length: 72 }, (_, index) => index);
  return <div className="latency-scope" aria-label="Animated pipeline latency scope"><div className="scope-head"><span><span className="scope-live" /> LIVE TRACE / 01</span><b>93 ms TTFT</b></div><div className="scope-visual"><div className="scope-halo" /><svg className="scope-svg" viewBox="0 0 260 260" role="img" aria-label="Circular pipeline latency visualization"><circle className="scope-track" cx="130" cy="130" r="99" /><circle className="scope-ring ring-one" cx="130" cy="130" r="99" /><circle className="scope-ring ring-two" cx="130" cy="130" r="87" /><circle className="scope-ring ring-three" cx="130" cy="130" r="75" /><circle className="scope-arc arc-stt" cx="130" cy="130" r="99" /><circle className="scope-arc arc-search" cx="130" cy="130" r="99" /><circle className="scope-arc arc-ground" cx="130" cy="130" r="99" /><circle className="scope-arc arc-llm" cx="130" cy="130" r="99" />{ticks.map((tick) => <line key={tick} className="scope-tick" x1="130" y1="21" x2="130" y2={tick % 6 === 0 ? 27 : 24} transform={`rotate(${tick * 5} 130 130)`} />)}</svg><div className="scope-center"><span>TTFT</span><strong>93</strong><small>milliseconds</small><i><Zap size={11} /> STREAMING</i></div></div><div className="scope-legend"><span><i className="legend-dot stt" /> STT <b>18</b></span><span><i className="legend-dot search" /> SEARCH <b>23</b></span><span><i className="legend-dot ground" /> GROUND <b>10</b></span><span><i className="legend-dot llm" /> SYNTH <b>42</b></span></div><div className="scope-timeline"><div className="timeline-label"><span>PIPELINE CLOCK</span><b>0.093 s</b></div><div className="timeline-track"><span className="timeline-progress" /><span className="timeline-playhead" /></div><div className="timeline-events"><span>STT</span><span>QDRANT</span><span>GROUND</span><span>LLM</span></div></div></div>;
}

function MetricCard() {
  return <section className="panel telemetry-panel"><div className="panel-heading compact"><div><span className="eyebrow"><Activity size={13} /> LIVE OBSERVABILITY</span><h2>Pipeline latency</h2></div><span className="tiny-status">TTFT <b>93 ms</b></span></div><LatencyScope /><div className="metric-list">{metrics.map(({ icon: Icon, label, value, tone }) => <div className="metric-row" key={label}><div className={`metric-icon ${tone}`}><Icon size={15} /></div><span>{label}</span><b className={tone}>{value}</b></div>)}</div></section>;
}

function AnalyticsCard() {
  return <section className="panel analytics-panel"><div className="analytics-title"><Gauge size={15} /><span>Latency analytics <b>(fixed 128)</b></span><strong>Recall: 88.4%</strong></div><div className="analytics-grid"><div><span>P50 (MEDIAN)</span><b>42 <small>ms</small></b></div><div><span>P70</span><b>68 <small>ms</small></b></div><div><span>P100 (MAX)</span><b>110 <small>ms</small></b></div></div></section>;
}

function ExecutionCard() {
  const steps = ["Audio STT transcription", "Vector index search (Qdrant)", "Grounding validation check", "LLM response synthesis"];
  return <section className="panel execution-panel"><div className="panel-heading compact"><div><span className="eyebrow"><Waves size={13} /> TRACE</span><h2>Execution steps</h2></div><button className="icon-button" aria-label="Trace details"><Paperclip size={15} /></button></div><ol>{steps.map((step, index) => <li key={step}><span className={`step-dot ${index === 0 ? "active" : ""}`}>{index === 0 ? <Check size={11} /> : index + 1}</span><span>{step}</span>{index === 0 && <span className="step-live">LIVE</span>}</li>)}</ol></section>;
}

const latencyStages = [{ key: "stt", label: "Audio STT", detail: "voice → transcript", value: 18, color: "#f08b62" }, { key: "search", label: "Vector Search", detail: "Qdrant retrieval", value: 23, color: "#36c8b1" }, { key: "ground", label: "Grounding", detail: "evidence check", value: 10, color: "#f4bc45" }, { key: "llm", label: "Synthesis", detail: "spoken response", value: 42, color: "#8398ff" }];

function LatencySequence() {
  const sectionRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const [isPinned, setIsPinned] = useState(false);
  const traceProgress = Math.min(1, Math.max(0, (progress - 0.16) / 0.84));
  const activeStage = Math.min(3, Math.floor(traceProgress * 4.05));
  const activeStageData = latencyStages[activeStage];
  const slideProgress = traceProgress * traceProgress * (3 - 2 * traceProgress);
  useEffect(() => { const update = () => { const panel = sectionRef.current; const scene = panel?.parentElement; if (!panel || !scene) return; const start = scene.offsetTop; const travel = Math.max(1, scene.offsetHeight - window.innerHeight); const raw = (window.scrollY - start) / travel; setProgress(Math.min(1, Math.max(0, raw))); setIsPinned(raw >= -0.12); }; update(); window.addEventListener("scroll", update, { passive: true }); window.addEventListener("resize", update); return () => { window.removeEventListener("scroll", update); window.removeEventListener("resize", update); }; }, []);
  return <section ref={sectionRef} className={`latency-sequence ${isPinned ? "is-pinned" : ""}`} style={{ "--slide-progress": slideProgress }}><div className="sequence-sticky"><div className="sequence-topline"><span>TRACE VISUALIZER / 02</span><b>FIXED-128 <i /> 93 MS TOTAL</b></div><div className="sequence-scroll-cue"><span>SCROLL</span><b>04 STAGES</b><i /></div><div className="sequence-intro"><span className="eyebrow"><Activity size={13} /> SCROLL TO DECOMPOSE LATENCY</span><h2>See where the<br /><em>milliseconds go.</em></h2><p>Scroll through the trace. Each stage pulls away from the core so you can see its individual contribution to the response.</p><div className="sequence-progress"><span><b>{Math.round(traceProgress * 100)}%</b> TRACE EXPOSURE</span><i><b style={{ width: `${Math.max(4, traceProgress * 100)}%` }} /></i></div><div className="active-stage-card" style={{ "--active-color": activeStageData.color }}><span>NOW INSPECTING / 0{activeStage + 1}</span><strong>{activeStageData.label}</strong><b>{activeStageData.value} <small>ms contribution</small></b><em>{activeStageData.detail}</em></div></div><div className="exploded-stage" style={{ "--sequence-progress": progress }}><div className="exploded-core"><div className="core-grid" /><span>VOICE RAG</span><strong>{Math.round(93 - traceProgress * 8)}<small>ms</small></strong><em>{traceProgress > .88 ? "TRACE COMPLETE" : `STAGE 0${activeStage + 1} / 04`}</em></div><div className="trace-crosshair" />{latencyStages.map((stage, index) => { const stageReveal = Math.min(1, Math.max(0, traceProgress * 4 - index * .74)); return <div className={`exploded-part ${stage.key} ${index === activeStage ? "is-active" : ""}`} key={stage.key} style={{ "--stage-color": stage.color, "--stage-index": index, "--stage-progress": stageReveal }}><div className="part-node" /><div className="part-line" /><div className="part-copy"><span>{String(index + 1).padStart(2, "0")} / {stage.detail}</span><strong>{stage.label}</strong><b>{stage.value} <small>ms</small></b></div></div>; })}<div className="exploded-axis"><span>0 ms</span><i /><span>93 ms · total response</span></div></div></div></section>;
}

function QueryStage({ onSubmit, isListening, setIsListening }) {
  const [query, setQuery] = useState("");
  const [selectedStrategy, setSelectedStrategy] = useState("fixed-128");
  const [isRunning, setIsRunning] = useState(false);
  const selectedPreset = useMemo(() => presets.find((preset) => preset.query === query)?.label, [query]);

  const ask = () => {
    if (!query.trim()) return;
    setIsRunning(true);
    window.setTimeout(() => { onSubmit(query.trim()); setQuery(""); setIsRunning(false); }, 650);
  };
  const toggleListening = () => { setIsListening(!isListening); };

  return <main className="query-stage"><div className="stage-kicker"><span className="signal-line" /> VOICE RAG WORKBENCH <span className="stage-id">/ 02</span></div><div className="hero-copy"><h1>Ask the<br /><span>knowledge layer.</span></h1><p>Low-latency answers grounded in your own documents, with every retrieval step visible.</p></div><section className="query-panel panel"><div className="query-panel-header"><div><span className="eyebrow"><Sparkles size={13} /> CHUNKING STRATEGY</span><h2>Choose your retrieval profile</h2></div><span className="recall-badge">Recall@5 <b>88.4%</b></span></div><div className="strategy-tabs" role="tablist">{[["fixed-128", "Fixed-128", "Low latency"], ["fixed-256", "Fixed-256", "Balanced"], ["semantic", "Semantic", "Paragraphs"], ["window", "Window", "Sentence"]].map(([value, label, detail]) => <button key={value} className={selectedStrategy === value ? "selected" : ""} onClick={() => setSelectedStrategy(value)} role="tab" aria-selected={selectedStrategy === value}><strong>{label}</strong><span>{detail}</span></button>)}</div><div className="strategy-note"><Zap size={14} /><strong>Lowest latency</strong><span> P50: 42 ms · Minimal RAM footprint</span><AlertTriangle size={14} /><span className="warning-copy">May split long sentences across chunk boundaries</span></div><div className="preset-label">TRY A PRESET QUERY</div><div className="preset-row">{presets.map((preset) => <button className={selectedPreset === preset.label ? "preset active" : "preset"} key={preset.label} onClick={() => setQuery(preset.query)}><span>{preset.icon}</span>{preset.label}</button>)}</div><div className={`composer ${isListening ? "listening" : ""}`}><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) ask(); }} placeholder="Ask a question about Voice RAG, your source library, or low-latency retrieval…" aria-label="Question for the assistant" /><div className="composer-footer"><button className={`mic-button ${isListening ? "active" : ""}`} onClick={toggleListening} aria-label={isListening ? "Stop listening" : "Start listening"}><span className="mic-orb">{isListening ? <Pause size={20} /> : <Mic2 size={20} />}</span><span>{isListening ? "Listening…" : "Click mic to speak"}</span></button><span className="keyboard-hint">⌘ ↵ to ask</span><button className="ask-button" onClick={ask} disabled={isRunning}>{isRunning ? <><RotateCcw className="spin" size={15} /> Running pipeline</> : <>Ask assistant <Send size={15} /></>}</button></div></div></section><div className="architecture-strip"><div className="architecture-title"><div className="orbit-icon"><BrainCircuit size={17} /></div><div><span className="eyebrow">SYSTEM ARCHITECTURE</span><strong>Retrieval → grounding → synthesis</strong></div></div><div className="architecture-visual"><Waves size={20} /><span>RETRIEVAL MAP</span></div><button className="text-button">Inspect spec <ArrowUpRight size={14} /></button></div></main>;
}

function AnswerDock({ answer, isOpen, setIsOpen }) {
  const hasAnswer = Boolean(answer);
  return <div className={`answer-dock ${isOpen ? "is-expanded" : ""}`}>
    {!isOpen ? <button className="answer-tab" onClick={() => setIsOpen(true)} aria-label="Open answers">
      <span className="answer-tab-orb"><Bot size={18} /></span><span>ANSWERS</span><b>{hasAnswer ? "1" : "—"}</b>
    </button> : <section className="answer-sheet" role="dialog" aria-label="Grounded answer" aria-modal="false">
      <div className="answer-sheet-topline"><span><span className={hasAnswer ? "pulse-dot" : "live-dot"} /> {hasAnswer ? "ANSWER READY / GROUNDED RESPONSE" : "ANSWERS / READY FOR YOUR QUESTION"}</span><button className="answer-minimize" onClick={() => setIsOpen(false)}><Minimize2 size={17} /> Minimize</button></div>
      <div className="answer-sheet-grid"><div className="answer-sheet-copy"><span className="eyebrow"><Sparkles size={14} /> {hasAnswer ? "SYNTHESIS COMPLETE" : "ANSWER WORKSPACE"}</span><h2>{answer?.query ?? "Your grounded answer will appear here."}</h2><p>{answer?.answer ?? "Ask a question from the workbench, then open this sheet whenever you want to inspect the response, sources, and latency details."}</p><div className="answer-sheet-meta"><div><span>SOURCE COVERAGE</span><b>{hasAnswer ? "3 grounded passages" : "Waiting for retrieval"}</b></div><div><span>RESPONSE MODE</span><b>{hasAnswer ? "Voice-ready synthesis" : "Ready for synthesis"}</b></div><div><span>CONFIDENCE</span><b>{hasAnswer ? "High · 0.92" : "—"}</b></div></div></div><aside className="answer-sheet-aside"><div className="answer-source-card"><span>RETRIEVED FROM</span><strong>{answer?.source ?? "No source selected yet"}</strong><p>{hasAnswer ? "Evidence was verified before the response was synthesized." : "Source details will appear here as soon as a grounded answer is ready."}</p></div><div className="answer-latency-card"><span>END-TO-END LATENCY</span><b>{hasAnswer ? "93" : "—"} <small>ms</small></b><em><Zap size={13} /> {hasAnswer ? "Streaming answer" : "Waiting for query"}</em></div></aside></div>
      <div className="answer-sheet-footer"><span>{hasAnswer ? "Press minimize to return to the workbench." : "Ask any question to populate this answer sheet."}</span><button disabled={!hasAnswer}><Paperclip size={15} /> Save response</button></div>
    </section>}
  </div>;
}

export default function Home() {
  const [messages, setMessages] = useState(initialMessages);
  const [isListening, setIsListening] = useState(false);
  const [latestAnswer, setLatestAnswer] = useState(null);
  const [answersOpen, setAnswersOpen] = useState(false);
  const submitQuery = (query) => { const response = { id: Date.now(), query, answer: "The retrieval layer found a grounded answer across 3 source passages. The assistant would now synthesize a concise spoken response with citations and a confidence check.", time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), source: "Voice RAG index · 3 passages" }; setMessages((current) => [response, ...current]); setLatestAnswer(response); setAnswersOpen(true); };
  return <div className="slide-deck"><section className="slide-scene slide-one"><div className="app-shell"><Header onClear={() => setMessages([])} /><div className="dashboard-layout"><HistoryRail messages={messages} onClear={() => setMessages([])} /><QueryStage onSubmit={submitQuery} isListening={isListening} setIsListening={setIsListening} /><aside className="telemetry-rail"><MetricCard /><AnalyticsCard /><ExecutionCard /><section className="panel cache-panel"><div><span className="eyebrow"><Archive size={13} /> CACHE</span><strong>Warm retrieval cache</strong></div><button className="clear-cache"><Trash2 size={13} /> Clear</button></section></aside></div><footer className="bottom-status"><span><span className="live-dot" /> Frontend demo environment</span><span>Fixed-128 profile · Qdrant-ready</span><span>⌘ K <span className="muted">shortcuts</span></span></footer></div></section><section className="slide-scene slide-two"><LatencySequence /></section><AnswerDock answer={latestAnswer} isOpen={answersOpen} setIsOpen={setAnswersOpen} /></div>;
}
