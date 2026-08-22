import { useEffect, useMemo, useRef, useState } from "react";
import { clearCache, processQueryStream, transcribeAudio } from "./services/api";
import {
  Activity, AlertTriangle, Archive, ArrowUpRight, AudioLines, Bot, BrainCircuit,
  Check, Database, Gauge, HelpCircle, Layers3, Mic2, Minimize2, Paperclip,
  Pause, RotateCcw, Send, ShieldCheck, Sparkles, Trash2, Waves, Zap,
} from "lucide-react";

const presets = [
  { icon: "✦", label: "How does Voice RAG work?", query: "Explain the Voice RAG pipeline in plain language." },
  { icon: "⌁", label: "Compare chunking strategies", query: "Compare fixed-size chunking with semantic paragraph chunking." },
  { icon: "◈", label: "Test a grounded answer", query: "What are the guardrails for an off-topic question?" },
  { icon: "↯", label: "Inspect latency", query: "Where does most of the latency come from in this pipeline?" },
];

const initialMessages = [
  { id: 1, query: "What does the grounding check protect against?", answer: "It verifies that the response stays anchored to retrieved source passages before synthesis.", time: "09:42", source: "Guardrails handbook · p. 12" },
  { id: 2, query: "How should I tune fixed-size chunks?", answer: "Start with 128-token chunks for low-latency voice turns, then validate recall and time-to-first-token together.", time: "09:37", source: "RAG tuning guide · p. 08" },
];

const emptyMetrics = {
  sttLatencyMs: null, retrievalLatencyMs: null, groundingLatencyMs: null,
  generationLatencyMs: null, totalLatencyMs: null, serverTotalMs: null,
  streamRoundTripLatencyMs: null, endToEndLatencyMs: null, ttftMs: null,
  evaluationLatencyMs: null, retrievedDocsCount: null, retrievalMethod: null, evaluation: null,
};

const numberOrNull = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : null;
const formatMs = (value) => numberOrNull(value) !== null ? `${Math.round(numberOrNull(value))} ms` : "—";
const metricValue = (value) => numberOrNull(value) !== null ? Math.round(numberOrNull(value)) : "—";
const percentile = (values, percentage) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil((percentage / 100) * sorted.length) - 1)];
};

function Logo() {
  return <div className="brand-lockup"><div className="brand-mark"><Mic2 size={20} aria-hidden="true" /></div><div><strong>Voice RAG</strong><span>Assistant / Workbench</span></div></div>;
}

function Header() {
  return <header className="topbar"><Logo /><div className="topbar-center"><span className="live-dot" /> <span>WORKSPACE / GOA 2026</span><span className="slash">/</span><span className="muted">TASK 02</span></div><div className="topbar-actions"><div className="status-pill"><span className="pulse-dot" /> PIPELINE READY</div><button className="icon-button" aria-label="Help"><HelpCircle size={17} /></button><button className="avatar" aria-label="Profile">A</button></div></header>;
}

function HistoryRail({ messages, onClear, guardrail, transcript, sttStatus }) {
  return <aside className="history-rail panel"><div className="panel-heading"><div><span className="eyebrow"><Archive size={13} /> SESSION LOG</span><h2>Chat history</h2></div><button className="text-button" onClick={onClear}>{messages.length ? "Clear" : "Reset"}</button></div>{messages.length === 0 ? <div className="empty-history"><div className="empty-glyph"><AudioLines size={24} /></div><p>No past voice queries yet.</p><span>Submit a query to start building your chat history.</span></div> : <div className="history-list">{messages.map((message) => <button className="history-item" key={message.id}><span className="history-time">{message.time}</span><strong>{message.query}</strong><span>{message.source}</span><ArrowUpRight size={14} /></button>)}</div>}<div className="rail-footer"><div className="rail-footer-icon"><Layers3 size={15} /></div><div><span>MEMORY INDEX</span><strong>{messages.length ? "Live response log" : "Awaiting query"}</strong></div></div><TaskReadiness compact guardrail={guardrail} transcript={transcript} sttStatus={sttStatus} /></aside>;
}

function LiveExecutionPanel({ metrics, processing, phase, guardrail }) {
  const endToEnd = metrics.endToEndLatencyMs ?? metrics.streamRoundTripLatencyMs ?? metrics.serverTotalMs ?? metrics.totalLatencyMs;
  const steps = [
    { label: "Audio STT transcription", value: metrics.sttLatencyMs, icon: Mic2, tone: "terracotta", phase: "transcribing" },
    { label: "Vector index search", value: metrics.retrievalLatencyMs, icon: Database, tone: "blue", phase: "retrieving" },
    { label: "Grounding validation", value: metrics.groundingLatencyMs, icon: ShieldCheck, tone: "amber", phase: "grounding" },
    { label: "LLM response synthesis", value: metrics.generationLatencyMs, icon: BrainCircuit, tone: "teal", phase: "generating" },
  ];
  return <section className="panel telemetry-panel"><div className="panel-heading compact"><div><span className="eyebrow"><Waves size={13} /> LIVE BACKEND TRACE</span><h2>Execution steps</h2></div><span className="tiny-status">END-TO-END <b>{formatMs(endToEnd)}</b></span></div><ol className="live-execution-list">{steps.map((step) => { const Icon = step.icon; const active = processing && phase === step.phase; const complete = !processing && metricValue(step.value) !== "—"; return <li key={step.phase} className={active ? "active" : complete ? "completed" : ""}><span className={`live-step-icon ${step.tone}`}>{complete ? <Check size={12} /> : <Icon size={13} />}</span><span>{step.label}</span><b>{active ? "LIVE" : formatMs(step.value)}</b></li>; })}</ol><div className="trace-timing-grid"><div><span>TTFT</span><b>{formatMs(metrics.ttftMs)}</b></div><div><span>BACKEND</span><b>{formatMs(metrics.serverTotalMs)}</b></div><div><span>ROUND TRIP</span><b>{formatMs(metrics.streamRoundTripLatencyMs)}</b></div><div><span>EVAL</span><b>{formatMs(metrics.evaluationLatencyMs)}</b></div></div><div className="live-trace-footer"><span>METHOD</span><b>{metrics.retrievalMethod || "Awaiting request"}</b><span>DOCS</span><b>{metrics.retrievedDocsCount ?? "—"}</b></div><div className={`guardrail-state ${guardrail?.grounded ? "pass" : guardrail?.status ? "review" : "idle"}`}><ShieldCheck size={13} /><span>{guardrail?.grounded ? "Grounded answer verified" : guardrail?.reason || "Guardrail status will appear after a request"}</span></div></section>;
}

function LiveAnalytics({ metrics, latencyHistory }) {
  const p50 = percentile(latencyHistory, 50);
  const p70 = percentile(latencyHistory, 70);
  const p100 = percentile(latencyHistory, 100);
  const confidence = metrics.evaluation?.overall_quality || metrics.evaluation?.retrieval?.context_relevance;
  const endToEnd = metrics.endToEndLatencyMs ?? metrics.streamRoundTripLatencyMs ?? metrics.serverTotalMs ?? metrics.totalLatencyMs;
  return <section className="panel analytics-panel"><div className="analytics-title"><Gauge size={15} /><span>Latency analytics <b>({latencyHistory.length} live runs)</b></span><strong>{confidence ? `Quality: ${(confidence * 100).toFixed(1)}%` : "Awaiting trace"}</strong></div><div className="analytics-grid"><div><span>P50</span><b>{metricValue(p50)} <small>ms</small></b></div><div><span>P70</span><b>{metricValue(p70)} <small>ms</small></b></div><div><span>P100</span><b>{metricValue(p100)} <small>ms</small></b></div></div><div className="latency-target"><span>TARGET</span><b>&lt; 200 ms</b><em>{numberOrNull(endToEnd) ? `${formatMs(endToEnd)} latest` : "Awaiting backend response"}</em></div></section>;
}

function TaskReadiness({ guardrail, transcript, sttStatus, compact = false }) {
  const transcriptSummary = transcript
    ? `${transcript.slice(0, 34)}${transcript.length > 34 ? "…" : ""}`
    : "Speak, then tap the mic again";

  return <section className={compact ? "task-readiness history-task-readiness" : "panel task-readiness"}><div className="task-readiness-head"><div><span className="eyebrow"><Sparkles size={13} /> TASK 02 READINESS</span><h2>Voice RAG requirements</h2></div><span className="task-status">LIVE</span></div><div className="readiness-grid"><div><span>DATASET</span><b>MSMARCO-XI</b><small>AI4Bharat multilingual corpus</small></div><div><span>STT</span><b>Sarvam proxy</b><small>{sttStatus || "Ready for microphone input"}</small></div><div><span>CHUNKING</span><b>4 strategies</b><small>Fixed, semantic, windowed</small></div><div><span>HARNESS</span><b>SSE orchestration</b><small>Error-aware streaming flow</small></div><div><span>GUARDRAIL</span><b>{guardrail?.grounded ? "Grounded" : "Armed"}</b><small>{guardrail?.reason || "Off-topic and evidence checks"}</small></div><div><span>TRANSCRIPT</span><b>{transcript ? "Captured" : "Waiting"}</b><small>{transcriptSummary}</small></div></div></section>;
}

function TranscriptBand({ transcript, sttStatus, metrics }) {
  return <section className="panel transcript-band"><div><span className="eyebrow"><AudioLines size={13} /> LIVE SPEECH-TO-TEXT</span><b>{sttStatus || "Mic ready"}</b></div><p>{transcript || "Your transcribed voice question will appear here after you stop recording."}</p><span>{formatMs(metrics.sttLatencyMs)}</span></section>;
}

function LatencySequence({ metrics }) {
  const sectionRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const [isPinned, setIsPinned] = useState(false);
  const total = metricValue(metrics.endToEndLatencyMs ?? metrics.streamRoundTripLatencyMs ?? metrics.serverTotalMs ?? metrics.totalLatencyMs);
  const stages = [
    { key: "stt", label: "Audio STT", detail: "voice → transcript", value: metricValue(metrics.sttLatencyMs), color: "#f08b62" },
    { key: "search", label: "Vector Search", detail: metrics.retrievalMethod || "backend retrieval", value: metricValue(metrics.retrievalLatencyMs), color: "#36c8b1" },
    { key: "ground", label: "Grounding", detail: "evidence check", value: metricValue(metrics.groundingLatencyMs), color: "#f4bc45" },
    { key: "llm", label: "Synthesis", detail: "streamed response", value: metricValue(metrics.generationLatencyMs), color: "#8398ff" },
  ];
  const traceProgress = Math.min(1, Math.max(0, (progress - 0.16) / 0.84));
  const activeStage = Math.min(3, Math.floor(traceProgress * 4.05));
  const activeStageData = stages[activeStage];
  const traceWidth = `${Math.max(4, traceProgress * 100)}%`;

  useEffect(() => {
    const update = () => {
      const panel = sectionRef.current;
      const scene = panel?.parentElement;
      if (!panel || !scene) return;
      const start = scene.offsetTop;
      const travel = Math.max(1, scene.offsetHeight - window.innerHeight);
      const raw = (window.scrollY - start) / travel;
      setProgress(Math.min(1, Math.max(0, raw)));
      setIsPinned(raw >= -0.12);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);

  return (
    <section ref={sectionRef} className={isPinned ? "latency-sequence is-pinned" : "latency-sequence"}>
      <div className="sequence-sticky">
        <div className="sequence-topline">
          <span>TRACE VISUALIZER / LIVE</span>
          <b>{metrics.retrievalMethod || "AWAITING BACKEND"} <i /> {total} TOTAL</b>
        </div>
        <div className="sequence-scroll-cue"><span>SCROLL</span><b>04 STAGES</b><i /></div>
        <div className="sequence-intro">
          <span className="eyebrow"><Activity size={13} /> LIVE LATENCY DECOMPOSITION</span>
          <h2>See where the<br /><em>milliseconds go.</em></h2>
          <p>Each value comes from the latest backend trace. Scroll to inspect how the pipeline separates.</p>
          <div className="sequence-progress"><span><b>{Math.round(traceProgress * 100)}%</b> TRACE EXPOSURE</span><i><b style={{ width: traceWidth }} /></i></div>
          <div className="active-stage-card" style={{ "--active-color": activeStageData.color }}>
            <span>NOW INSPECTING / 0{activeStage + 1}</span><strong>{activeStageData.label}</strong><b>{activeStageData.value} <small>ms contribution</small></b><em>{activeStageData.detail}</em>
          </div>
        </div>
        <div className="exploded-stage">
          <div className="exploded-core"><div className="core-grid" /><span>VOICE RAG</span><strong>{total}<small> ms</small></strong><em>{total === "—" ? "AWAITING BACKEND" : traceProgress > .88 ? "TRACE COMPLETE" : `STAGE 0${activeStage + 1} / 04`}</em></div>
          <div className="trace-crosshair" />
          {stages.map((stage, index) => {
            const stageReveal = Math.min(1, Math.max(0, traceProgress * 4 - index * .74));
            const classes = `exploded-part ${stage.key} ${index === activeStage ? "is-active" : ""}`;
            return <div className={classes} key={stage.key} style={{ "--stage-color": stage.color, "--stage-index": index, "--stage-progress": stageReveal }}><div className="part-node" /><div className="part-line" /><div className="part-copy"><span>{String(index + 1).padStart(2, "0")} / {stage.detail}</span><strong>{stage.label}</strong><b>{stage.value} <small>ms</small></b></div></div>;
          })}
          <div className="exploded-axis"><span>0 ms</span><i /><span>{total} · live response</span></div>
        </div>
      </div>
    </section>
  );
}

function QueryStage({ onSubmit, onMicToggle, isListening, isProcessing, error, strategy, setStrategy }) {
  const [query, setQuery] = useState("");
  const selectedPreset = useMemo(() => presets.find((preset) => preset.query === query)?.label, [query]);
  const ask = () => { if (query.trim() && !isProcessing) { onSubmit(query.trim(), strategy); setQuery(""); } };
  return <main className="query-stage"><div className="stage-kicker"><span className="signal-line" /> VOICE RAG WORKBENCH <span className="stage-id">/ LIVE</span></div><div className="hero-copy"><h1>Ask the<br /><span>knowledge layer.</span></h1><p>Speak or type a question, then inspect the live retrieval trace and grounded response.</p></div><section className="query-panel panel"><div className="query-panel-header"><div><span className="eyebrow"><Sparkles size={13} /> CHUNKING STRATEGY</span><h2>Choose your retrieval profile</h2></div><span className="recall-badge">Backend <b>connected</b></span></div><div className="strategy-tabs" role="tablist">{[["fixed_128", "Fixed-128", "Low latency"], ["fixed_256", "Fixed-256", "Balanced"], ["semantic", "Semantic", "Paragraphs"], ["sentence_window", "Window", "Sentence"]].map(([value, label, detail]) => <button key={value} className={strategy === value ? "selected" : ""} onClick={() => setStrategy(value)} role="tab" aria-selected={strategy === value}><strong>{label}</strong><span>{detail}</span></button>)}</div><div className="strategy-note"><Zap size={14} /><strong>Live pipeline</strong><span> Backend response timings populate the dashboard.</span><AlertTriangle size={14} /><span className="warning-copy">Values appear after the first request.</span></div><div className="preset-label">TRY A PRESET QUERY</div><div className="preset-row">{presets.map((preset) => <button className={selectedPreset === preset.label ? "preset active" : "preset"} key={preset.label} onClick={() => setQuery(preset.query)}><span>{preset.icon}</span>{preset.label}</button>)}</div><div className={`composer ${isListening ? "listening" : ""}`}><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) ask(); }} placeholder="Ask a question about Voice RAG, your source library, or low-latency retrieval…" aria-label="Question for the assistant" /><div className="composer-footer"><button className={`mic-button ${isListening ? "active" : ""}`} onClick={() => onMicToggle(strategy)} disabled={isProcessing && !isListening} aria-label={isListening ? "Stop recording and transcribe" : "Start microphone recording"}><span className="mic-orb">{isListening ? <Pause size={20} /> : <Mic2 size={20} />}</span><span>{isListening ? "Tap to stop & transcribe" : "Click mic to speak"}</span></button><span className="keyboard-hint">⌘ ↵ to ask</span><button className="ask-button" onClick={ask} disabled={isProcessing}>{isProcessing ? <><RotateCcw className="spin" size={15} /> Running pipeline</> : <>Ask assistant <Send size={15} /></>}</button></div>{error && <div className="error-banner">{error}</div>}</div></section><div className="architecture-strip"><div className="architecture-title"><div className="orbit-icon"><BrainCircuit size={17} /></div><div><span className="eyebrow">SYSTEM ARCHITECTURE</span><strong>STT → retrieval → grounding → synthesis</strong></div></div><div className="architecture-visual"><Waves size={20} /><span>LIVE BACKEND</span></div><button className="text-button">Inspect spec <ArrowUpRight size={14} /></button></div></main>;
}

function AnswerDock({ answer, metrics, isOpen, setIsOpen }) {
  const hasAnswer = Boolean(answer?.answer);
  return <div className={`answer-dock ${isOpen ? "is-expanded" : ""}`}>{!isOpen ? <button className="answer-tab" onClick={() => setIsOpen(true)} aria-label="Open answers"><span className="answer-tab-orb"><Bot size={18} /></span><span>ANSWERS</span><b>{hasAnswer ? "1" : "—"}</b></button> : <section className="answer-sheet" role="dialog" aria-label="Grounded answer" aria-modal="false"><div className="answer-sheet-topline"><span><span className={hasAnswer ? "pulse-dot" : "live-dot"} /> {hasAnswer ? "ANSWER READY / LIVE BACKEND" : "ANSWERS / READY FOR YOUR QUESTION"}</span><button className="answer-minimize" onClick={() => setIsOpen(false)}><Minimize2 size={17} /> Minimize</button></div><div className="answer-sheet-grid"><div className="answer-sheet-copy"><span className="eyebrow"><Sparkles size={14} /> {hasAnswer ? "SYNTHESIS COMPLETE" : "ANSWER WORKSPACE"}</span><h2>{answer?.query || "Your grounded answer will appear here."}</h2><p>{answer?.answer || "Ask a question from the workbench, then open this sheet whenever you want to inspect the response, sources, and live backend timings."}</p><div className="answer-sheet-meta"><div><span>SOURCE COVERAGE</span><b>{metrics.retrievedDocsCount ?? "Waiting"} passages</b></div><div><span>RESPONSE MODE</span><b>{hasAnswer ? "Backend synthesis" : "Ready for synthesis"}</b></div><div><span>CONFIDENCE</span><b>{metrics.evaluation?.overall_quality ? `${(metrics.evaluation.overall_quality * 100).toFixed(1)}%` : "—"}</b></div></div></div><aside className="answer-sheet-aside"><div className="answer-source-card"><span>RETRIEVED FROM</span><strong>{answer?.source || "No source selected yet"}</strong><p>{hasAnswer ? "Sources and timing were returned by the RAG backend." : "Source details will appear here as soon as a grounded answer is ready."}</p></div><div className="answer-latency-card"><span>END-TO-END LATENCY</span><b>{metricValue(metrics.endToEndLatencyMs ?? metrics.streamRoundTripLatencyMs ?? metrics.serverTotalMs ?? metrics.totalLatencyMs)} <small>ms</small></b><em><Zap size={13} /> {hasAnswer ? "Live browser-to-answer timing" : "Waiting for query"}</em></div></aside></div><div className="answer-sheet-footer"><span>{hasAnswer ? "Press minimize to return to the workbench." : "Ask any question to populate this answer sheet."}</span><button disabled={!hasAnswer}><Paperclip size={15} /> Save response</button></div></section>}</div>;
}

export default function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pipelinePhase, setPipelinePhase] = useState("");
  const [error, setError] = useState("");
  const [metrics, setMetrics] = useState(emptyMetrics);
  const [latencyHistory, setLatencyHistory] = useState([]);
  const [guardrail, setGuardrail] = useState(null);
  const [latestAnswer, setLatestAnswer] = useState(null);
  const [answersOpen, setAnswersOpen] = useState(false);
  const [strategy, setStrategy] = useState("fixed_128");
  const [transcript, setTranscript] = useState("");
  const [sttStatus, setSttStatus] = useState("");
  const recorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => () => { recorderRef.current?.stream?.getTracks().forEach((track) => track.stop()); }, []);

  const runQuery = async (query, selectedStrategy, sttLatencyMs = 0) => {
    if (!query.trim()) return;
    setError(""); setIsProcessing(true); setPipelinePhase("retrieving");
    setLatestAnswer({ query, answer: "", source: "Receiving live backend response…" });
    setMetrics((current) => ({ ...current, sttLatencyMs: numberOrNull(sttLatencyMs), retrievalLatencyMs: null, groundingLatencyMs: null, generationLatencyMs: null, totalLatencyMs: null, serverTotalMs: null, streamRoundTripLatencyMs: null, endToEndLatencyMs: null, ttftMs: null, evaluationLatencyMs: null }));
    try {
      const result = await processQueryStream(query, sttLatencyMs, (answer) => setLatestAnswer((current) => ({ ...current, answer })), setPipelinePhase, (metadata) => setMetrics((current) => ({ ...current, ...metadata })), selectedStrategy);
      const liveMetrics = { ...result.metrics, sttLatencyMs: numberOrNull(result.metrics.sttLatencyMs) || numberOrNull(sttLatencyMs) };
      setMetrics((current) => ({ ...current, ...liveMetrics }));
      const total = numberOrNull(liveMetrics.endToEndLatencyMs ?? liveMetrics.streamRoundTripLatencyMs ?? liveMetrics.serverTotalMs ?? liveMetrics.totalLatencyMs);
      if (total) setLatencyHistory((current) => [...current, total].slice(-40));
      setGuardrail({ grounded: result.grounded || Boolean(result.sources?.length), status: result.status, reason: result.reason });
      const complete = { id: Date.now(), query, answer: result.answer, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), source: result.sources?.[0]?.title || "Live RAG backend" };
      setLatestAnswer(complete); setMessages((current) => [complete, ...current]); setAnswersOpen(true);
    } catch (requestError) { setError(requestError.message || "The backend request could not be completed."); }
    finally { setIsProcessing(false); setPipelinePhase(""); }
  };

  const toggleMicrophone = async (selectedStrategy) => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state === "recording") { setSttStatus("Transcribing…"); setIsListening(false); recorder.stop(); return; }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { setError("Microphone recording is not supported by this browser."); return; }
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { echoCancellation: true, noiseSuppression: true } 
      });
      audioChunksRef.current = [];
      const rawMimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((type) => MediaRecorder.isTypeSupported(type));
      const cleanMimeType = rawMimeType ? rawMimeType.split(";")[0] : "audio/webm";
      const nextRecorder = rawMimeType ? new MediaRecorder(stream, { mimeType: rawMimeType }) : new MediaRecorder(stream);
      nextRecorder.ondataavailable = (event) => { if (event.data.size > 0) audioChunksRef.current.push(event.data); };
      nextRecorder.onerror = () => { setError("The microphone recording could not be completed."); setIsListening(false); setSttStatus(""); stream.getTracks().forEach((track) => track.stop()); };
      nextRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: cleanMimeType });
        if (!audioBlob.size) { setError("No audio was captured. Please try speaking again."); setSttStatus(""); return; }
        setIsProcessing(true); setPipelinePhase("transcribing");
        try {
          const stt = await transcribeAudio(audioBlob);
          const spokenQuery = stt.transcript?.trim();
          if (!spokenQuery) throw new Error("Speech-to-text returned an empty transcript. Please try again.");
          setTranscript(spokenQuery); setSttStatus("Transcript ready");
          await runQuery(spokenQuery, selectedStrategy, stt.sttLatency);
        } catch (sttError) { setError(sttError.message || "Speech-to-text failed. Please try again."); setIsProcessing(false); setPipelinePhase(""); setSttStatus(""); }
      };
      recorderRef.current = nextRecorder; nextRecorder.start(); setIsListening(true); setSttStatus("Recording…");
    } catch (microphoneError) { 
      console.error("Microphone setup failed:", microphoneError);
      setError(`Microphone error: ${microphoneError.name} - ${microphoneError.message}`); 
      setIsListening(false); setSttStatus(""); 
    }
  };

  const resetCache = async () => { await clearCache(); setMetrics(emptyMetrics); setLatencyHistory([]); setGuardrail(null); };

  return <div className="slide-deck"><section className="slide-scene slide-one"><div className="app-shell"><Header /><div className="dashboard-layout"><HistoryRail messages={messages} onClear={() => setMessages([])} guardrail={guardrail} transcript={transcript} sttStatus={sttStatus} /><div className="main-workspace"><QueryStage onSubmit={runQuery} onMicToggle={toggleMicrophone} isListening={isListening} isProcessing={isProcessing} error={error} strategy={strategy} setStrategy={setStrategy} /><TranscriptBand transcript={transcript} sttStatus={sttStatus} metrics={metrics} /></div><aside className="telemetry-rail"><LiveExecutionPanel metrics={metrics} processing={isProcessing} phase={pipelinePhase} guardrail={guardrail} /><LiveAnalytics metrics={metrics} latencyHistory={latencyHistory} /><section className="panel cache-panel"><div><span className="eyebrow"><Archive size={13} /> CACHE</span><strong>Backend retrieval cache</strong></div><button className="clear-cache" onClick={resetCache}><Trash2 size={13} /> Clear</button></section></aside></div><footer className="bottom-status"><span><span className="live-dot" /> Live backend environment</span><span>{metrics.retrievalMethod || "Awaiting first request"}</span><span>{formatMs(metrics.endToEndLatencyMs ?? metrics.streamRoundTripLatencyMs ?? metrics.serverTotalMs ?? metrics.totalLatencyMs)}</span></footer></div></section><section className="slide-scene slide-two"><LatencySequence metrics={metrics} /></section><AnswerDock answer={latestAnswer} metrics={metrics} isOpen={answersOpen} setIsOpen={setAnswersOpen} /></div>;
}
