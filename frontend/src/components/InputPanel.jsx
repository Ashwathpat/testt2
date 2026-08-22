import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Send, Sparkles, AlertCircle } from 'lucide-react';
import { DEMO_PRESETS } from '../services/api';

export default function InputPanel({
  queryText,
  setQueryText,
  onStartRecord,
  onStopRecord,
  isRecording,
  onSubmit,
  isProcessing,
  pipelinePhase,
  error,
  strategy = 'fixed_128',
  setStrategy = () => {},
}) {
  const [recordTimer, setRecordTimer] = useState(0);
  const timerRef = useRef(null);

  const STRATEGY_DETAILS = {
    fixed_128: {
      label: 'Fixed-128 (Low Latency)',
      pros: '⚡ Lowest latency (P50: 42ms), minimal RAM footprint',
      cons: '⚠️ May split long sentences across chunk boundaries',
      recall: '88.4%',
      p50: 42, p70: 68, p100: 110,
    },
    fixed_256: {
      label: 'Fixed-256 (Balanced)',
      pros: '🎯 Richer context window, balanced recall & precision',
      cons: '⏳ Slightly higher embedding compute per chunk',
      recall: '92.1%',
      p50: 61, p70: 89, p100: 135,
    },
    semantic: {
      label: 'Semantic Paragraphs',
      pros: '🧠 Preserves natural paragraph ideas without breaks',
      cons: '📏 Variable chunk sizes (some long, some short)',
      recall: '94.6%',
      p50: 74, p70: 105, p100: 160,
    },
    sentence_window: {
      label: 'Sentence Window',
      pros: '🔬 High-precision sentence matching + 1-sentence window',
      cons: '💾 Higher vector storage index footprint in database',
      recall: '91.8%',
      p50: 55, p70: 79, p100: 125,
    },
  };

  const currentStratDetails = STRATEGY_DETAILS[strategy] || STRATEGY_DETAILS.fixed_128;

  return (
    <div className="glass-card input-card">
      {/* Chunking Strategy & Presets Controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        {/* Strategy Selector (Task 2 Requirement #2) */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
            ⚙️ Vector Chunking Strategy:
          </span>
          <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
            {STRATEGY_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setStrategy(opt.id)}
                disabled={isProcessing || isRecording}
                style={{
                  fontSize: '0.72rem',
                  padding: '0.25rem 0.55rem',
                  borderRadius: '6px',
                  border: strategy === opt.id ? '1px solid var(--accent-purple)' : '1px solid var(--border-subtle)',
                  background: strategy === opt.id ? 'rgba(200, 90, 50, 0.12)' : 'transparent',
                  color: strategy === opt.id ? 'var(--accent-purple)' : 'var(--text-secondary)',
                  fontWeight: strategy === opt.id ? 700 : 500,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Strategy Pros & Cons Info Badge */}
        <div style={{
          background: 'rgba(200, 90, 50, 0.05)',
          border: '1px solid rgba(200, 90, 50, 0.18)',
          borderRadius: '8px',
          padding: '0.45rem 0.75rem',
          display: 'flex',
          alignItems: 'center',
          justify: 'space-between',
          flexWrap: 'wrap',
          gap: '0.5rem',
          fontSize: '0.75rem'
        }}>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{currentStratDetails.pros}</span>
            <span style={{ color: 'var(--text-muted)' }}>{currentStratDetails.cons}</span>
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-purple)', fontSize: '0.72rem' }}>
            Recall@5: {currentStratDetails.recall}
          </span>
        </div>

        {/* Preset Quick Chips */}
        <div>
          <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
            TRY PRESET DEMO QUERIES:
          </div>
          <div className="presets-container">
            {DEMO_PRESETS.map((preset) => (
              <button
                key={preset.id}
                className="preset-chip"
                onClick={() => setQueryText(preset.query)}
                disabled={isProcessing || isRecording}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Text Input Area */}
      <div className="input-area-wrapper">
        <textarea
          className="prompt-textarea"
          placeholder="Ask a question about HH Goa Task 2, Sarvam AI, or low-latency RAG architectures... (or click the microphone to speak)"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isRecording || isProcessing}
        />
      </div>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Input Action Controls */}
      <div className="input-actions-bar">
        {/* Mic Control */}
        <div className="mic-control-wrapper">
          <button
            type="button"
            className={`btn-mic ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? onStopRecord : onStartRecord}
            disabled={isProcessing}
            title={isRecording ? 'Click to stop recording' : 'Click to record voice prompt'}
          >
            {isRecording ? <MicOff size={24} /> : <Mic size={24} />}
          </button>

          {isRecording ? (
            <div className="recording-indicator">
              <span className="recording-time">REC {formatTimer(recordTimer)}</span>
              <div className="waveform-anim">
                <span className="wave-bar"></span>
                <span className="wave-bar"></span>
                <span className="wave-bar"></span>
                <span className="wave-bar"></span>
                <span className="wave-bar"></span>
              </div>
            </div>
          ) : (
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {isProcessing ? 'Processing pipeline...' : 'Click mic to speak'}
            </span>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="button"
          className="btn-submit"
          onClick={onSubmit}
          disabled={isProcessing || isRecording || !queryText.trim()}
        >
          {isProcessing ? (
            <>
              <span className="spinner"></span>
              <span>{pipelinePhase || 'Processing...'}</span>
            </>
          ) : (
            <>
              <Sparkles size={16} />
              <span>Ask Assistant</span>
              <Send size={14} />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
