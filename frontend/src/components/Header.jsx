import React, { useState, useEffect } from 'react';
import { Mic, Zap, ShieldCheck } from 'lucide-react';

export default function Header() {
  return (
    <header className="glass-card header-bar">
      <div className="brand-section">
        <div className="brand-icon">
          <Mic size={22} />
        </div>
        <div>
          <div className="brand-title">HH Goa 2026 — Task 2</div>
          <div className="brand-subtitle">Groq Whisper & Voice RAG Pipeline</div>
        </div>
      </div>

      <div className="header-badges">
        <span
          className="badge"
          style={{
            background: 'rgba(16, 185, 129, 0.15)',
            color: 'var(--accent-emerald)',
            borderColor: 'rgba(16, 185, 129, 0.3)',
          }}
        >
          <Zap size={13} />
          GROQ WHISPER STT READY
        </span>
        <span className="badge badge-status">
          <ShieldCheck size={13} />
          FASTAPI BACKEND CONNECTED
        </span>
      </div>
    </header>
  );
}
