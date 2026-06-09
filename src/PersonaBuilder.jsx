import { useState, useCallback } from "react";

// ─── Constants ───────────────────────────────────────────────────────────────
const ACCENT_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"];

const LOADING_STEPS = [
  "Fetching content",
  "Grounding with labor data",
  "Scoring candidate pool",
  "Building personas",
];

const AXIS_LABELS = {
  axis_A: "Motivation Diversity",
  axis_B: "Age / Life Stage",
  axis_C: "HH Income Range",
  axis_D: "Background & Education",
  axis_E: "Employment Context",
};

// ─── Inline Styles ───────────────────────────────────────────────────────────
const S = {
  root: {
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    background: "#f8fafc",
    minHeight: "100vh",
    color: "#1e293b",
  },
  header: {
    background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    color: "#fff",
    padding: "32px 24px 24px",
    textAlign: "center",
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 800,
    margin: "0 0 6px",
    letterSpacing: "-0.5px",
  },
  headerSub: {
    fontSize: 14,
    opacity: 0.85,
    margin: 0,
  },
  main: {
    maxWidth: 1100,
    margin: "0 auto",
    padding: "32px 20px 64px",
  },
  inputCard: {
    background: "#fff",
    borderRadius: 16,
    boxShadow: "0 1px 4px rgba(0,0,0,0.08), 0 4px 24px rgba(0,0,0,0.04)",
    padding: "28px 28px 24px",
    marginBottom: 28,
  },
  label: {
    display: "block",
    fontSize: 13,
    fontWeight: 600,
    color: "#64748b",
    marginBottom: 6,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  inputRow: {
    display: "flex",
    gap: 10,
    alignItems: "stretch",
  },
  input: {
    flex: 1,
    border: "1.5px solid #e2e8f0",
    borderRadius: 10,
    padding: "12px 16px",
    fontSize: 15,
    outline: "none",
    transition: "border-color 0.15s",
    color: "#1e293b",
    background: "#fff",
  },
  textarea: {
    width: "100%",
    border: "1.5px solid #e2e8f0",
    borderRadius: 10,
    padding: "12px 16px",
    fontSize: 14,
    outline: "none",
    resize: "vertical",
    minHeight: 120,
    color: "#1e293b",
    background: "#fff",
    boxSizing: "border-box",
    fontFamily: "inherit",
  },
  btnPrimary: {
    background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    color: "#fff",
    border: "none",
    borderRadius: 10,
    padding: "12px 24px",
    fontSize: 15,
    fontWeight: 700,
    cursor: "pointer",
    whiteSpace: "nowrap",
    transition: "opacity 0.15s, transform 0.1s",
  },
  btnSecondary: {
    background: "transparent",
    color: "#6366f1",
    border: "1.5px solid #6366f1",
    borderRadius: 8,
    padding: "6px 14px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    marginTop: 10,
  },
  toggleRow: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    marginTop: 10,
  },
  toggleLabel: {
    fontSize: 13,
    color: "#64748b",
    cursor: "pointer",
    userSelect: "none",
  },
  toggle: {
    width: 36,
    height: 20,
    borderRadius: 999,
    border: "none",
    cursor: "pointer",
    position: "relative",
    transition: "background 0.2s",
    padding: 0,
    flexShrink: 0,
  },
  toggleThumb: {
    position: "absolute",
    top: 2,
    width: 16,
    height: 16,
    borderRadius: "50%",
    background: "#fff",
    boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
    transition: "left 0.2s",
  },
  errorBox: {
    background: "#fff0f0",
    border: "1.5px solid #fca5a5",
    borderRadius: 10,
    padding: "12px 16px",
    color: "#dc2626",
    fontSize: 14,
    marginTop: 12,
  },
  // Loading
  loadingCard: {
    background: "#fff",
    borderRadius: 16,
    boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
    padding: "40px 28px",
    textAlign: "center",
    marginBottom: 28,
  },
  loadingTitle: {
    fontSize: 18,
    fontWeight: 700,
    color: "#1e293b",
    marginBottom: 24,
  },
  stepRow: {
    display: "flex",
    justifyContent: "center",
    gap: 12,
    flexWrap: "wrap",
  },
  stepBubble: (active, done) => ({
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "8px 16px",
    borderRadius: 999,
    fontSize: 13,
    fontWeight: 600,
    background: done ? "#ecfdf5" : active ? "#eef2ff" : "#f1f5f9",
    color: done ? "#10b981" : active ? "#6366f1" : "#94a3b8",
    border: `1.5px solid ${done ? "#6ee7b7" : active ? "#a5b4fc" : "#e2e8f0"}`,
    transition: "all 0.3s",
  }),
  spinner: {
    width: 18,
    height: 18,
    borderRadius: "50%",
    border: "2.5px solid #c7d2fe",
    borderTopColor: "#6366f1",
    animation: "spin 0.8s linear infinite",
    display: "inline-block",
  },
  // Banner
  banner: {
    background: "#fff",
    borderRadius: 16,
    boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
    padding: "24px 28px",
    marginBottom: 28,
  },
  bannerTitle: {
    fontSize: 20,
    fontWeight: 800,
    color: "#1e293b",
    marginBottom: 4,
  },
  bannerBrief: {
    fontSize: 14,
    color: "#475569",
    lineHeight: 1.6,
    marginBottom: 16,
    whiteSpace: "pre-wrap",
  },
  localCtxRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 20,
  },
  localCtxChip: {
    background: "#f1f5f9",
    borderRadius: 8,
    padding: "6px 12px",
    fontSize: 13,
    color: "#475569",
  },
  localCtxKey: {
    fontWeight: 700,
    color: "#1e293b",
  },
  axisGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(5, 1fr)",
    gap: 8,
    marginBottom: 12,
  },
  axisCell: (score) => ({
    borderRadius: 10,
    padding: "10px 8px",
    textAlign: "center",
    background: score === 3 ? "#ecfdf5" : score === 2 ? "#fffbeb" : "#fff1f2",
    border: `1.5px solid ${score === 3 ? "#6ee7b7" : score === 2 ? "#fcd34d" : "#fca5a5"}`,
  }),
  axisCellScore: (score) => ({
    fontSize: 22,
    fontWeight: 800,
    color: score === 3 ? "#10b981" : score === 2 ? "#f59e0b" : "#ef4444",
    lineHeight: 1,
    marginBottom: 2,
  }),
  axisCellLabel: {
    fontSize: 10,
    fontWeight: 600,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  axisCellRationale: {
    fontSize: 11,
    color: "#94a3b8",
    marginTop: 4,
    lineHeight: 1.3,
  },
  divScoreSummary: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    flexWrap: "wrap",
  },
  scorePill: (color, bg) => ({
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 14px",
    borderRadius: 999,
    background: bg,
    color: color,
    fontSize: 13,
    fontWeight: 700,
  }),
  // Cards
  cardsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(480px, 1fr))",
    gap: 24,
    marginBottom: 32,
  },
  card: (color) => ({
    background: "#fff",
    borderRadius: 16,
    boxShadow: "0 1px 4px rgba(0,0,0,0.08), 0 4px 24px rgba(0,0,0,0.04)",
    overflow: "hidden",
    borderTop: `4px solid ${color}`,
    display: "flex",
    flexDirection: "column",
    position: "relative",
    // Don't let a card split across a printed/PDF page (causes garbled text at the break).
    breakInside: "avoid",
    pageBreakInside: "avoid",
  }),
  cardHeader: {
    display: "flex",
    alignItems: "center",
    gap: 14,
    padding: "20px 20px 16px",
    borderBottom: "1px solid #f1f5f9",
  },
  avatar: {
    width: 56,
    height: 56,
    borderRadius: "50%",
    objectFit: "cover",
    flexShrink: 0,
    background: "#f1f5f9",
  },
  avatarFallback: (color) => ({
    width: 56,
    height: 56,
    borderRadius: "50%",
    background: color,
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 22,
    fontWeight: 800,
    flexShrink: 0,
  }),
  cardName: {
    fontSize: 17,
    fontWeight: 800,
    color: "#1e293b",
    margin: "0 0 2px",
  },
  cardArchetype: {
    fontSize: 13,
    color: "#64748b",
    margin: "0 0 6px",
  },
  badgeRow: {
    display: "flex",
    gap: 6,
    flexWrap: "wrap",
  },
  badge: (color, bg) => ({
    display: "inline-block",
    padding: "2px 10px",
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 700,
    color,
    background: bg,
  }),
  lockBtn: (locked) => ({
    position: "absolute",
    top: 14,
    right: 14,
    background: locked ? "#fff1f2" : "#f8fafc",
    border: `1.5px solid ${locked ? "#fca5a5" : "#e2e8f0"}`,
    borderRadius: 8,
    padding: "4px 8px",
    cursor: "pointer",
    fontSize: 15,
  }),
  cardBody: {
    padding: "16px 20px",
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: 14,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: 700,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    marginBottom: 8,
  },
  infoGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 8,
  },
  infoCell: {
    background: "#f8fafc",
    borderRadius: 8,
    padding: "8px 10px",
  },
  infoCellKey: {
    fontSize: 11,
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
    marginBottom: 2,
  },
  infoCellVal: {
    fontSize: 13,
    fontWeight: 600,
    color: "#1e293b",
  },
  motRow: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  motItem: {
    fontSize: 13,
    color: "#1e293b",
    lineHeight: 1.5,
  },
  painItem: {
    fontSize: 13,
    color: "#475569",
    lineHeight: 1.5,
  },
  panel: (bg, border) => ({
    background: bg,
    border: `1.5px solid ${border}`,
    borderRadius: 10,
    padding: "12px 14px",
    breakInside: "avoid",
    pageBreakInside: "avoid",
  }),
  panelTitle: {
    fontSize: 11,
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.07em",
    marginBottom: 8,
  },
  confRow: {
    display: "flex",
    gap: 8,
    flexWrap: "wrap",
    marginBottom: 8,
  },
  confTag: (color, bg) => ({
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    padding: "3px 10px",
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 600,
    color,
    background: bg,
  }),
  evidenceBullets: {
    fontSize: 12,
    color: "#64748b",
    paddingLeft: 16,
    margin: "4px 0 0",
    lineHeight: 1.6,
  },
  sourcingChips: {
    display: "flex",
    gap: 8,
    flexWrap: "wrap",
    marginBottom: 10,
  },
  sourcingChip: (color) => ({
    display: "inline-flex",
    alignItems: "center",
    gap: 5,
    padding: "4px 12px",
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 600,
    background: color + "15",
    color: color,
    border: `1px solid ${color}40`,
  }),
  hookBox: {
    background: "#fff",
    borderRadius: 8,
    padding: "10px 12px",
    marginBottom: 8,
  },
  hookHeadline: {
    fontSize: 13,
    fontWeight: 700,
    color: "#1e293b",
    marginBottom: 3,
  },
  hookProp: {
    fontSize: 12,
    color: "#475569",
    lineHeight: 1.5,
  },
  frictionKiller: {
    fontSize: 12,
    color: "#6d28d9",
    fontWeight: 600,
    display: "flex",
    alignItems: "flex-start",
    gap: 6,
  },
  // Job Ad Rewrite
  jawCard: {
    background: "#fff",
    borderRadius: 16,
    boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
    padding: "24px 28px",
    marginBottom: 28,
  },
  jawTitle: {
    fontSize: 18,
    fontWeight: 800,
    marginBottom: 16,
    color: "#1e293b",
  },
  jawGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
    gap: 12,
  },
  jawCell: (bg, border, titleColor) => ({
    background: bg,
    border: `1.5px solid ${border}`,
    borderRadius: 12,
    padding: "14px 16px",
    titleColor,
  }),
  jawCellTitle: (color) => ({
    fontSize: 11,
    fontWeight: 700,
    color,
    textTransform: "uppercase",
    letterSpacing: "0.07em",
    marginBottom: 6,
  }),
  jawCellText: {
    fontSize: 13,
    color: "#1e293b",
    lineHeight: 1.55,
  },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────
function normalizeSegments(personas) {
  if (!personas || personas.length === 0) return personas;
  const total = personas.reduce(
    (s, p) => s + (p.metadata?.segment_size_percentage || 0),
    0
  );
  if (total === 0) return personas;
  return personas.map((p) => ({
    ...p,
    metadata: {
      ...p.metadata,
      segment_size_percentage: Math.round(
        ((p.metadata?.segment_size_percentage || 0) / total) * 100
      ),
    },
  }));
}

function confidenceColor(level) {
  if (!level) return ["#94a3b8", "#f1f5f9"];
  const l = level.toLowerCase();
  if (l === "high") return ["#10b981", "#ecfdf5"];
  if (l === "medium") return ["#f59e0b", "#fffbeb"];
  if (l === "low") return ["#ef4444", "#fff1f2"];
  if (l === "inferred") return ["#8b5cf6", "#f5f3ff"];
  return ["#94a3b8", "#f1f5f9"];
}

function scoreColor(score) {
  if (score >= 3) return "#10b981";
  if (score === 2) return "#f59e0b";
  return "#ef4444";
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function LoadingPanel({ step }) {
  return (
    <div style={S.loadingCard}>
      <p style={S.loadingTitle}>Analyzing job description…</p>
      <div style={S.stepRow}>
        {LOADING_STEPS.map((label, i) => {
          const done = i < step;
          const active = i === step;
          return (
            <div key={label} style={S.stepBubble(active, done)}>
              {done ? (
                <span>✓</span>
              ) : active ? (
                <span style={S.spinner} />
              ) : (
                <span style={{ opacity: 0.4 }}>○</span>
              )}
              {label}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AxisGrid({ diversityScoring }) {
  const axes = ["axis_A", "axis_B", "axis_C", "axis_D", "axis_E"];
  return (
    <div style={S.axisGrid}>
      {axes.map((key) => {
        const axis = diversityScoring?.[key];
        const score = axis?.score ?? 1;
        return (
          <div key={key} style={S.axisCell(score)}>
            <div style={S.axisCellScore(score)}>{score}</div>
            <div style={S.axisCellLabel}>
              {key.replace("axis_", "Axis ")}
            </div>
            <div style={S.axisCellLabel} title={AXIS_LABELS[key]}>
              {AXIS_LABELS[key].split(" ")[0]}
            </div>
            {axis?.rationale && (
              <div style={S.axisCellRationale}>
                {axis.rationale.slice(0, 48)}
                {axis.rationale.length > 48 ? "…" : ""}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Banner({ data }) {
  const { role_summary, recruiter_brief, local_context, diversity_scoring, market_salary } = data;
  const totalScore = diversity_scoring?.total_score ?? 0;
  const personaCount = diversity_scoring?.target_persona_count ?? 0;
  const bridgeIncluded = diversity_scoring?.bridge_persona_included;
  return (
    <div style={S.banner}>
      <h2 style={S.bannerTitle}>{role_summary}</h2>
      <p style={S.bannerBrief}>{recruiter_brief}</p>

      {local_context && (
        <div style={S.localCtxRow}>
          {local_context.metro_area && (
            <span style={S.localCtxChip}>
              <span style={S.localCtxKey}>Metro: </span>
              {local_context.metro_area}
            </span>
          )}
          {local_context.posted_compensation ? (
            <span style={{ ...S.localCtxChip, background: "#ecfdf5", borderColor: "#a7f3d0" }}>
              <span style={{ ...S.localCtxKey, color: "#047857" }}>Target salary: </span>
              {local_context.posted_compensation}
            </span>
          ) : (
            <span style={{ ...S.localCtxChip, color: "#94a3b8" }}>
              <span style={S.localCtxKey}>Target salary: </span>
              Not mentioned
            </span>
          )}
          {local_context.cost_of_living_index && (
            <span style={S.localCtxChip}>
              <span style={S.localCtxKey}>COL Index: </span>
              {local_context.cost_of_living_index}
            </span>
          )}
          {local_context.role_type && (
            <span style={S.localCtxChip}>
              <span style={S.localCtxKey}>Type: </span>
              {local_context.role_type}
            </span>
          )}
          {local_context.hiring_volume && (
            <span style={S.localCtxChip}>
              <span style={S.localCtxKey}>Volume: </span>
              {local_context.hiring_volume}
            </span>
          )}
        </div>
      )}

      {market_salary && market_salary.average != null && (() => {
        const sym = market_salary.currency_symbol || "";
        const code = market_salary.currency_code ? ` ${market_salary.currency_code}` : "";
        const fmt = (n) => (n == null ? null : `${sym}${Number(n).toLocaleString()}`);
        const avg = fmt(market_salary.average);
        const lo = fmt(market_salary.low);
        const hi = fmt(market_salary.high);
        const range = lo && hi ? `${lo}–${hi}` : avg;
        const cnt = market_salary.posting_count;
        return (
          <div
            style={{
              marginTop: 10,
              padding: "10px 14px",
              background: "#eff6ff",
              border: "1px solid #bfdbfe",
              borderRadius: 10,
              display: "flex",
              flexWrap: "wrap",
              alignItems: "baseline",
              gap: 8,
            }}
          >
            <span style={{ fontSize: 13, fontWeight: 700, color: "#1d4ed8" }}>
              💵 Market Salary Range
            </span>
            <span style={{ fontSize: 15, fontWeight: 700, color: "#0f172a" }}>
              {range}{code}
            </span>
            {avg && lo && hi && (
              <span style={{ fontSize: 12, color: "#475569" }}>(avg {avg})</span>
            )}
            <span style={{ fontSize: 12, color: "#64748b" }}>
              · {market_salary.basis}
            </span>
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: "#1d4ed8",
                background: "#dbeafe",
                borderRadius: 6,
                padding: "2px 7px",
              }}
            >
              {market_salary.source}{cnt ? ` · ${cnt} live postings` : ""} · {market_salary.confidence}
            </span>
            <span style={{ flexBasis: "100%", fontSize: 11, color: "#94a3b8", marginTop: 2 }}>
              Market value for this role in this city — not the salary in this posting.
            </span>
          </div>
        );
      })()}

      {diversity_scoring && (
        <>
          <div
            style={{
              fontSize: 12,
              fontWeight: 700,
              color: "#94a3b8",
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              marginBottom: 10,
            }}
          >
            Diversity Scoring — {diversity_scoring.preset_used}
          </div>
          <AxisGrid diversityScoring={diversity_scoring} />
          <div style={S.divScoreSummary}>
            <span
              style={S.scorePill(
                scoreColor(totalScore),
                scoreColor(totalScore) + "18"
              )}
            >
              Total Score: {totalScore}/15
            </span>
            <span style={S.scorePill("#6366f1", "#eef2ff")}>
              {personaCount} Personas
            </span>
            {bridgeIncluded && (
              <span style={S.scorePill("#f59e0b", "#fffbeb")}>
                🌉 Bridge Persona Included
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function AvatarImg({ name, color }) {
  const [failed, setFailed] = useState(false);
  const seed = encodeURIComponent(name || "candidate");
  const src = `https://api.dicebear.com/7.x/adventurer/svg?seed=${seed}`;

  if (failed) {
    return (
      <div style={S.avatarFallback(color)}>
        {(name || "?")[0].toUpperCase()}
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={name}
      style={S.avatar}
      onError={() => setFailed(true)}
    />
  );
}

function ConfidencePanel({ evidence_confidence }) {
  if (!evidence_confidence) return null;
  const {
    overall_score,
    sourced_vs_inferred,
    confidence_breakdown,
    notes,
    evidence_basis,
  } = evidence_confidence;

  return (
    <div style={S.panel("#f8fafc", "#e2e8f0")}>
      <div style={{ ...S.panelTitle, color: "#64748b" }}>
        Evidence & Confidence
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 8,
          flexWrap: "wrap",
        }}
      >
        <div
          style={{
            fontSize: 24,
            fontWeight: 800,
            color:
              overall_score >= 75
                ? "#10b981"
                : overall_score >= 50
                ? "#f59e0b"
                : "#ef4444",
          }}
        >
          {overall_score}%
        </div>
        {sourced_vs_inferred && (
          <div style={{ fontSize: 12, color: "#475569", lineHeight: 1.5, flex: 1, minWidth: 200 }}>
            <span style={{ fontWeight: 700, color: "#64748b" }}>Sourced vs inferred: </span>
            {sourced_vs_inferred}
          </div>
        )}
      </div>
      {confidence_breakdown && (confidence_breakdown.high || confidence_breakdown.medium || confidence_breakdown.low) && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
          {[
            ["High", confidence_breakdown.high, "#10b981", "#ecfdf5"],
            ["Medium", confidence_breakdown.medium, "#f59e0b", "#fffbeb"],
            ["Low", confidence_breakdown.low, "#ef4444", "#fef2f2"],
          ].map(([label, items, color, bg]) =>
            items && items.length > 0 ? (
              <span key={label} style={{ fontSize: 11, color, background: bg, border: `1px solid ${color}33`, borderRadius: 8, padding: "3px 9px", lineHeight: 1.4 }}>
                <strong>{label}:</strong> {items.join(", ")}
              </span>
            ) : null
          )}
        </div>
      )}
      {notes && (
        <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.5, marginBottom: 6 }}>
          {notes}
        </div>
      )}
      {evidence_basis && evidence_basis.length > 0 && (
        <ul style={S.evidenceBullets}>
          {evidence_basis.map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function MiniBtn({ onClick, href, children, primary }) {
  const style = {
    fontSize: 11,
    fontWeight: 600,
    padding: "4px 9px",
    borderRadius: 6,
    border: "1px solid " + (primary ? "#7c3aed" : "#cbd5e1"),
    background: primary ? "#7c3aed" : "#fff",
    color: primary ? "#fff" : "#475569",
    cursor: "pointer",
    textDecoration: "none",
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
  };
  return href ? (
    <a href={href} target="_blank" rel="noreferrer" style={style}>{children}</a>
  ) : (
    <button onClick={onClick} style={style}>{children}</button>
  );
}

function SearchStringBlock({ query }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    try { navigator.clipboard.writeText(query); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch {}
  };
  const enc = encodeURIComponent(query);
  const links = [
    ["LinkedIn Recruiter", `https://www.linkedin.com/talent/search?keywords=${enc}`],
    ["LinkedIn", `https://www.linkedin.com/search/results/people/?keywords=${enc}`],
    ["Indeed", `https://www.indeed.com/jobs?q=${enc}`],
    ["Google X-ray", `https://www.google.com/search?q=${encodeURIComponent("site:linkedin.com/in " + query)}`],
  ];
  return (
    <div style={{ marginBottom: 8 }}>
      <div
        style={{
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          fontSize: 11.5, color: "#334155", background: "#f1f5f9",
          border: "1px solid #e2e8f0", borderRadius: 8, padding: "8px 10px",
          lineHeight: 1.5, whiteSpace: "pre-wrap", wordBreak: "break-word",
        }}
      >
        🔍 {query}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
        <MiniBtn onClick={copy} primary>{copied ? "✓ Copied" : "Copy string"}</MiniBtn>
        {links.map(([label, href]) => (
          <MiniBtn key={label} href={href}>{label} ↗</MiniBtn>
        ))}
      </div>
    </div>
  );
}

function buildSourcingPack(persona) {
  const m = persona.metadata || {};
  const ra = persona.recruiter_action || {};
  const sc = ra.sourcing_channel || {};
  const hook = ra.conversion_hook || {};
  const drop = ra.application_dropoff_risk || {};
  const lines = [
    `SOURCING PACK — ${m.name || "Persona"}${m.archetype ? ` (${m.archetype})` : ""}`,
    m.segment_size_percentage != null ? `Segment: ${m.segment_size_percentage}% of pool` : "",
    "",
    sc.primary ? `Channel: ${sc.primary}` : "",
    sc.search_string ? `Boolean: ${sc.search_string}` : "",
    (sc.target_companies && sc.target_companies.length) ? `Target companies: ${sc.target_companies.join(", ")}` : "",
    sc.organic_play ? `Organic play: ${sc.organic_play}` : "",
    "",
    hook.headline ? `Conversion headline: ${hook.headline}` : "",
    hook.core_value_prop ? `Value prop: ${hook.core_value_prop}` : "",
    ra.outreach_script ? `Outreach: ${ra.outreach_script}` : "",
    "",
    drop.risk ? `Drop-off risk: ${drop.risk}` : "",
    drop.fix ? `Fix: ${drop.fix}` : "",
  ];
  return lines.filter((l) => l !== undefined).join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function OutreachScript({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    try { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch {}
  };
  return (
    <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, padding: "8px 10px", marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#15803d", textTransform: "uppercase", letterSpacing: 0.5 }}>✉️ Outreach script</span>
        <MiniBtn onClick={copy}>{copied ? "✓ Copied" : "Copy"}</MiniBtn>
      </div>
      <div style={{ fontSize: 12.5, color: "#166534", lineHeight: 1.5 }}>{text}</div>
    </div>
  );
}

function ExportPackButton({ persona }) {
  const [done, setDone] = useState(false);
  const copy = () => {
    try { navigator.clipboard.writeText(buildSourcingPack(persona)); setDone(true); setTimeout(() => setDone(false), 1500); } catch {}
  };
  return <MiniBtn onClick={copy}>{done ? "✓ Copied pack" : "📋 Export sourcing pack"}</MiniBtn>;
}

function PersonaCard({ persona, index, locked, onToggleLock }) {
  const color = ACCENT_COLORS[index % ACCENT_COLORS.length];
  const { metadata, demographics, financials, drivers_and_friction, tech_profile, evidence_confidence, screening_question, recruiter_action, deal_breakers, candidate_journey } = persona;

  return (
    <div className="nova-card" style={S.card(color)}>
      {/* Lock Button */}
      <button
        style={S.lockBtn(locked)}
        onClick={onToggleLock}
        title={locked ? "Locked — won't regenerate" : "Unlocked — will regenerate"}
        aria-label={locked ? "Unlock card" : "Lock card"}
      >
        {locked ? "🔒" : "🔓"}
      </button>

      {/* Card Header */}
      <div style={S.cardHeader}>
        <AvatarImg name={metadata?.name} color={color} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={S.cardName}>{metadata?.name || "Unnamed Persona"}</p>
          <p style={S.cardArchetype}>{metadata?.archetype}</p>
          <div style={S.badgeRow}>
            {metadata?.segment_size_percentage != null && (
              <span style={S.badge("#fff", color)}>
                {metadata.segment_size_percentage}% of pool
              </span>
            )}
            {metadata?.is_bridge_persona && (
              <span style={S.badge("#92400e", "#fffbeb")}>
                🌉 Bridge
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="nova-cardbody" style={S.cardBody}>
        {/* Demographics + Financials */}
        <div>
          <div style={S.sectionTitle}>Demographics & Financials</div>
          <div style={S.infoGrid}>
            {demographics?.age_range && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Age Range</div>
                <div style={S.infoCellVal}>{demographics.age_range}</div>
              </div>
            )}
            {demographics?.education && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Education</div>
                <div style={S.infoCellVal}>{demographics.education}</div>
              </div>
            )}
            {demographics?.employment_status && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Employment</div>
                <div style={S.infoCellVal}>{demographics.employment_status}</div>
              </div>
            )}
            {financials?.pew_household_income_tier && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Income Tier</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span
                    style={{
                      ...S.infoCellVal,
                      padding: "2px 8px",
                      borderRadius: 999,
                      background: "#eef2ff",
                      color: "#6366f1",
                      fontSize: 12,
                      fontWeight: 700,
                    }}
                  >
                    {financials.pew_household_income_tier}
                  </span>
                </div>
              </div>
            )}
            {financials?.household_income_range && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>HH Income</div>
                <div style={S.infoCellVal}>{financials.household_income_range}</div>
                {financials.household_income_note && (
                  <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>
                    {financials.household_income_note}
                  </div>
                )}
              </div>
            )}
            {financials?.target_monthly_income_from_role && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Expected Monthly</div>
                <div style={S.infoCellVal}>{financials.target_monthly_income_from_role}</div>
              </div>
            )}
            {financials?.hours_per_week_expected && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Hrs/Week</div>
                <div style={S.infoCellVal}>{financials.hours_per_week_expected}</div>
              </div>
            )}
            {financials?.income_dependency && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Dependency</div>
                <div style={S.infoCellVal}>{financials.income_dependency}</div>
              </div>
            )}
            {financials?.payment_preference && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Payment Pref</div>
                <div style={S.infoCellVal}>{financials.payment_preference}</div>
              </div>
            )}
            {tech_profile?.tech_savviness_score != null && (
              <div style={S.infoCell}>
                <div style={S.infoCellKey}>Tech Score</div>
                <div style={S.infoCellVal}>{tech_profile.tech_savviness_score}/5</div>
              </div>
            )}
          </div>
          {tech_profile?.hardware_devices?.length > 0 && (
            <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
              {tech_profile.hardware_devices.map((d) => (
                <span
                  key={d}
                  style={{
                    padding: "2px 8px",
                    background: "#f1f5f9",
                    borderRadius: 6,
                    fontSize: 11,
                    color: "#475569",
                    fontWeight: 600,
                  }}
                >
                  {d}
                </span>
              ))}
              {tech_profile.key_apps?.map((a) => (
                <span
                  key={a}
                  style={{
                    padding: "2px 8px",
                    background: "#eef2ff",
                    borderRadius: 6,
                    fontSize: 11,
                    color: "#6366f1",
                    fontWeight: 600,
                  }}
                >
                  {a}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Motivations */}
        {(drivers_and_friction?.primary_motivation || drivers_and_friction?.secondary_motivation) && (
          <div>
            <div style={S.sectionTitle}>Motivations</div>
            <div style={S.motRow}>
              {drivers_and_friction.primary_motivation && (
                <div style={S.motItem}>
                  🎯 <strong>Primary:</strong> {drivers_and_friction.primary_motivation}
                </div>
              )}
              {drivers_and_friction.secondary_motivation && (
                <div style={S.motItem}>
                  ✨ <strong>Secondary:</strong> {drivers_and_friction.secondary_motivation}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Pain Points */}
        {(drivers_and_friction?.pain_point_1 || drivers_and_friction?.pain_point_2) && (
          <div>
            <div style={S.sectionTitle}>Pain Points</div>
            <div style={S.motRow}>
              {drivers_and_friction.pain_point_1 && (
                <div style={S.painItem}>⚡ {drivers_and_friction.pain_point_1}</div>
              )}
              {drivers_and_friction.pain_point_2 && (
                <div style={S.painItem}>⚡ {drivers_and_friction.pain_point_2}</div>
              )}
            </div>
          </div>
        )}

        {/* Evidence & Confidence */}
        <ConfidencePanel evidence_confidence={evidence_confidence} />

        {/* Optional interview signal (de-emphasized, collapsible) */}
        {screening_question?.question && (
          <details style={{ ...S.panel("#f8fafc", "#e2e8f0"), padding: "10px 14px" }}>
            <summary
              style={{
                ...S.panelTitle,
                color: "#94a3b8",
                cursor: "pointer",
                listStyle: "revert",
                margin: 0,
              }}
            >
              Optional interview signal
            </summary>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#1e293b", margin: "8px 0", lineHeight: 1.4 }}>
              "{screening_question.question}"
            </div>
            {screening_question.high_risk_answer && (
              <div
                style={{
                  fontSize: 12,
                  color: "#9a3412",
                  background: "#fff7ed",
                  border: "1px solid #fed7aa",
                  borderRadius: 8,
                  padding: "8px 10px",
                  marginBottom: 6,
                  lineHeight: 1.5,
                }}
              >
                🚩 <strong>High-risk answer:</strong> {screening_question.high_risk_answer}
              </div>
            )}
            {screening_question.risk_rationale && (
              <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6, lineHeight: 1.5 }}>
                <strong style={{ color: "#475569" }}>Why it's a risk: </strong>
                {screening_question.risk_rationale}
              </div>
            )}
            {screening_question.why_it_matters && (
              <div style={{ fontSize: 12, color: "#3b82f6", lineHeight: 1.5 }}>
                💡 {screening_question.why_it_matters}
              </div>
            )}
          </details>
        )}

        {/* Churn Risk */}
        {drivers_and_friction?.anti_pattern_signals && (
          <div style={S.panel("#fff1f2", "#fca5a5")}>
            <div style={{ ...S.panelTitle, color: "#dc2626" }}>
              Churn Risk Signals
            </div>
            {drivers_and_friction.anti_pattern_signals.interview_red_flag && (
              <div style={{ fontSize: 13, color: "#7f1d1d", marginBottom: 6, lineHeight: 1.4 }}>
                🚩 <strong>Interview Red Flag:</strong>{" "}
                {drivers_and_friction.anti_pattern_signals.interview_red_flag}
              </div>
            )}
            {drivers_and_friction.anti_pattern_signals.churn_trigger && (
              <div style={{ fontSize: 13, color: "#7f1d1d", lineHeight: 1.4 }}>
                ⚠️ <strong>Churn Trigger:</strong>{" "}
                {drivers_and_friction.anti_pattern_signals.churn_trigger}
              </div>
            )}
          </div>
        )}

        {/* Deal-breakers */}
        {deal_breakers && deal_breakers.length > 0 && (
          <div style={S.panel("#fff1f2", "#fecdd3")}>
            <div style={{ ...S.panelTitle, color: "#be123c" }}>Deal-breakers</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {deal_breakers.map((d, i) => (
                <span key={i} style={{ fontSize: 12, fontWeight: 600, color: "#be123c", background: "#fff", border: "1px solid #fecdd3", borderRadius: 999, padding: "3px 10px" }}>
                  ⛔ {d}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Candidate journey drop-off map */}
        {candidate_journey && Object.values(candidate_journey).some((v) => v && String(v).trim()) && (
          <div style={S.panel("#fff7ed", "#fed7aa")}>
            <div style={{ ...S.panelTitle, color: "#c2410c" }}>Candidate journey — where they drop off</div>
            {[
              ["Discovery", candidate_journey.discovery_risk],
              ["Click", candidate_journey.click_risk],
              ["Apply", candidate_journey.apply_risk],
              ["Interview", candidate_journey.interview_risk],
              ["Offer", candidate_journey.offer_risk],
              ["Early churn", candidate_journey.early_churn_risk],
            ].map(([stage, risk]) =>
              risk ? (
                <div key={stage} style={{ fontSize: 12.5, color: "#7c2d12", lineHeight: 1.5, marginBottom: 4, display: "flex", gap: 6 }}>
                  <span style={{ fontWeight: 700, minWidth: 84, color: "#9a3412" }}>{stage}:</span>
                  <span>{risk}</span>
                </div>
              ) : null
            )}
          </div>
        )}

        {/* Recruiter Action */}
        {recruiter_action && (
          <div style={S.panel("#faf5ff", "#d8b4fe")}>
            <div style={{ ...S.panelTitle, color: "#7c3aed", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>Recruiter Action</span>
              <ExportPackButton persona={persona} />
            </div>
            {/* Sourcing */}
            {(recruiter_action.sourcing_channel?.primary || recruiter_action.sourcing_channel?.organic_play) && (
              <div style={S.sourcingChips}>
                {recruiter_action.sourcing_channel.primary && (
                  <span style={S.sourcingChip("#6366f1")}>
                    📣 {recruiter_action.sourcing_channel.primary}
                  </span>
                )}
                {recruiter_action.sourcing_channel.organic_play && (
                  <span style={S.sourcingChip("#10b981")}>
                    🌱 {recruiter_action.sourcing_channel.organic_play}
                  </span>
                )}
              </div>
            )}
            {/* Boolean search string — copy + deep-link launch */}
            {recruiter_action.sourcing_channel?.search_string && (
              <SearchStringBlock query={recruiter_action.sourcing_channel.search_string} />
            )}
            {/* Target companies */}
            {recruiter_action.sourcing_channel?.target_companies?.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                {recruiter_action.sourcing_channel.target_companies.map((c, i) => (
                  <span
                    key={i}
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: "#7c3aed",
                      background: "#f3e8ff",
                      border: "1px solid #e9d5ff",
                      borderRadius: 999,
                      padding: "3px 9px",
                    }}
                  >
                    {c}
                  </span>
                ))}
              </div>
            )}
            {/* Conversion Hook */}
            {recruiter_action.conversion_hook && (
              <div style={S.hookBox}>
                {recruiter_action.conversion_hook.headline && (
                  <div style={S.hookHeadline}>
                    "{recruiter_action.conversion_hook.headline}"
                  </div>
                )}
                {recruiter_action.conversion_hook.core_value_prop && (
                  <div style={S.hookProp}>
                    {recruiter_action.conversion_hook.core_value_prop}
                  </div>
                )}
              </div>
            )}
            {/* Outreach script (copyable) */}
            {recruiter_action.outreach_script && (
              <OutreachScript text={recruiter_action.outreach_script} />
            )}
            {/* Application drop-off risk + fix */}
            {recruiter_action.application_dropoff_risk?.risk && (
              <div
                style={{
                  background: "#fffbeb",
                  border: "1px solid #fde68a",
                  borderRadius: 8,
                  padding: "8px 10px",
                  marginBottom: 8,
                  lineHeight: 1.5,
                }}
              >
                <div style={{ fontSize: 12, color: "#92400e" }}>
                  <strong>📉 Drop-off risk: </strong>
                  {recruiter_action.application_dropoff_risk.risk}
                </div>
                {recruiter_action.application_dropoff_risk.fix && (
                  <div style={{ fontSize: 12, color: "#166534", marginTop: 4 }}>
                    <strong>✅ Fix: </strong>
                    {recruiter_action.application_dropoff_risk.fix}
                  </div>
                )}
              </div>
            )}
            {/* Funnel Friction Killer */}
            {recruiter_action.funnel_friction_killer && (
              <div style={S.frictionKiller}>
                <span>⚡</span>
                <span>{recruiter_action.funnel_friction_killer}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function JobAdRewritePanel({ jaw }) {
  if (!jaw) return null;
  // Don't render an empty section if the model omitted/truncated this block.
  if (!Object.values(jaw).some((v) => v && String(v).trim())) return null;

  const cells = [
    {
      key: "current_jd_risk",
      label: "Current JD Risk",
      bg: "#fff1f2",
      border: "#fca5a5",
      color: "#dc2626",
      value: jaw.current_jd_risk,
    },
    {
      key: "missing_motivator",
      label: "Missing Motivator",
      bg: "#fffbeb",
      border: "#fcd34d",
      color: "#d97706",
      value: jaw.missing_motivator,
    },
    {
      key: "recommended_headline",
      label: "Recommended Headline",
      bg: "#ecfdf5",
      border: "#6ee7b7",
      color: "#059669",
      value: jaw.recommended_headline,
    },
    {
      key: "bullet_to_add",
      label: "Bullet to Add",
      bg: "#eff6ff",
      border: "#bfdbfe",
      color: "#1d4ed8",
      value: jaw.bullet_to_add,
    },
    {
      key: "bullet_to_remove",
      label: "Bullet to Remove",
      bg: "#faf5ff",
      border: "#d8b4fe",
      color: "#7c3aed",
      value: jaw.bullet_to_remove,
    },
    {
      key: "cta_improvement",
      label: "CTA Improvement",
      bg: "#fff7ed",
      border: "#fed7aa",
      color: "#ea580c",
      value: jaw.cta_improvement,
    },
  ];

  return (
    <div style={S.jawCard}>
      <h3 style={S.jawTitle}>Job Ad Rewrite Recommendations</h3>
      <div style={S.jawGrid}>
        {cells.map(({ key, label, bg, border, color, value }) =>
          value ? (
            <div key={key} style={{ background: bg, border: `1.5px solid ${border}`, borderRadius: 12, padding: "14px 16px" }}>
              <div style={S.jawCellTitle(color)}>{label}</div>
              <div style={S.jawCellText}>{value}</div>
            </div>
          ) : null
        )}
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

const DEFINITIONS = [
  {
    group: "Persona Scores",
    items: [
      ["Segment Size %", "The share of the qualified candidate pool this persona represents. All personas in a map add up to 100%."],
      ["Archetype", "A short label for the persona's real prior background and path into the role (e.g. \"Agency Campaign Operator\") — not a personality type."],
      ["Tech Savviness (out of 5)", "How comfortable this segment is with digital tools. 1 = minimal, needs assisted use · 2 = basic apps · 3 = confident everyday user · 4 = advanced, multi-tool · 5 = highly technical power user. Shown for gig, hourly, and frontline roles only — for office/corporate roles it's always high, so it carries no signal and is hidden."],
      ["Evidence Confidence (0–100%)", "How well-supported this persona is by the JD and any external data. High ≥ 75% · Medium 50–74% · Low < 50%. Lower scores mean more of the persona is inferred."],
      ["Sourced vs Inferred", "A plain statement of which parts of the persona are grounded in the JD or real data versus inferred by the model — so you know exactly how much to trust each card."],
      ["Key Apps", "The apps and platforms this segment lives in — useful for knowing where to source and reach them."],
    ],
  },
  {
    group: "Financial Terms",
    items: [
      ["Household Income Tier", "Where the persona's household sits on the local income scale: Lower · Lower-middle · Middle · Upper-middle · Upper. Bands are calibrated to the role's country and cost of living (US Pew tiers for US roles, local tiers for India/UK)."],
      ["Income Dependency", "How much this role's pay matters to the household. Primary = main or sole income · Secondary = a second household income · Supplemental = a top-up to other earnings."],
      ["Target Monthly Income", "The gross monthly pay this segment expects from the role, anchored to the JD's stated salary when available."],
      ["Payment Preference", "How and how often the segment prefers to be paid (e.g. Daily instant, Weekly, Monthly). Shown for gig and hourly roles only — salaried roles are always monthly, so it's hidden there."],
    ],
  },
  {
    group: "How Segments Are Built",
    items: [
      ["5-Axis Score", "Each map is scored 1–3 on five axes; the total (5–15) sets how many personas are generated. A higher total means a more varied candidate market and more personas."],
      ["Axis A — Motivational Diversity", "How varied candidates' reasons for taking the role are (1 = one shared motive, 3 = many competing motives)."],
      ["Axis B — Age / Life Stage", "How many distinct life stages apply (1 = one band, 3 = all generations)."],
      ["Axis C — Household Income", "How spread out candidate incomes are (1 = uniform, 3 = full spectrum)."],
      ["Axis D — Background / Education", "How open the role is to different backgrounds (1 = strict gatekeeping, 3 = no barriers)."],
      ["Axis E — Employment Context", "Stability of the work type (1 = single stable status, 3 = gig or volatile)."],
      ["Bridge Persona", "An extra crossover segment, included only when the role is broadly accessible (low barrier + flexible structure + economic accessibility + wide applicant pool). It replaces the weakest segment rather than adding one."],
    ],
  },
  {
    group: "Recruiter Fields",
    items: [
      ["Sourcing Channel", "Where to actually find this segment — the specific platform and how to use it (e.g. LinkedIn Recruiter, Naukri/Instahyre for India)."],
      ["Search String", "A copy-paste boolean search (role-title synonyms + industry terms + skills) you can drop straight into LinkedIn Recruiter or a job board to surface this pool."],
      ["Target Companies", "5–8 named companies or talent pools to source this persona from — competitors and adjacent-industry leaders to poach from."],
      ["Conversion Hook", "The persona-specific headline and pitch most likely to get this segment to apply — different for each pool."],
      ["Drop-off Risk & Fix", "Why this segment abandons before applying or declines the offer, and the precise change to the JD, process, or messaging that removes it. This is the activation lever, more useful than an interview question."],
      ["Interview Red Flag", "What this candidate might say that signals a high risk of quitting within 30 days."],
      ["Churn Trigger", "The single operational change (shift, on-site rule, pay timing) most likely to make this segment disengage or ghost."],
      ["Optional Interview Signal", "A secondary, collapsible aid: one recommended question plus the high-risk answer to listen for. Nice-to-have, not the core product."],
    ],
  },
];

function DefinitionsPanel() {
  return (
    <div style={{ maxWidth: 860, margin: "0 auto" }}>
      <div style={{ fontSize: 22, fontWeight: 800, color: "#1e293b", marginBottom: 6 }}>
        How to read this map
      </div>
      <p style={{ color: "#64748b", fontSize: 14, marginBottom: 24 }}>
        Definitions for the scores and terms used across each persona.
      </p>
      {DEFINITIONS.map((sec) => (
        <div key={sec.group} style={{ marginBottom: 26 }}>
          <div
            style={{
              fontSize: 12,
              fontWeight: 800,
              textTransform: "uppercase",
              letterSpacing: 0.6,
              color: "#6366f1",
              marginBottom: 10,
            }}
          >
            {sec.group}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {sec.items.map(([term, def]) => (
              <div
                key={term}
                style={{
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 10,
                  padding: "12px 16px",
                }}
              >
                <div style={{ fontWeight: 700, color: "#1e293b", fontSize: 14, marginBottom: 3 }}>
                  {term}
                </div>
                <div style={{ color: "#475569", fontSize: 13.5, lineHeight: 1.5 }}>{def}</div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function MismatchBlock({ data }) {
  if (!data || !(data.you_want || data.jd_attracts)) return null;
  const Row = ({ label, value, color }) =>
    value ? (
      <div style={{ marginBottom: 6, fontSize: 13, lineHeight: 1.5 }}>
        <span style={{ fontWeight: 700, color }}>{label} </span>
        <span style={{ color: "#334155" }}>{value}</span>
      </div>
    ) : null;
  return (
    <div style={{ background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 14, padding: "16px 18px", marginBottom: 20 }}>
      <div style={{ fontSize: 13, fontWeight: 800, textTransform: "uppercase", letterSpacing: 0.6, color: "#b45309", marginBottom: 10 }}>
        ⚖️ Who you want vs. who your JD attracts
      </div>
      <Row label="You want:" value={data.you_want} color="#0f766e" />
      <Row label="Your JD attracts:" value={data.jd_attracts} color="#b45309" />
      <Row label="Mismatch:" value={data.mismatch} color="#dc2626" />
      <Row label="Fix:" value={data.fix} color="#166534" />
    </div>
  );
}

function SearchPageChooser({ data, onMarketMap, onPaste }) {
  const act = (id) => {
    if (id === "market_map") onMarketMap();
    else if (id === "paste") onPaste();
  };
  const mkt = data.market && data.market !== "Global" ? ` (${data.market})` : "";
  return (
    <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 16, padding: 24, maxWidth: 760, margin: "0 auto" }}>
      <div style={{ fontSize: 18, fontWeight: 800, color: "#1e293b", marginBottom: 6 }}>
        This looks like a job-search results page
      </div>
      <p style={{ color: "#64748b", fontSize: 14, marginBottom: 18, lineHeight: 1.5 }}>
        {data.message}
        {data.role_hint ? ` Detected category: “${data.role_hint}”${mkt}.` : ""} Choose how to proceed:
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {(data.options || []).map((opt) => (
          <button
            key={opt.id}
            onClick={() => act(opt.id)}
            disabled={opt.id === "pick_job"}
            style={{
              textAlign: "left", padding: "14px 16px", borderRadius: 10,
              border: "1px solid " + (opt.id === "market_map" ? "#6366f1" : "#e2e8f0"),
              background: opt.id === "market_map" ? "#eef2ff" : "#fff",
              color: "#1e293b", fontSize: 14, fontWeight: 600,
              cursor: opt.id === "pick_job" ? "default" : "pointer",
            }}
          >
            {opt.id === "market_map" ? "🗺️ " : opt.id === "paste" ? "📋 " : "🔗 "}
            {opt.label}
            {opt.id === "pick_job" && (
              <div style={{ fontWeight: 400, color: "#94a3b8", fontSize: 12, marginTop: 3 }}>
                Open the results page, click into a specific job, and paste its URL above.
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function PersonaBuilder() {
  const [url, setUrl] = useState("");
  const [pasteMode, setPasteMode] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [lockedCards, setLockedCards] = useState({});
  const [tab, setTab] = useState("map");

  const stepIntervalRef = { current: null };

  const startStepCycle = useCallback(() => {
    let step = 0;
    setLoadingStep(0);
    const id = setInterval(() => {
      step = Math.min(step + 1, LOADING_STEPS.length - 1);
      setLoadingStep(step);
    }, 2200);
    return id;
  }, []);

  const handleGenerate = useCallback(async (modeArg) => {
    const mode = typeof modeArg === "string" ? modeArg : null;
    const hasUrl = url.trim().length > 0;
    const hasText = pasteText.trim().length > 0;

    if (!hasUrl && !hasText) {
      setError("Please enter a job URL or paste job description text.");
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    setLockedCards({});

    const intervalId = startStepCycle();

    try {
      const body = pasteMode && hasText ? { text: pasteText.trim() } : { url: url.trim() };
      if (mode) body.mode = mode;
      const res = await fetch("/api/analyze-jd", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        // Prefer the server's user-facing message from the JSON body
        let msg = "Something went wrong. Please try again.";
        try {
          const errData = await res.json();
          if (errData && errData.error) msg = errData.error;
        } catch {
          /* non-JSON response — keep the generic message */
        }
        if (res.status === 429) {
          msg = "You're sending requests too quickly. Please wait a minute and try again.";
        }
        throw new Error(msg);
      }
      const data = await res.json();
      // Normalize segment percentages
      if (data.personas) {
        data.personas = normalizeSegments(data.personas);
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze job description. Please try again.");
    } finally {
      clearInterval(intervalId);
      setLoading(false);
    }
  }, [url, pasteText, pasteMode, startStepCycle]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && !pasteMode) handleGenerate();
    },
    [handleGenerate, pasteMode]
  );

  const toggleLock = useCallback((idx) => {
    setLockedCards((prev) => ({ ...prev, [idx]: !prev[idx] }));
  }, []);

  return (
    <div style={S.root}>
      {/* Keyframe injection */}
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        @media print {
          /* A persona card taller than a page must NOT be forced whole (that causes
             garbled text at a mid-block break). Instead, let the card flow and keep
             each inner section intact, so any page break lands cleanly BETWEEN blocks. */
          .nova-card { break-inside: auto !important; page-break-inside: auto !important; }
          .nova-cardbody > * { break-inside: avoid !important; page-break-inside: avoid !important; }
          .nova-card, .nova-card * { orphans: 3; widows: 3; }
          /* Clean export: drop all interactive chrome so the PDF is just the map. */
          .no-print { display: none !important; }
          .nova-card button { display: none !important; }   /* lock icons */
          body { background: #fff !important; }
          @page { margin: 12mm; }
        }
      `}</style>

      {/* Header */}
      <header style={S.header}>
        <h1 style={S.headerTitle}>Nova Candidate Map</h1>
        <p style={S.headerSub}>
          AI-powered candidate persona builder — paste a JD URL or text to generate data-grounded hiring personas
        </p>
      </header>

      <main style={S.main}>
        {/* Tab bar */}
        <div className="no-print" style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {[
            ["map", "Persona Map"],
            ["definitions", "Definitions"],
          ].map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              style={{
                padding: "8px 18px",
                borderRadius: 999,
                border: "1px solid " + (tab === key ? "#6366f1" : "#cbd5e1"),
                background: tab === key ? "#6366f1" : "#fff",
                color: tab === key ? "#fff" : "#475569",
                fontWeight: 700,
                fontSize: 14,
                cursor: "pointer",
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {tab === "definitions" ? (
          <DefinitionsPanel />
        ) : (
        <>
        {/* Input Card */}
        <div className="no-print" style={S.inputCard}>
          {/* Toggle */}
          <div style={S.toggleRow}>
            <button
              style={{
                ...S.toggle,
                background: pasteMode ? "#6366f1" : "#cbd5e1",
              }}
              onClick={() => setPasteMode((v) => !v)}
              aria-pressed={pasteMode}
              aria-label="Toggle paste text mode"
            >
              <span
                style={{
                  ...S.toggleThumb,
                  left: pasteMode ? 16 : 2,
                }}
              />
            </button>
            <span style={S.toggleLabel} onClick={() => setPasteMode((v) => !v)}>
              {pasteMode ? "Paste text mode" : "URL mode"} — click to switch
            </span>
          </div>

          <div style={{ marginTop: 14 }}>
            {!pasteMode ? (
              <>
                <label style={S.label} htmlFor="jd-url">
                  Job Description URL
                </label>
                <div style={S.inputRow}>
                  <input
                    id="jd-url"
                    type="url"
                    placeholder="https://careers.example.com/job/12345"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    onKeyDown={handleKeyDown}
                    style={S.input}
                    disabled={loading}
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <button
                    style={{
                      ...S.btnPrimary,
                      opacity: loading ? 0.7 : 1,
                      cursor: loading ? "not-allowed" : "pointer",
                    }}
                    onClick={handleGenerate}
                    disabled={loading}
                  >
                    {loading ? "Analyzing…" : "Generate →"}
                  </button>
                </div>
              </>
            ) : (
              <>
                <label style={S.label} htmlFor="jd-text">
                  Paste Job Description Text
                </label>
                <textarea
                  id="jd-text"
                  placeholder="Paste the full job description here…"
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  style={S.textarea}
                  disabled={loading}
                  spellCheck={false}
                />
                <div style={{ marginTop: 10, display: "flex", justifyContent: "flex-end" }}>
                  <button
                    style={{
                      ...S.btnPrimary,
                      opacity: loading ? 0.7 : 1,
                      cursor: loading ? "not-allowed" : "pointer",
                    }}
                    onClick={handleGenerate}
                    disabled={loading}
                  >
                    {loading ? "Analyzing…" : "Generate →"}
                  </button>
                </div>
              </>
            )}
          </div>

          {error && <div style={S.errorBox}>{error}</div>}
        </div>

        {/* Loading */}
        {loading && <LoadingPanel step={loadingStep} />}

        {/* Results */}
        {result && !loading && result.page_type === "search_results" && (
          <SearchPageChooser
            data={result}
            onMarketMap={() => handleGenerate("market_map")}
            onPaste={() => { setPasteMode(true); setResult(null); }}
          />
        )}

        {result && !loading && result.page_type !== "search_results" && (
          <>
            <div className="no-print" style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
              <button
                onClick={() => window.print()}
                title="Export this map as a PDF"
                style={{
                  display: "inline-flex", alignItems: "center", gap: 8,
                  padding: "9px 18px", borderRadius: 10, border: "none",
                  background: "#6366f1", color: "#fff", fontWeight: 700,
                  fontSize: 14, cursor: "pointer",
                  boxShadow: "0 1px 3px rgba(99,102,241,0.35)",
                }}
              >
                <span style={{ fontSize: 16 }}>⬇</span> Export PDF
              </button>
            </div>
            <Banner data={result} />
            <MismatchBlock data={result.persona_jd_mismatch} />

            {result.personas && result.personas.length > 0 && (
              <>
                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 800,
                    color: "#1e293b",
                    marginBottom: 16,
                  }}
                >
                  Candidate Personas
                  <span
                    style={{
                      marginLeft: 10,
                      fontSize: 13,
                      fontWeight: 600,
                      color: "#94a3b8",
                    }}
                  >
                    ({result.personas.length})
                  </span>
                </div>
                <div style={S.cardsGrid}>
                  {result.personas.map((persona, idx) => (
                    <PersonaCard
                      key={idx}
                      persona={persona}
                      index={idx}
                      locked={!!lockedCards[idx]}
                      onToggleLock={() => toggleLock(idx)}
                    />
                  ))}
                </div>
              </>
            )}

            <JobAdRewritePanel jaw={result.job_ad_rewrite} />
          </>
        )}
        </>
        )}
      </main>
    </div>
  );
}
