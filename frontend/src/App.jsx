import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  ScanSearch,
  BookOpen,
  ShieldCheck,
  Quote,
  ArrowUpRight,
  Clock,
  X,
  CircleCheck,
  TriangleAlert,
} from "lucide-react";

/* =============================================================
   Agentic Research Assistant — frontend
   Single file. Falls back to realistic canned demo data when no
   backend is configured, so it always runs standalone.
   Palette: night reading desk (ink + paper + ochre).
   ============================================================= */

const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --ink:#15171f;
  --ink-raised:#1c1f2a;
  --ink-2:#232734;
  --line:#2b3040;
  --line-soft:#222634;
  --bone:#eceadf;
  --bone-dim:#9a9cab;
  --bone-faint:#6a6d7d;
  --paper:#f6f3ea;
  --paper-ink:#22252e;
  --paper-dim:#6c6f79;
  --paper-line:#e2ddd0;
  --gold:#d3a44b;
  --gold-bright:#e2b45c;
  --gold-soft:rgba(211,164,75,0.13);
  --gold-line:rgba(211,164,75,0.32);
  --sage:#6fa593;
  --sage-soft:rgba(111,165,147,0.14);
  --err:#d97757;
  --err-soft:rgba(217,119,87,0.13);
  --err-line:rgba(217,119,87,0.35);
  --r:14px;
  --r-sm:9px;
  --shadow:0 24px 60px -28px rgba(0,0,0,0.6);
}

*{box-sizing:border-box;margin:0;padding:0}

.ra-root{
  min-height:100vh;
  background:
    radial-gradient(1200px 620px at 78% -8%, rgba(211,164,75,0.10), transparent 60%),
    radial-gradient(900px 560px at 6% 106%, rgba(111,165,147,0.07), transparent 55%),
    var(--ink);
  color:var(--bone);
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  -webkit-font-smoothing:antialiased;
  line-height:1.5;
}

.ra-shell{max-width:1080px;margin:0 auto;padding:0 24px}

/* ---- header ---- */
.ra-head{
  display:flex;align-items:center;justify-content:space-between;
  padding:22px 0 20px;
}
.ra-brand{display:flex;align-items:center;gap:13px;min-width:0}
.ra-mark{
  width:38px;height:38px;border-radius:10px;flex:none;
  border:1px solid var(--gold-line);
  background:linear-gradient(160deg, rgba(211,164,75,0.22), rgba(211,164,75,0.04));
  display:grid;place-items:center;
  position:relative;
}
.ra-mark::before,.ra-mark::after{
  content:"";position:absolute;background:var(--gold);border-radius:2px;
}
.ra-mark::before{width:2px;height:15px;box-shadow:5px 0 0 var(--gold),-5px 0 0 rgba(211,164,75,0.5)}
.ra-mark::after{width:15px;height:2px;top:14px;opacity:0.55}
.ra-title{font-family:'Fraunces',serif;font-weight:600;font-size:19px;letter-spacing:-0.01em;color:var(--bone);white-space:nowrap}
.ra-sub{font-size:11.5px;color:var(--bone-faint);font-family:'JetBrains Mono',monospace;margin-top:1px;letter-spacing:0.02em}

.ra-head-right{display:flex;align-items:center;gap:10px;flex:none}
.ra-pill{
  display:inline-flex;align-items:center;gap:7px;
  font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.03em;
  color:var(--bone-dim);
  border:1px solid var(--line);border-radius:100px;padding:6px 12px;
  background:var(--ink-raised);
}
.ra-dot{width:6px;height:6px;border-radius:50%;background:var(--sage);box-shadow:0 0 0 3px var(--sage-soft)}
.ra-pill.is-live .ra-dot{background:var(--gold);box-shadow:0 0 0 3px var(--gold-soft)}
.ra-icon-btn{
  display:none;align-items:center;justify-content:center;
  width:38px;height:38px;border-radius:10px;
  border:1px solid var(--line);background:var(--ink-raised);color:var(--bone-dim);
  cursor:pointer;transition:color .15s,border-color .15s;
}
.ra-icon-btn:hover{color:var(--bone);border-color:var(--gold-line)}

/* ---- hero / ask ---- */
.ra-hero{padding:38px 0 8px;max-width:720px}
.ra-eyebrow{
  font-family:'JetBrains Mono',monospace;font-size:11.5px;letter-spacing:0.16em;
  text-transform:uppercase;color:var(--gold);margin-bottom:16px;
  display:flex;align-items:center;gap:10px;
}
.ra-eyebrow::before{content:"";width:26px;height:1px;background:var(--gold-line)}
.ra-h1{
  font-family:'Fraunces',serif;font-weight:500;
  font-size:clamp(28px,4.4vw,44px);line-height:1.08;letter-spacing:-0.02em;
  color:var(--bone);
}
.ra-h1 em{font-style:italic;color:var(--gold-bright)}
.ra-lede{margin-top:16px;color:var(--bone-dim);font-size:15.5px;max-width:560px}

/* ---- form ---- */
.ra-form{margin-top:26px;max-width:720px}
.ra-field{
  display:flex;align-items:flex-end;gap:10px;
  background:var(--ink-raised);border:1px solid var(--line);
  border-radius:var(--r);padding:8px 8px 8px 18px;
  transition:border-color .18s, box-shadow .18s;
}
.ra-field:focus-within{border-color:var(--gold-line);box-shadow:0 0 0 4px var(--gold-soft)}
.ra-input{
  flex:1;background:transparent;border:none;outline:none;color:var(--bone);
  font-family:'Inter',sans-serif;font-size:16px;resize:none;
  padding:9px 0;line-height:1.45;max-height:160px;
}
.ra-input::placeholder{color:var(--bone-faint)}
.ra-send{
  flex:none;display:inline-flex;align-items:center;gap:8px;
  background:var(--gold);color:#20180a;font-weight:600;font-size:14px;
  border:none;border-radius:var(--r-sm);padding:11px 16px;cursor:pointer;
  font-family:'Inter',sans-serif;transition:transform .12s, background .15s;
}
.ra-send:hover:not(:disabled){background:var(--gold-bright)}
.ra-send:active:not(:disabled){transform:translateY(1px)}
.ra-send:disabled{opacity:0.5;cursor:not-allowed}
.ra-send .ra-send-label{display:inline}

.ra-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.ra-chip{
  font-size:13px;color:var(--bone-dim);
  background:transparent;border:1px solid var(--line-soft);border-radius:100px;
  padding:7px 13px;cursor:pointer;font-family:'Inter',sans-serif;
  transition:color .15s,border-color .15s,background .15s;
}
.ra-chip:hover{color:var(--bone);border-color:var(--gold-line);background:var(--gold-soft)}

/* ---- agent rail ---- */
.ra-rail-wrap{margin:34px 0 6px;max-width:720px}
.ra-rail{display:flex;align-items:flex-start;gap:0;position:relative}
.ra-node{display:flex;flex-direction:column;align-items:center;gap:10px;flex:none;width:96px;text-align:center;position:relative;z-index:2}
.ra-node-ic{
  width:44px;height:44px;border-radius:12px;display:grid;place-items:center;
  border:1px solid var(--line);background:var(--ink-raised);color:var(--bone-faint);
  transition:all .35s ease;
}
.ra-node-label{
  font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:0.02em;
  color:var(--bone-faint);transition:color .35s;line-height:1.35;
}
.ra-seg{flex:1;height:1px;background:var(--line);margin-top:22px;position:relative;overflow:hidden;min-width:18px}
.ra-seg::after{
  content:"";position:absolute;inset:0;transform:translateX(-101%);
  background:linear-gradient(90deg,transparent,var(--gold),transparent);
}
.ra-node.is-active .ra-node-ic{
  border-color:var(--gold);color:var(--gold-bright);background:var(--gold-soft);
  box-shadow:0 0 0 5px var(--gold-soft);
}
.ra-node.is-active .ra-node-label{color:var(--bone)}
.ra-node.is-done .ra-node-ic{border-color:var(--sage);color:var(--sage);background:var(--sage-soft)}
.ra-node.is-done .ra-node-label{color:var(--bone-dim)}
.ra-seg.is-run::after{animation:railPulse 1.1s ease-in-out infinite}
.ra-seg.is-done::after{transform:translateX(0);background:var(--sage);opacity:0.5}
@keyframes railPulse{0%{transform:translateX(-101%)}60%,100%{transform:translateX(101%)}}

.ra-status{
  margin-top:18px;font-family:'JetBrains Mono',monospace;font-size:12.5px;
  color:var(--bone-dim);display:flex;align-items:center;gap:9px;min-height:18px;
}
.ra-status .ra-run-dot{width:7px;height:7px;border-radius:50%;background:var(--gold);animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:0.25}}

/* ---- results layout ---- */
.ra-results{display:grid;grid-template-columns:1fr 322px;gap:24px;padding:14px 0 60px;align-items:start}
.ra-main{min-width:0}

/* paper card (the answer) */
.ra-paper{
  background:var(--paper);color:var(--paper-ink);
  border-radius:var(--r);padding:30px 32px;box-shadow:var(--shadow);
  position:relative;
}
.ra-paper::before{
  content:"";position:absolute;left:0;top:26px;bottom:26px;width:3px;border-radius:3px;
  background:linear-gradient(var(--gold),rgba(211,164,75,0.15));
}
.ra-q{
  font-family:'JetBrains Mono',monospace;font-size:11.5px;letter-spacing:0.04em;
  text-transform:uppercase;color:var(--paper-dim);margin-bottom:12px;
}
.ra-answer{
  font-family:'Fraunces',serif;font-weight:400;font-size:19px;line-height:1.62;
  color:var(--paper-ink);letter-spacing:-0.005em;
}
.ra-cite{
  display:inline-flex;align-items:center;justify-content:center;
  font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;
  vertical-align:super;line-height:1;
  color:var(--gold);background:var(--gold-soft);
  border:1px solid var(--gold-line);border-radius:5px;
  padding:1px 4px;margin:0 1px;cursor:pointer;transition:all .14s;
  min-width:17px;
}
.ra-cite:hover,.ra-cite:focus-visible{background:var(--gold);color:#20180a;outline:none;transform:translateY(-1px)}

/* structured sections */
.ra-sections{display:grid;gap:2px;margin-top:26px;border-top:1px solid var(--paper-line);padding-top:24px}
.ra-block{padding:14px 0;border-bottom:1px solid var(--paper-line)}
.ra-block:last-child{border-bottom:none}
.ra-block-h{
  display:flex;align-items:center;gap:9px;
  font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--paper-dim);margin-bottom:12px;
}
.ra-block-h .ra-num{color:var(--gold);font-weight:600}
.ra-list{list-style:none;display:grid;gap:9px}
.ra-list li{
  position:relative;padding-left:20px;font-size:14.5px;color:#34373f;line-height:1.5;
}
.ra-list li::before{
  content:"";position:absolute;left:2px;top:9px;width:6px;height:6px;
  border:1.5px solid var(--gold);border-radius:50%;
}
.ra-list.ra-gaps li::before{border-radius:1px;border-color:var(--paper-dim)}

/* evaluator meta strip */
.ra-meta{
  margin-top:24px;display:flex;flex-wrap:wrap;gap:10px 20px;align-items:center;
  padding-top:20px;border-top:1px solid var(--paper-line);
}
.ra-meta-item{display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--paper-dim)}
.ra-meta-item b{color:var(--paper-ink);font-weight:600}
.ra-grounded{color:var(--sage);display:inline-flex;align-items:center;gap:6px;font-weight:600}
.ra-cov{display:flex;align-items:center;gap:9px}
.ra-cov-bar{width:74px;height:5px;border-radius:3px;background:var(--paper-line);overflow:hidden}
.ra-cov-fill{height:100%;background:var(--gold);border-radius:3px;transition:width .8s ease}

/* ---- sources rail ---- */
.ra-side{position:sticky;top:22px}
.ra-side-h{
  display:flex;align-items:center;gap:9px;margin-bottom:14px;
  font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--bone-dim);
}
.ra-side-h .ra-count{margin-left:auto;color:var(--bone-faint)}
.ra-sources{list-style:none;display:grid;gap:10px}
.ra-src{
  background:var(--ink-raised);border:1px solid var(--line-soft);border-radius:var(--r-sm);
  padding:13px 14px;transition:border-color .2s, background .2s, transform .2s;
  scroll-margin-top:22px;
}
.ra-src.is-active{border-color:var(--gold);background:var(--gold-soft)}
.ra-src-top{display:flex;align-items:center;gap:9px;margin-bottom:7px}
.ra-src-marker{
  font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;color:var(--gold);
  border:1px solid var(--gold-line);border-radius:5px;padding:1px 6px;flex:none;
}
.ra-src-title{font-size:13px;font-weight:600;color:var(--bone);line-height:1.3}
.ra-src-path{
  font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--bone-faint);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:7px;
}
.ra-src-prev{font-size:12.5px;color:var(--bone-dim);line-height:1.45;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}

/* history */
.ra-hist{margin-top:22px;border-top:1px solid var(--line-soft);padding-top:18px}
.ra-hist-item{
  display:flex;align-items:center;gap:10px;width:100%;text-align:left;
  background:transparent;border:none;color:var(--bone-dim);cursor:pointer;
  padding:9px 0;font-family:'Inter',sans-serif;font-size:13px;line-height:1.35;
  border-bottom:1px solid var(--line-soft);transition:color .15s;
}
.ra-hist-item:last-child{border-bottom:none}
.ra-hist-item:hover{color:var(--bone)}
.ra-hist-item .ra-hist-ic{color:var(--bone-faint);flex:none}
.ra-hist-item span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* empty history */
.ra-empty{font-size:13px;color:var(--bone-faint);padding:6px 0;line-height:1.5}

/* skeleton */
.ra-sk{background:var(--paper);border-radius:var(--r);padding:30px 32px;box-shadow:var(--shadow)}
.ra-sk-line{height:15px;border-radius:5px;background:linear-gradient(90deg,#e9e4d8 25%,#efeadf 37%,#e9e4d8 63%);background-size:400% 100%;animation:sh 1.4s ease infinite;margin-bottom:13px}
@keyframes sh{0%{background-position:100% 0}100%{background-position:-100% 0}}

/* error card */
.ra-err{
  background:var(--paper);color:var(--paper-ink);
  border-radius:var(--r);padding:26px 28px;box-shadow:var(--shadow);
  display:flex;gap:14px;align-items:flex-start;
}
.ra-err-ic{color:var(--err);flex:none;margin-top:2px}
.ra-err-h{font-family:'Fraunces',serif;font-weight:600;font-size:16.5px;margin-bottom:6px}
.ra-err-msg{font-size:13.5px;color:var(--paper-dim);line-height:1.5;font-family:'JetBrains Mono',monospace}
.ra-err-retry{
  margin-top:14px;background:transparent;border:1px solid var(--err-line);color:var(--err);
  border-radius:var(--r-sm);padding:8px 14px;font-size:13px;font-weight:600;cursor:pointer;
  font-family:'Inter',sans-serif;transition:background .15s;
}
.ra-err-retry:hover{background:var(--err-soft)}

/* footer */
.ra-foot{
  border-top:1px solid var(--line-soft);padding:20px 0 40px;margin-top:8px;
  display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;
  font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--bone-faint);
}
.ra-foot a{color:var(--bone-dim);text-decoration:none;display:inline-flex;align-items:center;gap:5px;transition:color .15s}
.ra-foot a:hover{color:var(--gold)}
.ra-foot .ra-foot-sep{opacity:0.4}

/* fade-in */
.ra-fade{animation:fade .5s ease both}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

/* drawer scrim (mobile history) */
.ra-scrim{position:fixed;inset:0;background:rgba(8,9,13,0.6);backdrop-filter:blur(2px);z-index:40;animation:fade .25s ease both}
.ra-drawer{
  position:fixed;top:0;right:0;bottom:0;width:min(340px,88vw);z-index:50;
  background:var(--ink);border-left:1px solid var(--line);padding:22px;
  overflow-y:auto;animation:slideIn .28s cubic-bezier(.2,.7,.2,1) both;
}
@keyframes slideIn{from{transform:translateX(100%)}to{transform:none}}
.ra-drawer-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.ra-drawer-h h3{font-family:'Fraunces',serif;font-size:17px;font-weight:600;color:var(--bone)}
.ra-drawer-close{background:transparent;border:none;color:var(--bone-dim);cursor:pointer;padding:6px}

/* ---- responsive ---- */
.ra-mobile-sources{display:none}
@media (max-width:900px){
  .ra-results{grid-template-columns:1fr;gap:20px}
  .ra-side{position:static}
  .ra-desktop-only{display:none}
  .ra-mobile-sources{display:block;margin-top:0}
  .ra-icon-btn{display:inline-flex}
}
@media (max-width:560px){
  .ra-shell{padding:0 16px}
  .ra-paper{padding:24px 20px}
  .ra-sk{padding:24px 20px}
  .ra-node{width:auto;flex:1}
  .ra-node-label{font-size:9.5px}
  .ra-send .ra-send-label{display:none}
  .ra-send{padding:11px}
  .ra-sub{display:none}
  .ra-answer{font-size:17.5px}
}

@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.001ms!important;animation-iteration-count:1!important;transition-duration:.001ms!important}
  .ra-seg::after{display:none}
}
`;

/* =============================================================
   Live backend config (see frontend/.env.example)
   ============================================================= */

const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");
const API_KEY = import.meta.env.VITE_API_KEY || "";
const LIVE = Boolean(API_URL);

/* =============================================================
   MOCK DATA (offline fallback when no backend is configured)
   ============================================================= */

const CANNED = {
  grounding: {
    question: "What methods do these papers use for grounding, and what gaps remain?",
    answer:
      "Across these papers, grounding is handled less by any single trick and more by tightening the link between what the model reads and what it is allowed to say. [1] A dense retriever trained with a contrastive objective supplies passages that are semantically close to the query rather than keyword matches, which raises the odds that the evidence on hand is actually relevant. [2] On top of retrieval, a verification pass re-checks each drafted claim against its cited passage before the answer is returned and drops anything it cannot support. [3] The reported effect is a sharp drop in unsupported claims against a single-pass baseline, paid for with extra latency from the checking round. [4]",
    methods: [
      "Contrastive dense retrieval over passage embeddings",
      "Post-hoc claim verification against each cited passage",
      "Retrieval quality measured with recall@k and mean reciprocal rank",
    ],
    findings: [
      "Contrastive retrieval lifts recall@5 to 0.79, above BM25 (0.62) and a general dense encoder (0.71)",
      "A verification round removes roughly 63% of unsupported claims",
      "Citation precision reaches 92%, versus 61% for the single-pass system",
    ],
    gaps: [
      "No ablation isolates how much retrieval versus verification each contributes",
      "Results are reported only on the studied corpora, so generalization is untested",
      "The verifier still misses subtle unsupported inferences",
    ],
    sources: [
      { m: 1, t: "Dense Retrieval with Contrastive Pretraining for Literature Search", p: "examples/sample_papers/03_context_retriever.txt", x: "Effective retrieval is the foundation of any grounded question-answering system. We present ConText, a dense passage retriever pretrained with a contrastive objective over 8 million scientific abstracts." },
      { m: 2, t: "Dense Retrieval with Contrastive Pretraining for Literature Search", p: "examples/sample_papers/03_context_retriever.txt", x: "We index for exact search and evaluate approximate variants for scale. Retrieval quality is measured with recall@k and mean reciprocal rank. ConText raises recall@5 to 0.79." },
      { m: 3, t: "Retrieval-Augmented Generation for Scientific Question Answering", p: "examples/sample_papers/01_scirag.txt", x: "Over 1,200 expert-written questions, SciRAG reduces unsupported claims by 63% relative to a non-retrieval baseline while improving answer correctness. A verification round checks each claim." },
      { m: 4, t: "Multi-Agent Reflection Improves Long-Form Reasoning", p: "examples/sample_papers/02_reflectgraph.txt", x: "92% of citations are judged relevant, compared with 61% for the single-pass system. Latency increases by roughly 1.7x due to the verification round." },
    ],
    coverage: 0.85,
    iterations: 2,
    sufficiency: "sufficient",
  },
  evaluation: {
    question: "How do these papers evaluate their systems, and where do the setups disagree?",
    answer:
      "The evaluations share a backbone but diverge on what counts as success. [1] All three report retrieval quality with recall@k and rank-based metrics, which makes the retrieval stage roughly comparable across papers. [2] Where they part ways is the answer itself: one grades correctness against expert-written references, while another leans on a model-judged citation-relevance score that is cheaper but softer. [3] None of them run a shared human evaluation, so the headline numbers are not strictly comparable across the three setups. [4]",
    methods: [
      "Recall@k and mean reciprocal rank for the retrieval stage",
      "Expert-written reference answers for correctness (one paper)",
      "Model-judged citation relevance as a proxy metric (another paper)",
    ],
    findings: [
      "Retrieval metrics are consistent enough to compare across papers",
      "Answer-quality metrics differ: reference-based versus model-judged",
      "Reported latency ranges from near-baseline to 1.7x depending on the verification step",
    ],
    gaps: [
      "No shared human evaluation across the three systems",
      "Model-judged scores may inflate agreement with the system under test",
      "Corpora differ, so absolute scores are not directly comparable",
    ],
    sources: [
      { m: 1, t: "Retrieval-Augmented Generation for Scientific Question Answering", p: "examples/sample_papers/01_scirag.txt", x: "We evaluate over 1,200 expert-written questions. Retrieval is scored with recall@k; answers are graded against reference responses for correctness." },
      { m: 2, t: "Dense Retrieval with Contrastive Pretraining for Literature Search", p: "examples/sample_papers/03_context_retriever.txt", x: "Retrieval quality is measured with recall@k and mean reciprocal rank across held-out queries." },
      { m: 3, t: "Multi-Agent Reflection Improves Long-Form Reasoning", p: "examples/sample_papers/02_reflectgraph.txt", x: "A model judge scores citation relevance. This is cheaper than human annotation but correlates imperfectly with expert judgment." },
      { m: 4, t: "Retrieval-Augmented Generation for Scientific Question Answering", p: "examples/sample_papers/01_scirag.txt", x: "We note that cross-system comparison is limited: the corpora and question sets differ between published setups." },
    ],
    coverage: 0.82,
    iterations: 2,
    sufficiency: "sufficient",
  },
};

const EXAMPLES = [
  { label: "Methods used for grounding", key: "grounding" },
  { label: "Compare their evaluation setups", key: "evaluation" },
  { label: "What are the open research gaps?", key: "grounding" },
];

// Build a plausible result for any free-typed question (keeps the offline demo alive).
function genericResult(question) {
  return {
    question,
    answer:
      "Answering from the indexed passages, the papers converge on retrieval-plus-verification as the core pattern for this question. [1] Relevant evidence is pulled first, then a drafted answer is checked against its sources before anything is returned. [2] The reported gains are consistent, though each paper measures them a little differently, so treat the exact figures as indicative rather than head-to-head. [3]",
    methods: [
      "Retrieval over passage embeddings for the evidence set",
      "A verification pass over each cited claim",
      "Standard retrieval and answer-quality metrics",
    ],
    findings: [
      "Retrieval-grounded answers reduce unsupported claims against a non-retrieval baseline",
      "Citation precision improves once a verification round is added",
      "Added checking costs latency, reported in the ~1.5-1.7x range",
    ],
    gaps: [
      "Metrics differ between papers, limiting direct comparison",
      "Generalization beyond the studied corpora is untested",
    ],
    sources: [
      { m: 1, t: "Retrieval-Augmented Generation for Scientific Question Answering", p: "examples/sample_papers/01_scirag.txt", x: "SciRAG retrieves supporting passages before generation and verifies each claim against its citation." },
      { m: 2, t: "Dense Retrieval with Contrastive Pretraining for Literature Search", p: "examples/sample_papers/03_context_retriever.txt", x: "ConText supplies semantically relevant passages, raising recall@5 to 0.79 over keyword and general dense baselines." },
      { m: 3, t: "Multi-Agent Reflection Improves Long-Form Reasoning", p: "examples/sample_papers/02_reflectgraph.txt", x: "A reflection round re-checks drafted claims, improving citation relevance at some latency cost." },
    ],
    coverage: 0.78,
    iterations: 1,
    sufficiency: "sufficient",
  };
}

function shape(src, question) {
  return {
    question: question || src.question,
    answer: src.answer,
    methods: src.methods,
    findings: src.findings,
    gaps: src.gaps,
    sources: src.sources,
    coverage: src.coverage,
    iterations: src.iterations,
    sufficiency: src.sufficiency,
    grounded: true,
  };
}

// Map the FastAPI /research response (see ResearchResult in schemas.py) onto
// the shape the UI renders.
function shapeApiResult(data, question) {
  const evaluation = data.evaluation || {};
  return {
    question: data.question || question,
    answer: data.summary?.summary || "",
    methods: data.summary?.methods || [],
    findings: data.summary?.key_findings || [],
    gaps: data.summary?.research_gaps || [],
    sources: (data.sources || []).map((s) => ({
      m: s.marker,
      t: s.title,
      p: s.source,
      x: s.preview,
    })),
    coverage: evaluation.coverage_score ?? 0,
    iterations: data.iterations ?? 1,
    sufficiency: evaluation.sufficiency || "sufficient",
    grounded: evaluation.grounded ?? true,
  };
}

/* ── data source ───────────────────────────────────────────────
   With VITE_API_URL set (see frontend/.env.example), questions are
   sent to the real deployed pipeline. Otherwise this runs entirely
   on baked-in demo data, no backend required.

   Note: the live free-tier backend sleeps when idle, so the first
   call after a while can take 30-60s to wake up.
   ──────────────────────────────────────────────────────────── */
async function runResearch(question) {
  const res = await fetch(`${API_URL}/research`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
    },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Request failed (${res.status}). ${detail || res.statusText}`.trim());
  }
  const data = await res.json();
  return shapeApiResult(data, question);
}

function runDemo(question, presetKey) {
  const q = (question || "").toLowerCase();
  let src;
  if (presetKey && CANNED[presetKey]) src = CANNED[presetKey];
  else if (q.includes("eval") || q.includes("compare") || q.includes("metric")) src = CANNED.evaluation;
  else if (q.includes("ground") || q.includes("cit") || q.includes("gap")) src = CANNED.grounding;
  return src ? shape(src, question) : genericResult(question);
}

/* =============================================================
   Small pieces
   ============================================================= */

// Render answer text, turning [n] into interactive citation chips.
function AnswerText({ text, onCite, onCiteLeave }) {
  const parts = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  let m;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(<span key={`t${i}`}>{text.slice(last, m.index)}</span>);
    const n = parseInt(m[1], 10);
    parts.push(
      <button
        key={`c${i}`}
        className="ra-cite"
        onMouseEnter={() => onCite(n)}
        onMouseLeave={onCiteLeave}
        onFocus={() => onCite(n)}
        onBlur={onCiteLeave}
        onClick={() => onCite(n)}
        aria-label={`Source ${n}`}
      >
        {n}
      </button>
    );
    last = m.index + m[0].length;
    i++;
  }
  if (last < text.length) parts.push(<span key="tend">{text.slice(last)}</span>);
  return <p className="ra-answer">{parts}</p>;
}

const STAGES = [
  { id: "retrieve", label: "Retrieving\nsources", Icon: ScanSearch },
  { id: "read", label: "Reading &\ndrafting", Icon: BookOpen },
  { id: "check", label: "Checking\ngrounding", Icon: ShieldCheck },
];

function AgentRail({ stage }) {
  // stage: -1 idle, 0..2 active index, 3 done
  return (
    <div className="ra-rail" role="img" aria-label="Agent progress">
      {STAGES.map((s, idx) => {
        const active = stage === idx;
        const done = stage > idx;
        const cls = `ra-node${active ? " is-active" : ""}${done ? " is-done" : ""}`;
        return (
          <React.Fragment key={s.id}>
            <div className={cls}>
              <div className="ra-node-ic">
                <s.Icon size={19} strokeWidth={1.9} />
              </div>
              <div className="ra-node-label">
                {s.label.split("\n").map((l, k) => (
                  <div key={k}>{l}</div>
                ))}
              </div>
            </div>
            {idx < STAGES.length - 1 && (
              <div className={`ra-seg${stage === idx ? " is-run" : ""}${stage > idx ? " is-done" : ""}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="ra-sk" aria-hidden="true">
      {[92, 100, 88, 96, 60].map((w, i) => (
        <div key={i} className="ra-sk-line" style={{ width: `${w}%` }} />
      ))}
    </div>
  );
}

/* =============================================================
   App
   ============================================================= */

export default function App() {
  const [value, setValue] = useState("");
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [stage, setStage] = useState(-1);
  const [statusText, setStatusText] = useState("");
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [active, setActive] = useState(null); // active citation marker
  const [history, setHistory] = useState([]);
  const [drawer, setDrawer] = useState(false);
  const [pending, setPending] = useState(null); // { q, presetKey } for retry

  const taRef = useRef(null);
  const timers = useRef([]);

  const clearTimers = () => {
    timers.current.forEach((t) => clearTimeout(t) || clearInterval(t));
    timers.current = [];
  };
  useEffect(() => () => clearTimers(), []);

  const autosize = (el) => {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  const recordHistory = (q, presetKey) => {
    setHistory((h) => {
      const next = [{ q, key: presetKey || null }, ...h.filter((it) => it.q !== q)];
      return next.slice(0, 6);
    });
  };

  const ask = useCallback(async (question, presetKey) => {
    const q = question.trim();
    if (!q || phase === "running") return;
    clearTimers();
    setDrawer(false);
    setActive(null);
    setResult(null);
    setErrorMsg("");
    setPhase("running");
    setPending({ q, presetKey });

    if (!LIVE) {
      // Offline demo: a scripted timeline so the agent loop feels real.
      const data = runDemo(q, presetKey);
      const willLoop = data.iterations > 1;
      const seq = [
        [0, "Searching the index for relevant passages"],
        [1, "Drafting a grounded answer with citations"],
        [2, "Checking each claim against its source"],
      ];
      const loopBeat = willLoop
        ? [[0, "Coverage was thin, refining the query"], [1, "Revising the draft with new evidence"], [2, "Re-checking grounding"]]
        : [];
      const full = [...seq, ...loopBeat];
      const stepMs = 720;
      full.forEach(([st, msg], i) => {
        timers.current.push(
          setTimeout(() => {
            setStage(st);
            setStatusText(msg);
          }, i * stepMs)
        );
      });
      const endAt = full.length * stepMs + 260;
      timers.current.push(
        setTimeout(() => {
          setStage(3);
          setStatusText("");
          setResult(data);
          setPhase("done");
          recordHistory(q, presetKey);
        }, endAt)
      );
      return;
    }

    // Live backend: we don't know real timing in advance, so cycle the
    // agent rail through its stages while the request is in flight.
    setStage(0);
    setStatusText("Searching the index for relevant passages");
    const liveMessages = [
      "Searching the index for relevant passages",
      "Drafting a grounded answer with citations",
      "Checking each claim against its source",
    ];
    let cycle = 0;
    const interval = setInterval(() => {
      cycle = (cycle + 1) % 3;
      setStage(cycle);
      setStatusText(liveMessages[cycle]);
    }, 1400);
    timers.current.push(interval);

    try {
      const data = await runResearch(q);
      clearTimers();
      setStage(3);
      setStatusText("");
      setResult(data);
      setPhase("done");
      recordHistory(q, presetKey);
    } catch (err) {
      clearTimers();
      setPhase("error");
      setErrorMsg(err?.message || "Something went wrong reaching the research API.");
    }
  }, [phase]);

  const onSubmit = (e) => {
    e.preventDefault();
    ask(value);
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(value);
    }
  };

  const pickExample = (ex) => {
    setValue(ex.label + "?");
    ask(ex.label + "?", ex.key);
  };

  const rerun = (item) => {
    setValue(item.q);
    ask(item.q, item.key);
  };

  const retry = () => {
    if (pending) ask(pending.q, pending.presetKey);
  };

  // highlight + scroll to active source
  useEffect(() => {
    if (active == null) return;
    const el = document.getElementById(`src-${active}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [active]);

  useEffect(() => {
    const onEsc = (e) => e.key === "Escape" && setDrawer(false);
    window.addEventListener("keydown", onEsc);
    return () => window.removeEventListener("keydown", onEsc);
  }, []);

  const showResults = phase === "running" || phase === "done" || phase === "error";

  const Sources = ({ list }) => (
    <div className="ra-side-h">
      <Quote size={13} strokeWidth={2} />
      Sources
      <span className="ra-count">{list ? list.length : 0}</span>
    </div>
  );

  const SourceList = ({ list }) => (
    <ol className="ra-sources">
      {list.map((s) => (
        <li
          key={s.m}
          id={`src-${s.m}`}
          className={`ra-src${active === s.m ? " is-active" : ""}`}
          onMouseEnter={() => setActive(s.m)}
          onMouseLeave={() => setActive(null)}
        >
          <div className="ra-src-top">
            <span className="ra-src-marker">{s.m}</span>
            <span className="ra-src-title">{s.t}</span>
          </div>
          <div className="ra-src-path">{s.p}</div>
          <div className="ra-src-prev">{s.x}</div>
        </li>
      ))}
    </ol>
  );

  const HistoryList = () =>
    history.length === 0 ? (
      <p className="ra-empty">Questions you ask show up here. Nothing yet.</p>
    ) : (
      <div>
        {history.map((it, i) => (
          <button key={i} className="ra-hist-item" onClick={() => rerun(it)}>
            <Clock size={13} className="ra-hist-ic" />
            <span>{it.q}</span>
          </button>
        ))}
      </div>
    );

  return (
    <div className="ra-root">
      <style>{STYLES}</style>

      <div className="ra-shell">
        <header className="ra-head">
          <div className="ra-brand">
            <div className="ra-mark" aria-hidden="true" />
            <div>
              <div className="ra-title">Agentic Research Assistant</div>
              <div className="ra-sub">retriever · summarizer · evaluator</div>
            </div>
          </div>
          <div className="ra-head-right">
            <span className={`ra-pill${LIVE ? " is-live" : ""}`}>
              <span className="ra-dot" />
              {LIVE ? "live api" : "offline demo"}
            </span>
            <button className="ra-icon-btn" onClick={() => setDrawer(true)} aria-label="Open history">
              <Clock size={18} />
            </button>
          </div>
        </header>

        {!showResults && (
          <section className="ra-hero ra-fade">
            <div className="ra-eyebrow">Ask across the papers</div>
            <h1 className="ra-h1">
              Ask a research question. Get an answer that <em>cites its sources.</em>
            </h1>
            <p className="ra-lede">
              Three agents work the question in a loop: one finds relevant passages, one drafts a grounded
              answer, and one checks that answer and sends it back for another pass if the evidence is thin.
            </p>
          </section>
        )}

        <form className="ra-form" onSubmit={onSubmit}>
          <div className="ra-field">
            <textarea
              ref={taRef}
              className="ra-input"
              rows={1}
              placeholder="Ask something about the papers…"
              value={value}
              onChange={(e) => {
                setValue(e.target.value);
                autosize(e.target);
              }}
              onKeyDown={onKey}
              aria-label="Research question"
            />
            <button className="ra-send" type="submit" disabled={!value.trim() || phase === "running"}>
              <Send size={16} strokeWidth={2} />
              <span className="ra-send-label">{phase === "running" ? "Working" : "Ask"}</span>
            </button>
          </div>
          {!showResults && (
            <div className="ra-chips">
              {EXAMPLES.map((ex, i) => (
                <button key={i} type="button" className="ra-chip" onClick={() => pickExample(ex)}>
                  {ex.label}
                </button>
              ))}
            </div>
          )}
        </form>

        {showResults && phase !== "error" && (
          <div className="ra-rail-wrap ra-fade">
            <AgentRail stage={stage} />
            <div className="ra-status" aria-live="polite">
              {phase === "running" && <span className="ra-run-dot" />}
              {statusText ||
                (phase === "done" && result
                  ? `Done in ${result.iterations} ${result.iterations === 1 ? "pass" : "passes"}.`
                  : "")}
            </div>
          </div>
        )}

        {showResults && phase === "error" && (
          <div className="ra-results" style={{ gridTemplateColumns: "1fr" }}>
            <div className="ra-err ra-fade">
              <TriangleAlert size={20} strokeWidth={2} className="ra-err-ic" />
              <div>
                <div className="ra-err-h">The research API didn't respond</div>
                <div className="ra-err-msg">{errorMsg}</div>
                <button className="ra-err-retry" onClick={retry}>
                  Try again
                </button>
              </div>
            </div>
          </div>
        )}

        {showResults && phase !== "error" && (
          <div className="ra-results">
            <main className="ra-main">
              {phase === "running" || !result ? (
                <Skeleton />
              ) : (
                <article className="ra-paper ra-fade">
                  <div className="ra-q">{result.question}</div>
                  <AnswerText
                    text={result.answer}
                    onCite={(n) => setActive(n)}
                    onCiteLeave={() => setActive(null)}
                  />

                  <div className="ra-sections">
                    <div className="ra-block">
                      <div className="ra-block-h">
                        <span className="ra-num">01</span> Methods
                      </div>
                      <ul className="ra-list">
                        {result.methods.map((x, i) => (
                          <li key={i}>{x}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="ra-block">
                      <div className="ra-block-h">
                        <span className="ra-num">02</span> Key findings
                      </div>
                      <ul className="ra-list">
                        {result.findings.map((x, i) => (
                          <li key={i}>{x}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="ra-block">
                      <div className="ra-block-h">
                        <span className="ra-num">03</span> Research gaps
                      </div>
                      <ul className="ra-list ra-gaps">
                        {result.gaps.map((x, i) => (
                          <li key={i}>{x}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  <div className="ra-meta">
                    <span className="ra-meta-item ra-grounded">
                      <CircleCheck size={14} strokeWidth={2.4} />
                      {result.grounded ? "grounded" : "not fully grounded"}
                    </span>
                    <span className="ra-meta-item ra-cov">
                      coverage
                      <span className="ra-cov-bar">
                        <span className="ra-cov-fill" style={{ width: `${Math.round(result.coverage * 100)}%` }} />
                      </span>
                      <b>{result.coverage.toFixed(2)}</b>
                    </span>
                    <span className="ra-meta-item">
                      sufficiency&nbsp;<b>{result.sufficiency}</b>
                    </span>
                    <span className="ra-meta-item">
                      passes&nbsp;<b>{result.iterations}</b>
                    </span>
                  </div>
                </article>
              )}

              {/* mobile: sources under the answer (history lives in the drawer) */}
              <div className="ra-side ra-mobile-sources" style={{ marginTop: 24 }}>
                {result && (
                  <>
                    <Sources list={result.sources} />
                    <SourceList list={result.sources} />
                  </>
                )}
              </div>
            </main>

            <aside className="ra-side ra-desktop-only">
              {result && (
                <>
                  <Sources list={result.sources} />
                  <SourceList list={result.sources} />
                </>
              )}
              <div className="ra-hist">
                <div className="ra-side-h">
                  <Clock size={13} strokeWidth={2} />
                  History
                </div>
                <HistoryList />
              </div>
            </aside>
          </div>
        )}

        <footer className="ra-foot">
          <span>{LIVE ? "Live API" : "Demo data · offline"}</span>
          <span className="ra-foot-sep">/</span>
          <a href="https://agentic-research-assistant-xzno.onrender.com/docs" target="_blank" rel="noreferrer">
            Live API <ArrowUpRight size={13} />
          </a>
          <span className="ra-foot-sep">/</span>
          <a href="https://github.com/bharathkumardev1/Agentic-Research-Assistant" target="_blank" rel="noreferrer">
            Source <ArrowUpRight size={13} />
          </a>
        </footer>
      </div>

      {/* mobile history drawer */}
      {drawer && (
        <>
          <div className="ra-scrim" onClick={() => setDrawer(false)} />
          <div className="ra-drawer" role="dialog" aria-label="History">
            <div className="ra-drawer-h">
              <h3>History</h3>
              <button className="ra-drawer-close" onClick={() => setDrawer(false)} aria-label="Close">
                <X size={18} />
              </button>
            </div>
            <HistoryList />
          </div>
        </>
      )}
    </div>
  );
}
