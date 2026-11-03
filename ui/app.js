"use strict";
// Agentic Black-Litterman debate viewer. Pure vanilla JS; rebuilds a structured
// model from the ordered event stream (replay) or a live WebSocket, and renders
// the per-ticker/per-round timeline, agent cards, influence flow, convergence,
// outcome panel, and cross-month backtest views.

const AGENTS = ["quant", "fundamental", "sentiment", "economist"];
const SINGLE_MODEL_AGENT = "single_gemini";
const SINGLE_MODEL_AGENTS = ["single_gemma", "single_gemini"];
const COLOR = {quant:"var(--quant)",fundamental:"var(--fundamental)",sentiment:"var(--sentiment)",economist:"var(--economist)",moderator:"var(--moderator)",single_gemma:"var(--sentiment)",single_gemini:"var(--accent)"};
const $ = s => document.querySelector(s);
const el = (t,c,h)=>{const e=document.createElement(t); if(c)e.className=c; if(h!=null)e.innerHTML=h; return e;};
const fmt = (x,d=4)=> (x==null||isNaN(x))?"—":Number(x).toFixed(d);
const pct = x => (x==null||isNaN(x))?"—":(Number(x)*100).toFixed(2)+"%";
const money = x => (x==null||isNaN(x))?"—":(Number(x)<0?"-$":"$")+Math.abs(Number(x)).toLocaleString(undefined,{maximumFractionDigits:0});
// LLM- and news-generated strings must be escaped before being injected into
// innerHTML. Without this, a single "<" in a transcript line can break the UI.
const esc = x => String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const nl2br = x => esc(x).replace(/\n/g, "<br>");
const singleAgentFor = tk => SINGLE_MODEL_AGENTS.find(a => tk?.signals?.[a]);
const singleAgentLabel = a => a === "single_gemma" ? "Gemma single" : "Gemini single";

function parseUniverseInput(){
  const raw = $("#universe")?.value || "";
  return raw
    .split(",")
    .map(x => x.trim().toUpperCase())
    .filter(Boolean);
}

function universeLabel(universe){
  return (universe || []).join(", ");
}

let STATE = {
  events: [],
  seek: 0,
  ticker: null,
  model: null,
  playing: false,
  isBacktest: false,
  logmeta: {},
  backtest: {
    name: null,
    runs: []
  },
  live: {
    runId: null,
    lastSeq: 0,
    timer: null,
    status: "idle"
  }
};// Resolve API paths relative to where the page is served, so the UI works behind a
// path-prefix proxy (e.g. Colab's serve_kernel_port) as well as at the root.
const BASE = location.pathname.replace(/[^/]*$/, "");
const api = p => BASE + p;

// ---------------------------------------------------------------- data load
async function loadList(){
  const runs = await (await fetch(api("api/runs"))).json();
  const sel = $("#runsel"); sel.innerHTML="";
  runs.forEach(r=>{const o=el("option"); o.value=r.name; o.textContent=r.name; sel.appendChild(o);});
  const bts = await (await fetch(api("api/backtests"))).json();
  const bsel = $("#btsel"); bsel.innerHTML="";
  bts.forEach(r=>{const o=el("option"); o.value=r.name; o.textContent=r.name; bsel.appendChild(o);});
}

async function loadConfigDefaults(){
  try{
    const cfg = await (await fetch(api("api/config"))).json();

    if(cfg.default_start && $("#start")) $("#start").value = cfg.default_start;
    if(cfg.default_end && $("#end")) $("#end").value = cfg.default_end;

    if(cfg.universe && $("#universe")){
      $("#universe").value = cfg.universe.join(", ");
    }
    if($("#runmode")){
      $("#runmode").value = cfg.run_single_gemini_c1 ? "gemini_c1" : "gemma_debate";
      $("#runmode").title = "Choose a matched model/architecture configuration or the complete 2x2 factorial run";
    }
    if($("#geminic1")){
      $("#geminic1").checked = !!cfg.run_single_gemini_c1;
      $("#geminic1").title = `Run single-agent Gemini C1 only (${cfg.gemini_model || "configured model"}); this skips the Gemma multi-agent graph`;
    }
  }catch(e){
    console.warn("Could not load config defaults", e);
  }
}

async function loadRun(name){
  const log = await (await fetch(api("api/runs/"+encodeURIComponent(name)))).json();
  const prov = (log.meta||{}).provider || "?";
  const anyMock = log.meta?.strict_model===false;
  $("#runmeta").textContent = `${log.portfolio_id} · ${log.as_of_date} · ${prov} data · ${log.events.length} events`;
  STATE.logmeta = log.meta || {};
  startReplay(log.events);
}

// ---------------------------------------------------------------- replay
function startReplay(events){
  STATE.events = events; STATE.seek = events.length; STATE.ticker=null; STATE.playing=false;
  $("#scrubbar").style.display="flex";
  const sk=$("#seek"); sk.max=events.length; sk.value=events.length;
  rebuild();
}
function rebuild(){
  STATE.model = buildModel(STATE.events.slice(0, STATE.seek));
  const last = STATE.events[STATE.seek-1];
  $("#narr").textContent = last ? `#${last.seq} ${last.phase}/${last.type} ${last.ticker||""} ${last.round?("r"+last.round):""} ${last.agent||""}` : "(start)";
  if(!STATE.ticker || !STATE.model.tickers[STATE.ticker]) STATE.ticker = chooseDefaultTicker(STATE.model);
  renderDebate(); renderOutcome();
}

function chooseDefaultTicker(m){
  const ks = Object.keys(m.tickers);
  if(!ks.length) return null;
  // Prefer a ticker that actually debated. Saved runs often start with a
  // non-contested ticker, which made the transcript panel look empty.
  return ks.find(t => m.tickers[t].hasDebate) ||
         ks.find(t => (m.tickers[t].transcript || []).length > 1) ||
         ks[0];
}
function step(d){ STATE.seek=Math.max(0,Math.min(STATE.events.length, STATE.seek+d)); $("#seek").value=STATE.seek; rebuild(); }

// ---------------------------------------------------------------- model
function buildModel(events){
  const m = {tickers:{}, validator:{}, optimized:null, finalViews:{}, meta:{}, mock:{}, warnings:[], maxRound:0, singleModelMeta:null, singleModelPairwiseViews:[]};
  const T = t => (m.tickers[t] ||= {signals:{}, baseline:null, rounds:{}, consensus:null, final:null, contested:false, transcript:[], hasDebate:false, pairwiseViews:[]});
  for(const e of events){
    const p=e.payload||{};
    if(e.type==="run_started"){ m.meta=p; }
    if(e.type==="warning"){ m.warnings.push(p.message||JSON.stringify(p)); }
    if(e.type==="signal_emitted"&&e.ticker){ T(e.ticker).signals[e.agent]=p; if(p.used_mock)m.mock[e.agent]=true; }
    if(e.type==="single_model_result"){
      const a=e.agent||p.agent_name||SINGLE_MODEL_AGENT;
      m.singleModelMeta={...p,agent_name:a}; if(p.used_mock)m.mock[a]=true;
    }
    if(e.type==="single_model_emitted"&&e.ticker){
      const tk=T(e.ticker);
      const a=e.agent||SINGLE_MODEL_AGENT;
      tk.signals[a]=p;
      if(p.used_mock)m.mock[a]=true;
      tk.transcript.push({kind:"single_model_signal", seq:e.seq, round:0, agent:a, direction:p.direction, q:null, message:p.rationale||`${singleAgentLabel(a)} emitted ${p.signal_label||p.direction||"neutral"}.`});
    }
    if(e.type==="single_model_pairwise_view"){
      const a=e.agent||m.singleModelMeta?.agent_name||SINGLE_MODEL_AGENT;
      m.singleModelPairwiseViews.push({...p,agent:a}); if(p.used_mock)m.mock[a]=true;
      const longT=p.long_ticker, shortT=p.short_ticker;
      const msg=`${singleAgentLabel(a)} pairwise BL view: ${longT} should outperform ${shortT} by q=${fmt(p.q,3)} annualized confidence=${fmt(p.confidence,2)}. ${p.rationale||""}`;
      if(longT){ const tk=T(longT); tk.pairwiseViews.push(p); tk.transcript.push({kind:"single_model_pair", seq:e.seq, round:0, agent:a, direction:"long", q:p.q, message:msg}); }
      if(shortT){ const tk=T(shortT); tk.pairwiseViews.push(p); tk.transcript.push({kind:"single_model_pair", seq:e.seq+0.001, round:0, agent:a, direction:"short", q:-p.q, message:msg}); }
    }
    if(e.type==="baseline_view"&&e.ticker){ T(e.ticker).baseline=p; T(e.ticker).contested=!!p.contested; }
    // Newer runs emit explicit agent_message events. Older/sparse runs may not;
    // buildTranscriptFallback() below reconstructs those from structured events.
    if(e.type==="agent_message"&&e.ticker){
      const tk=T(e.ticker);
      tk.transcript.push({kind:"msg", seq:e.seq, round:e.round, agent:e.agent, ...p});
      if((e.round||0)>0) tk.hasDebate = true;
    }
    if(e.type==="consensus_result"&&e.ticker){ T(e.ticker).transcript.push({kind:"consensus", seq:e.seq, round:e.round, ...p}); }
    if(e.round)m.maxRound=Math.max(m.maxRound,e.round);
    if(["opening_statement","rebuttal","moderator_synthesis","vote_cast","consensus_result"].includes(e.type)){
      const tk=T(e.ticker); const r=(tk.rounds[e.round] ||= {openings:{},rebuttals:{},synthesis:null,votes:{},consensus:null});
      if(e.round>0) tk.hasDebate = true;
      if(e.type==="opening_statement")r.openings[e.agent]=p;
      if(e.type==="rebuttal"){
        r.rebuttals[e.agent]=p;
        tk.transcript.push({kind:"rebuttal", synthetic:true, seq:e.seq+0.001, round:e.round, agent:e.agent,
          responding_to:p.responded_to, direction:p.revised_direction, q:p.revised_q, message:rebuttalMessage(p)});
      }
      if(e.type==="moderator_synthesis")r.synthesis=p;
      if(e.type==="vote_cast")r.votes[e.agent]=p;
      if(e.type==="consensus_result")r.consensus=p;
    }
    if(e.type==="consensus_failure"){ T(e.ticker).consensus={resolved:false,failure:p}; }
    if(e.type==="validator_result"){ m.validator[e.agent]=p; }
    if(e.type==="final_view"&&e.ticker){ T(e.ticker).final=p; }
    if(e.type==="final_view"&&!e.ticker&&(p.long_ticker||p.short_ticker)){
      if(p.long_ticker) T(p.long_ticker).final={...p,direction:"long",q:p.q,in_view:true};
      if(p.short_ticker) T(p.short_ticker).final={...p,direction:"short",q:-p.q,in_view:true};
    }   
    if(e.type==="optimized_portfolio"){ m.optimized=p; }
  }
  // If a run predates agent_message events, or a live stream drops them, the UI
  // can still show a debate transcript by reconstructing it from openings,
  // rebuttals, syntheses, tie-breaks, and vote results.
  for(const [t,tk] of Object.entries(m.tickers)){
    const explicitRebuttals = new Set((tk.transcript||[])
      .filter(x => !x.synthetic && x.kind === "rebuttal")
      .map(x => `${x.round}|${x.agent}`));
    tk.transcript = (tk.transcript||[]).filter(x =>
      !(x.synthetic && x.kind === "rebuttal" && explicitRebuttals.has(`${x.round}|${x.agent}`))
    );
    const hasMessages = (tk.transcript||[]).some(x => x.kind !== "consensus");
    if(!hasMessages && tk.hasDebate) tk.transcript = buildTranscriptFallback(events, t);
    tk.transcript.sort((a,b)=>(a.seq||0)-(b.seq||0));
  }
  return m;
}

function rebuttalMessage(p){
  const bits=[];
  bits.push(`Revised position: ${p.revised_direction||"—"} q=${fmt(p.revised_q,3)} with confidence ${fmt(p.revised_confidence,2)}.`);
  if((p.influenced_by||[]).length) bits.push(`Influenced by ${p.influenced_by.join(", ")}.`);
  if((p.concessions||[]).length) bits.push(`Concedes: ${p.concessions.join("; ")}.`);
  if((p.still_objects||[]).length) bits.push(`Still objects: ${p.still_objects.join("; ")}.`);
  if(p.strongest_counterargument_agent) bits.push(`Strongest counterargument: ${p.strongest_counterargument_agent}.`);
  return bits.join(" ");
}

function buildTranscriptFallback(events, ticker){
  const out=[];
  for(const e of events){
    if(e.ticker !== ticker) continue;
    const p=e.payload||{};
    if(e.type==="baseline_view"){
      out.push({kind:"msg", seq:e.seq, round:0, agent:"moderator", direction:p.direction, q:p.q,
        message:`Baseline view for ${ticker}: ${p.direction||"—"} (q=${fmt(p.q,3)}, confidence ${fmt(p.confidence,2)}). ${p.contested?"Agents disagree on direction — opening the floor.":"Agents broadly agree; no debate needed."}`});
    } else if(e.type==="opening_statement"){
      out.push({kind:"msg", seq:e.seq, round:e.round, agent:e.agent, direction:p.direction, q:p.q,
        message:p.position_rationale || p.rationale || "Opening statement recorded."});
    } else if(e.type==="rebuttal"){
      const replies=p.replies||[];
      if(replies.length){
        replies.forEach((rep,i)=>out.push({kind:"msg", seq:e.seq+i/100, round:e.round, agent:e.agent,
          responding_to:rep.responding_to, stance:rep.stance, quoted_claim:rep.quoted_claim,
          direction:p.revised_direction, q:p.revised_q, message:rep.message || p.revision_rationale || "Rebuttal recorded."}));
      } else {
        out.push({kind:"msg", seq:e.seq, round:e.round, agent:e.agent, direction:p.revised_direction, q:p.revised_q,
          message:p.revision_rationale || rebuttalMessage(p)});
      }
    } else if(e.type==="tiebreak"){
      out.push({kind:"msg", seq:e.seq, round:e.round, agent:p.tiebreaker||e.agent||"moderator", direction:p.favored, q:p.q_after,
        message:`${(p.tiebreaker||e.agent||"moderator").toString()} breaks the tie (${p.favored||"—"}): ${p.reason||""}`});
    } else if(e.type==="moderator_synthesis"){
      out.push({kind:"msg", seq:e.seq, round:e.round, agent:"moderator", direction:p.recommended_direction, q:p.recommended_q,
        message:(p.moderator_reasoning_summary||"Moderator synthesis recorded.") + ` Decision: ${p.recommended_direction||"—"} q=${fmt(p.recommended_q,3)}.`});
    } else if(e.type==="consensus_result"){
      out.push({kind:"consensus", seq:e.seq, round:e.round, ...p});
    }
  }
  return out;
}

// ---------------------------------------------------------------- render: debate
function dirClass(d){return "dir "+(d||"neutral");}
function tickerDirection(tk){
  const single=singleAgentFor(tk);
  return tk.final?.direction || tk.baseline?.direction || (single ? tk.signals[single]?.direction : null) || "—";
}
function tickerQ(tk){
  if(tk.final?.q!=null) return tk.final.q;
  if(tk.baseline?.q!=null) return tk.baseline.q;
  if((tk.pairwiseViews||[]).length) return tk.pairwiseViews[0].q;
  return null;
}
function renderDebate(){
  const m=STATE.model; if(!m){return;}
  // ticker list
  const tl=$("#tklist"); tl.innerHTML="";
  for(const t of Object.keys(m.tickers)){
    const tk=m.tickers[t];
    const res = tk.consensus?.resolved ?? (tk.final?tk.final.resolved:undefined);
    const single=singleAgentFor(tk);
    const isSingle = !!single && !tk.baseline && !tk.hasDebate;
    let badge = isSingle ? `<span class="badge resolved">${singleAgentLabel(single)}</span>` : (tk.contested? '<span class="badge contested">contested</span>':'<span class="badge resolved">agreed</span>');
    if(tk.consensus){ badge = tk.consensus.resolved? '<span class="badge resolved">resolved</span>':'<span class="badge failed">failed</span>'; }
    const q = tickerQ(tk);
    const dir = tickerDirection(tk);
    const debateNote = isSingle ? `${(tk.pairwiseViews||[]).length} pairwise view${(tk.pairwiseViews||[]).length===1?"":"s"}` : (tk.hasDebate ? `${Object.keys(tk.rounds).length} round${Object.keys(tk.rounds).length===1?"":"s"} · ${(tk.transcript||[]).length} transcript items` : "no debate; baseline only");
    const row=el("div","tk"+(t===STATE.ticker?" sel":""));
    row.innerHTML=`<div><b>${esc(t)}</b> <span class="${dirClass(dir)}">${esc(dir)}</span><br><span class="muted small">q=${fmt(q,3)} · ${esc(debateNote)}</span></div>${badge}`;
    row.onclick=()=>{STATE.ticker=t; renderDebate();};
    tl.appendChild(row);
  }
  renderCouncil();
  renderConvStrip();
  renderTranscript();
  renderAgentCards();
  renderInfluence();
  renderInputs();
}

// ---------------------------------------------------------------- council canvas
const ICON={quant:"📊",fundamental:"📒",sentiment:"📰",economist:"🏛️",moderator:"⚖️",single_gemma:"◆",single_gemini:"✨"};
const SEATS={quant:[22,28],fundamental:[78,28],sentiment:[22,82],economist:[78,82],moderator:[50,12],single_gemma:[50,50],single_gemini:[50,50]};
const ROUND_JOBS={0:"Baseline view",1:"R1 · Sharing what each only knows",2:"R2 · What happens next month?",3:"R3 · Settling the disagreement"};
function glyph(d){return d==="long"?'<span class="glyph" style="color:var(--long)">▲</span>':d==="short"?'<span class="glyph" style="color:var(--short)">▼</span>':'<span class="glyph" style="color:var(--neutral)">●</span>';}
function shortLine(msg){
  let s=(msg||"").trim();
  const dot=s.indexOf(". "); if(dot>10&&dot<90) s=s.slice(0,dot);
  const words=s.split(/\s+/); if(words.length>16) s=words.slice(0,16).join(" ")+"…";
  return s.length>120? s.slice(0,120)+"…":s;
}
function evidenceChip(t,cur){
  if(cur.kind==="reply"&&cur.quoted_claim) return "“"+cur.quoted_claim.slice(0,40)+"”";
  if(cur.kind==="opening"){
    const cv=(STATE.model.tickers[t]?.signals[cur.agent]?.cited_values||[])[0];
    if(cv) return `${cv.name}=${typeof cv.value==="number"?fmt(cv.value,3):cv.value}`;
  }
  if(cur.q!=null) return `q=${fmt(cur.q,3)}`;
  return "";
}
function currentMsgForTicker(t){
  const items=(STATE.model.tickers[t]?.transcript||[]).filter(x=>x.kind!=="consensus");
  const seq=currentMsgSeq(); let cur=null;
  for(const it of items){ if(it.seq<=seq) cur=it; else break; }
  return cur||items[0]||null;
}
function renderCouncil(){
  const host=$("#council"), t=STATE.ticker, m=STATE.model;
  $("#council-ticker").textContent=t||"—";
  if(!t||!m.tickers[t]){ host.innerHTML='<div class="muted" style="padding:24px">Pick a ticker.</div>'; $("#round-job").textContent=""; return; }
  const cur=currentMsgForTicker(t);
  $("#round-job").textContent = cur? (ROUND_JOBS[cur.round]||"") : "";
  let html='<div class="ctable"></div>';
  // arrow from peer -> speaker (who influenced whom)
  if(cur && cur.responding_to && SEATS[cur.responding_to] && SEATS[cur.agent]){
    const f=SEATS[cur.responding_to], s=SEATS[cur.agent];
    const col=cur.stance==="agree"?"var(--long)":cur.stance==="disagree"?"var(--short)":"var(--sentiment)";
    html+=`<svg style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2">
      <defs><marker id="ah" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="${col}"/></marker></defs>
      <line x1="${f[0]}%" y1="${f[1]}%" x2="${s[0]}%" y2="${s[1]}%" stroke="${col}" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#ah)" opacity="0.85"/></svg>`;
  }
  // seats
  for(const ag of Object.keys(SEATS)){
    const p=SEATS[ag], speaking=cur&&cur.agent===ag, glow=cur&&cur.kind==="tiebreak"&&cur.agent===ag;
    html+=`<div class="seat ${speaking?'speaking':''} ${glow?'glow':''}" style="left:${p[0]}%;top:${p[1]}%">
      <div class="avatar" style="border-color:${COLOR[ag]};color:${COLOR[ag]}">${ICON[ag]}</div>
      <div class="nm" style="color:${COLOR[ag]}">${ag}</div></div>`;
  }
  // current speech bubble
  if(cur){
    const p=SEATS[cur.agent]||SEATS.moderator, above=p[1]>50, top=above? p[1]-30 : p[1]+15;
    const bcol=cur.kind==="tiebreak"?"gold":(cur.stance==="agree"?"var(--long)":cur.stance==="disagree"?"var(--short)":cur.stance==="partial"?"var(--sentiment)":COLOR[cur.agent]);
    const chip=evidenceChip(t,cur);
    html+=`<div class="bubble2 ${cur.kind==='tiebreak'?'tb':''}" style="left:${p[0]}%;top:${top}%;border-left-color:${bcol}" title="${esc(cur.message)}">
      ${cur.kind==="tiebreak"?'<div class="replyto" style="color:gold">⚖ breaks the tie</div>':''}
      ${cur.responding_to?`<div class="replyto">↳ <b style="color:${COLOR[cur.responding_to]}">${esc(cur.responding_to)}</b>${cur.stance?` · ${esc(cur.stance)}`:''}</div>`:''}
      <div class="line">${glyph(cur.direction)} ${esc(shortLine(cur.message))}</div>
      ${chip?`<div class="echip">${esc(chip)}</div>`:''}</div>`;
  }
  host.innerHTML=html;
}

const STANCE_CLS = {agree:"agree", disagree:"disagree", partial:"partial"};
function renderTranscript(){
  const m=STATE.model, t=STATE.ticker, host=$("#transcript");
  const transcriptTickerEl = $("#transcript-ticker");
  if(transcriptTickerEl) transcriptTickerEl.textContent = t||"—";
  if(!t||!m.tickers[t]){ host.innerHTML='<span class="muted">No ticker.</span>'; return; }
  const items=m.tickers[t].transcript;
  if(!items.length){ host.innerHTML='<span class="muted">No transcript for this ticker yet. It may have been resolved at baseline, or this run predates transcript events.</span>'; return; }
  // the message at/just-before the current scrubber position is highlighted
  const curSeq = currentMsgSeq();
  host.innerHTML="";
  let lastRound=null;
  for(const it of items){
    const rnd = it.round||0;
    if(rnd!==lastRound){ lastRound=rnd; host.appendChild(el("div","rounddiv", rnd===0?"Baseline":`Round ${rnd}`)); }
    if(it.kind==="consensus"){
      const tally=it.tally||{};
      const chips=AGENTS.map(a=>{const v=(tally[a]||{}).vote; return v?`<span class="vote ${esc(v)}"><span style="color:${COLOR[a]}">${esc(a.slice(0,4))}</span> ${esc(v)}</span>`:"";}).join("");
      host.appendChild(el("div","votestrip",
        `${chips}<span class="vote ${it.resolved?'accept':'reject'}">${it.resolved?'✓ consensus':'✗ no consensus'} ${esc(it.accepts)}/${esc(it.threshold)}</span>`));
      continue;
    }
    const isMod = it.agent==="moderator";
    const color = COLOR[it.agent]||COLOR.moderator;
    const cls = isMod? "moderator" : (STANCE_CLS[it.stance]||"");
    const b=el("div","bubble "+cls+(isMod?"":" me")+(it.seq===curSeq?" cur":""));
    b.style.borderLeftColor=isMod?"var(--moderator)":(it.stance?undefined:color);
    let reply="";
    if(it.responding_to){
      reply=`<div class="replyto">↳ replying to <b style="color:${COLOR[it.responding_to]||'#cbd5e1'}">${esc(it.responding_to)}</b>${it.stance?` · <i>${esc(it.stance)}</i>`:""}${it.quoted_claim?` — “${esc((it.quoted_claim||'').slice(0,90))}”`:""}</div>`;
    }
    const pos = (it.q!=null)? `<span class="pos">${esc(it.direction||"")} q=${fmt(it.q,3)}</span>`:"";
    b.innerHTML=`<div class="who"><span style="color:${color}">${esc(it.agent)}${it.kind&&it.kind!=='reply'?` · ${esc(it.kind)}`:""}</span><span class="seq">#${fmt(it.seq,2)}${pos?" · "+pos:""}</span></div>${reply}<div>${nl2br(it.message||"")}</div>`;
    host.appendChild(b);
    if(it.seq===curSeq) setTimeout(()=>b.scrollIntoView({block:"nearest"}),0);
  }
}
// the seq of the latest agent_message at/before the scrubber
function currentMsgSeq(){
  let s=null;
  for(const e of STATE.events.slice(0,STATE.seek)){
    if(["agent_message","opening_statement","rebuttal","moderator_synthesis","tiebreak","single_model_emitted","single_model_pairwise_view"].includes(e.type)) s=e.seq;
  }
  return s;
}
function msgSeqs(){
  const fromTranscript=[];
  const model = STATE.model || buildModel(STATE.events.slice(0, STATE.seek || STATE.events.length));
  for(const tk of Object.values(model.tickers||{})){
    for(const it of (tk.transcript||[])){ if(it.kind!=="consensus" && it.seq!=null) fromTranscript.push(Math.floor(it.seq)); }
  }
  return [...new Set(fromTranscript)].sort((a,b)=>a-b);
}

function dirOf(q){return q>0.005?"long":(q<-0.005?"short":"neutral");}
function renderConvStrip(){
  const m=STATE.model, host=$("#convstrip"); host.innerHTML="";
  for(const t of Object.keys(m.tickers)){
    const tk=m.tickers[t];
    const pts=[];  // [label, q, direction]
    if(tk.baseline) pts.push(["base", tk.baseline.q, dirOf(tk.baseline.q)]);
    if(!tk.baseline && (tk.pairwiseViews||[]).length) pts.push(["C1", tk.pairwiseViews[0].q, tickerDirection(tk)]);
    for(let r=1;r<=m.maxRound;r++){ const s=tk.rounds[r]?.synthesis; if(s) pts.push(["R"+r, s.recommended_q, dirOf(s.recommended_q)]); }
    // end state
    let end="";
    if(tk.consensus){ end = tk.consensus.resolved? '<span class="badge resolved">resolved</span>':'<span class="badge failed">failed</span>'; }
    else if(tk.final){ end = tk.final.in_view? '<span class="badge resolved">in view</span>':'<span class="badge contested" style="opacity:.6">prior</span>'; }
    let ribbon="";
    pts.forEach((p,i)=>{
      if(i) ribbon+='<span class="arr">→</span>';
      const col=p[2]==="long"?"var(--long)":p[2]==="short"?"var(--short)":"var(--neutral)";
      const g=p[2]==="long"?"▲":p[2]==="short"?"▼":"●";
      ribbon+=`<div class="dot" title="${p[0]} q=${fmt(p[1],3)}" style="color:${col};border-color:${col}">${g}</div>`;
    });
    const box=el("div"); box.style.marginBottom="6px";
    box.innerHTML=`<div class="small" style="display:flex;justify-content:space-between"><b>${t}</b>${end}</div><div class="ribbon">${ribbon||'<span class="muted small">no debate</span>'}</div>`;
    host.appendChild(box);
  }
}

function agentCard(agent, opening, rebuttal){
  const c=el("div","acard");
  const dir = rebuttal?.revised_direction || opening?.direction || "—";
  const q = rebuttal?.revised_q ?? opening?.q;
  const conf = rebuttal?.revised_confidence ?? opening?.confidence;
  c.innerHTML=`<div class="nm" style="border-color:${COLOR[agent]};color:${COLOR[agent]}">${agent}${(opening?.used_mock||rebuttal?.used_mock)?' <span class="mockflag">mock</span>':''}</div>
    <div class="kv"><span>direction</span><span class="${dirClass(dir)}">${dir}</span></div>
    <div class="kv"><span>q</span><span>${fmt(q,3)}</span></div>
    <div class="kv"><span>confidence</span><span>${fmt(conf,2)}</span></div>`;
  // what the agent actually SAID this round (full utterance text)
  if(opening?.position_rationale) c.appendChild(el("div","small",'“'+opening.position_rationale.slice(0,260)+'”'));
  if(opening?.objection_to_current_view) c.appendChild(el("div","small obj-txt",'⤷ objects: '+opening.objection_to_current_view.slice(0,180)));
  if(rebuttal?.responded_to) c.appendChild(el("div","small muted",'↳ responded to: '+rebuttal.responded_to));
  const chips=el("div","chips");
  (opening?.evidence||[]).slice(0,4).forEach(ev=>chips.appendChild(el("span","chip", `${ev.name}=${typeof ev.value==="number"?fmt(ev.value,3):ev.value}`)));
  (rebuttal?.concessions||[]).forEach(x=>chips.appendChild(el("span","chip conc","conceded: "+x.slice(0,60))));
  (rebuttal?.still_objects||[]).forEach(x=>chips.appendChild(el("span","chip obj","objects: "+x.slice(0,60))));
  (rebuttal?.influenced_by||[]).forEach(x=>chips.appendChild(el("span","chip infl","← "+x)));
  if(rebuttal?.strongest_counterargument_agent) chips.appendChild(el("span","chip","vs "+rebuttal.strongest_counterargument_agent));
  c.appendChild(chips);
  return c;
}

const LABELCLS = {strongly_long:"long",long:"long",neutral:"neutral",underweight:"short",avoid:"short"};
function sigBadge(label){
  const c=LABELCLS[label]||"neutral";
  return `<span class="dir ${c}" style="font-size:11px">${(label||"—").replace("_"," ")}</span>`;
}
// Pre-debate Phase-1 card: categorical signal + agent-specific fields (NO q/P here).
function phase1Card(agent, sig){
  const c=el("div","acard");
  let extra="";
  const x=sig?.extra||{};
  if(agent==="quant") extra=`<div class="kv"><span>factors</span><span>${(x.primary_factors||[]).join(", ")}</span></div>
     <div class="kv"><span>alignment</span><span>${x.factor_alignment||"—"}</span></div>
     <div class="kv"><span>alpha</span><span>${fmt(x.alpha_contribution,3)}</span></div>`;
  else if(agent==="fundamental") extra=`<div class="kv"><span>valuation</span><span>${x.valuation_call||"—"}</span></div>
     <div class="kv"><span>quality</span><span>${x.quality_tier||"—"}</span></div>
     <div class="kv"><span>drivers</span><span>${(x.key_drivers||[]).join(", ")}</span></div>`;
  else if(agent==="sentiment") extra=`<div class="kv"><span>catalyst</span><span>${x.primary_catalyst||"—"}</span></div>
     <div class="kv"><span>event-driven</span><span>${x.event_driven?'<span style="color:var(--long)">yes</span>':'<span class="muted">opinion</span>'}</span></div>`;
  else if(agent==="economist") extra=`<div class="kv"><span>sector / stance</span><span>${x.sector||"—"} · <b>${x.stance||"neutral"}</b></span></div>
     <div class="kv"><span>rate</span><span>${x.rate_environment||"—"}</span></div>
     <div class="kv"><span>risk</span><span>${x.risk_regime||"—"}</span></div>
     <div class="kv"><span>cycle</span><span>${x.cycle||"—"}</span></div>`;
  else if(SINGLE_MODEL_AGENTS.includes(agent)) extra=`<div class="kv"><span>direction</span><span class="${dirClass(sig?.direction)}">${sig?.direction||"—"}</span></div>
     <div class="kv"><span>source</span><span>${sig?.source||singleAgentLabel(agent)}</span></div>
     <div class="kv"><span>confidence</span><span>${fmt(sig?.confidence,2)}</span></div>`;
  c.innerHTML=`<div class="nm" style="border-color:${COLOR[agent]};color:${COLOR[agent]}">${agent}${sig?.used_mock?' <span class="mockflag">mock</span>':''}</div>
    <div class="kv"><span>signal</span>${sigBadge(sig?.signal_label)}</div>${extra}`;
  if(sig?.rationale){
    const r=el("div","ratio");
    r.innerHTML=`<div class="lbl">rationale</div><div>“${sig.rationale}”</div>`;
    c.appendChild(r);
  }
  if((sig?.cited_values||[]).length){
    const cv=el("div"); cv.appendChild(el("div","lbl","cited from input"));
    const chips=el("div","chips");
    sig.cited_values.slice(0,6).forEach(c2=>chips.appendChild(el("span","chip",`${c2.name}=${typeof c2.value==="number"?fmt(c2.value,3):c2.value}`)));
    cv.appendChild(chips); c.appendChild(cv);
  }
  return c;
}

function renderTimeline(){
  const m=STATE.model, t=STATE.ticker, host=$("#tlbody");
  if(!t){host.innerHTML='<span class="muted">No ticker.</span>';return;}
  host.innerHTML="";
  const tk=m.tickers[t];
  // --- economist macro view (runs first, sector-level, feeds the others) ---
  const ev = tk.signals.economist?.agent_view;
  if(ev){
    const eb=el("div","mod");
    eb.innerHTML=`<b style="color:var(--economist)">◷ Economist (runs first)</b> — regime <b>${ev.risk_regime||"—"}</b> · rates <b>${ev.rate_environment||"—"}</b>
      <div class="small muted">${(ev.regime_rationale||"").slice(0,220)}</div>
      <div class="small">favored: ${(ev.favored_sectors||[]).join(", ")||"—"} · avoid: ${(ev.risk_sectors||[]).join(", ")||"—"}</div>`;
    host.appendChild(eb);
  }
  // --- pre-debate: each agent's independent Phase-1 view (categorical, no q/P) ---
  const pre=el("div","round");
  const singleAgent = singleAgentFor(tk);
  const singleOnly = !!singleAgent && !AGENTS.some(a=>tk.signals[a]);
  pre.appendChild(el("div","rh", singleOnly ? `<b>${singleAgentLabel(singleAgent)}</b> · one-shot single-model signal + pairwise BL views` : "<b>Initial views</b> · Phase 1 — agents emit a signal + rationale only (q/P appear later, at the moderator)"));
  const pg=el("div","agentgrid");
  const phaseAgents = singleOnly ? [singleAgent] : AGENTS;
  phaseAgents.forEach(a=>pg.appendChild(phase1Card(a, tk.signals[a])));
  pre.appendChild(pg);
  host.appendChild(pre);

  if(tk.baseline){
    const b=el("div","mod");
    b.innerHTML=`<b style="color:var(--moderator)">⚖ Moderator — baseline BL view</b> — <span class="${dirClass(tk.baseline.direction)}">${tk.baseline.direction}</span>
      q=${fmt(tk.baseline.q,3)} · confidence=${fmt(tk.baseline.confidence,2)} · Ω=${fmt(tk.baseline.omega,6)}
      · ${tk.contested?'<span class="badge contested">contested → debated</span>':'<span class="badge resolved">agreement</span>'}
      <div class="small muted">agent signals: ${Object.entries(tk.baseline.agent_signals||{}).map(([a,v])=>`<span style="color:${COLOR[a]}">${a} ${fmt(v,2)}</span>`).join(" · ")}</div>`;
    host.appendChild(b);
  }
  if((tk.pairwiseViews||[]).length){
    const pairAgent = singleAgent || m.singleModelMeta?.agent_name || SINGLE_MODEL_AGENT;
    const rows = tk.pairwiseViews.map(v=>`<tr><td>${esc(v.long_ticker)} &gt; ${esc(v.short_ticker)}</td><td>${fmt(v.q,4)}</td><td>${fmt(v.confidence,2)}</td><td class="small muted" style="text-align:left">${esc(v.rationale||"").slice(0,180)}</td></tr>`).join("");
    host.appendChild(el("div","mod",`<b style="color:${COLOR[pairAgent]}">${ICON[pairAgent]||"◆"} ${singleAgentLabel(pairAgent)} pairwise BL views</b><table style="margin-top:6px"><tr><th>view</th><th>Q</th><th>conf</th><th>rationale</th></tr>${rows}</table>`));
  }
  for(let r=1;r<=m.maxRound;r++){
    const rd=tk.rounds[r]; if(!rd)continue;
    const box=el("div","round");
    box.appendChild(el("div","rh",`<b>Round ${r}</b> ${rd.consensus?('· '+(rd.consensus.resolved?'<span class="badge resolved">consensus reached</span>':'<span class="badge contested">no consensus ('+rd.consensus.accepts+'/'+rd.consensus.threshold+')</span>')):''}`));
    const grid=el("div","agentgrid");
    AGENTS.forEach(a=>grid.appendChild(agentCard(a, rd.openings[a], rd.rebuttals[a])));
    box.appendChild(grid);
    if(rd.synthesis){
      const s=rd.synthesis;
      box.appendChild(el("div","mod",`<b>Moderator synthesis</b> → <span class="${dirClass(s.recommended_direction)}">${s.recommended_direction}</span>
        q=${fmt(s.recommended_q,3)} · conf=${fmt(s.recommended_confidence,2)}<br>
        <span class="small">${(s.moderator_reasoning_summary||"").slice(0,260)}</span><br>
        <span class="small muted">swayed by: ${(s.who_swayed_the_decision||[]).map(a=>`<span style="color:${COLOR[a]}">${a}</span>`).join(", ")||"—"}
        · changed: ${(s.agents_that_changed_position||[]).join(", ")||"—"}</span>`));
    }
    if(rd.votes && Object.keys(rd.votes).length){
      const v=el("div","votes");
      AGENTS.forEach(a=>{const vt=rd.votes[a]; if(!vt)return;
        v.appendChild(el("span","vote "+vt.vote, `<span style="color:${COLOR[a]}">${a}</span>: ${vt.vote} <span class="muted">(${vt.position_source})</span>`));});
      box.appendChild(v);
    }
    host.appendChild(box);
  }
  if(tk.consensus && tk.consensus.resolved===false && tk.consensus.failure){
    host.appendChild(el("div","mod",`<span class="mockflag">CONSENSUS FAILURE</span> — ${tk.consensus.failure.message||""}`));
  }
}

function renderAgentCards(){
  const m=STATE.model, t=STATE.ticker, host=$("#agentcards"); host.innerHTML="";
  if(!t){return;}
  const tk=m.tickers[t];
  // latest round with data
  let lastR=null; for(let r=m.maxRound;r>=1;r--){ if(tk.rounds[r]){lastR=r;break;} }
  const grid=el("div","agentgrid");
  const singleAgent = singleAgentFor(tk);
  const cardAgents = (singleAgent && !AGENTS.some(a=>tk.signals[a])) ? [singleAgent] : AGENTS;
  cardAgents.forEach(a=>{
    const op = lastR? tk.rounds[lastR].openings[a] : null;
    const rb = lastR? tk.rounds[lastR].rebuttals[a] : null;
    if(op||rb){ grid.appendChild(agentCard(a,op,rb)); }
    else { grid.appendChild(agentCard(a, tk.signals[a]?{direction:tk.signals[a].direction,q:tk.signals[a].signal*0.12,confidence:tk.signals[a].confidence,position_rationale:tk.signals[a].rationale,used_mock:tk.signals[a].used_mock}:null, null)); }
  });
  host.appendChild(grid);
}

function renderInfluence(){
  const m=STATE.model, t=STATE.ticker, host=$("#influence"); host.innerHTML="";
  if(!t){return;}
  const tk=m.tickers[t];
  const singleAgent = singleAgentFor(tk);
  if(singleAgent && !tk.hasDebate){host.innerHTML=`<span class="muted small">${singleAgentLabel(singleAgent)} has no multi-agent influence matrix; inspect its pairwise BL views above.</span>`;return;}
  let lastR=null; for(let r=m.maxRound;r>=1;r--){ if(tk.rounds[r]?.rebuttals){lastR=r;break;} }
  if(!lastR){host.innerHTML='<span class="muted small">No debate rounds for this ticker.</span>';return;}
  host.appendChild(el("div","small muted",`Round ${lastR}: arrow A → B means B was influenced by A.`));
  const tbl=el("table","matrix");
  let head="<tr><th>influenced↓ / by→</th>"+AGENTS.map(a=>`<th style="color:${COLOR[a]}">${a.slice(0,4)}</th>`).join("")+"</tr>";
  let body="";
  AGENTS.forEach(b=>{
    const rb=tk.rounds[lastR].rebuttals[b]||{};
    const infl=new Set(rb.influenced_by||[]);
    body+=`<tr><th style="color:${COLOR[b]};text-align:left">${b}</th>`+AGENTS.map(a=> `<td>${(a!==b&&infl.has(a))?'<span style="color:var(--accent)">●</span>':'<span class="muted">·</span>'}</td>`).join("")+"</tr>";
  });
  tbl.innerHTML=head+body; host.appendChild(tbl);
  // who swayed the moderator decision
  const s=tk.rounds[lastR].synthesis;
  if(s) host.appendChild(el("div","small",`<br><b>Swayed the decision:</b> ${(s.who_swayed_the_decision||[]).map(a=>`<span style="color:${COLOR[a]}">${a}</span>`).join(", ")||"—"}`));
}

// ---------------------------------------------------------------- render: agent inputs
function kvTable(obj, fmtMap){
  let rows="";
  for(const [k,v] of Object.entries(obj||{})){
    if(v==null) continue;
    let val = (typeof v==="number")? (fmtMap&&fmtMap[k]?fmtMap[k](v):fmt(v,4)) : (Array.isArray(v)? v.join(", ") : v);
    rows+=`<tr><td style="text-align:left">${k}</td><td>${val}</td></tr>`;
  }
  return `<table>${rows}</table>`;
}
function renderInputs(){
  const m=STATE.model, t=STATE.ticker, host=$("#inputsbody");
  $("#inputs-ticker").textContent = t||"—";
  const inputs = STATE.logmeta.agent_inputs;
  if(!inputs){ host.innerHTML='<div class="panel muted">Inputs are embedded in saved run logs — load a run from the picker (live runs: reload after it finishes).</div>'; return; }
  if(!t){ host.innerHTML='<div class="panel muted">Pick a ticker in the Debate tab.</div>'; return; }
  host.innerHTML="";
  const grid=el("div","grid2");

  // quant
  const q=inputs.quant?.stock_features?.[t]||{};
  grid.appendChild(panelFor("quant", `
    <div class="small muted">returns</div>${kvTable(q.returns,{})}
    <div class="small muted" style="margin-top:6px">risk</div>${kvTable(q.risk,{})}
    <div class="small muted" style="margin-top:6px">factor model (regression)</div>
    ${kvTable({n_obs:q.factor_model?.n_obs,r_squared:q.factor_model?.r_squared,alpha_annualized:q.factor_model?.alpha_annualized,net_style_factor_support:(q.factor_model?.net_style_factor_support ?? q.factor_model?.net_factor_support)})}
    <div class="small muted" style="margin-top:6px">top factor exposures (β, t, signal)</div>
    <table>${(q.factor_model?.factor_exposures||[]).slice(0,6).map(e=>`<tr><td style="text-align:left">${e.factor}</td><td>β=${fmt(e.beta,2)}</td><td>t=${fmt(e.t_stat,1)}</td><td>${e.signal_strength}</td></tr>`).join("")}</table>`));

  // fundamental
  const f=inputs.fundamental?.stocks?.[t]||{};
  grid.appendChild(panelFor("fundamental", `
    <div class="small muted">sector: ${f.sector||"—"} · GICS-sector percentile ranks (1.0 = best)</div>
    ${kvTable(f.sector_percentile_ranks,{})}
    <div class="small muted" style="margin-top:6px">consensus</div>${kvTable(f.consensus,{implied_upside:v=>pct(v)})}`));

  // sentiment (news headlines + yfinance analyst)
  const s=inputs.sentiment?.stocks?.[t]||{};
  const newsRows=(s.news||[]).map(n=>`<tr><td style="text-align:left">${(n.title||"").slice(0,70)}</td><td class="${(n.sentiment_score||0)>=0?'':'mockflag'}" style="${(n.sentiment_score||0)>=0?'color:var(--long)':''}">${fmt(n.sentiment_score,2)}</td><td class="muted">${n.source||""}</td></tr>`).join("");
  grid.appendChild(panelFor("sentiment", `
    ${kvTable({avg_factual_sentiment:s.avg_factual_sentiment,article_count:s.article_count,event_flags:s.event_flags,analyst_upgrades:s.analyst_upgrades,analyst_downgrades:s.analyst_downgrades})}
    <div class="small muted" style="margin-top:6px">news (AlphaVantage NEWS_SENTIMENT)</div>
    <table>${newsRows||'<tr><td class="muted">—</td></tr>'}</table>
    <div class="small muted" style="margin-top:6px">analyst (yfinance .info)</div>${kvTable(s.analyst,{implied_upside:v=>pct(v)})}
    <div class="small muted" style="margin-top:6px">market context</div>${kvTable(inputs.sentiment?.market_context,{})}`));

  // economist
  const e=inputs.economist||{};
  const sector=(e.universe_sector_mapping||{})[t];
  const dr=e.derived_regime||{};
  const regimeStr = (typeof dr==="object") ? `rate:${dr.rate_environment||"—"} · risk:${dr.risk_regime||"—"} · cycle:${dr.cycle||"—"}` : (dr||"—");
  grid.appendChild(panelFor("economist", `
    <div class="small muted">${t} sector: <b>${sector||"—"}</b> · stance: <b>${(e.sector_stances||{})[sector]||"neutral"}</b></div>
    <div class="small">regime → ${regimeStr}</div>
    <div class="small muted" style="margin-top:6px">macro snapshot</div>${kvTable(e.macro_snapshot,{})}`));

  host.appendChild(grid);
}
function panelFor(agent, inner){
  const p=el("div","panel");
  p.innerHTML=`<h3 style="color:${COLOR[agent]};border-left:3px solid ${COLOR[agent]};padding-left:6px">${agent} input</h3>`+inner;
  return p;
}

// ---------------------------------------------------------------- render: outcome
function renderOutcome(){
  const m=STATE.model; if(!m)return;
  const vt=$("#viewstable");
  if(m.optimized){
    const eq=m.optimized.equilibrium_returns||{}, mu=m.optimized.posterior_expected_returns||{};
    let rows="";
    for(const t of Object.keys(m.tickers)){
      const tk=m.tickers[t]; const f=tk.final||{}; const fail=tk.consensus?.resolved===false;
      const pi=eq[t]??f.pi, mb=mu[t]??f.mu_bl, tilt=(mb!=null&&pi!=null)?mb-pi:null;
      const inv = f.in_view ? '<span class="dir long" style="font-size:11px">in view</span>'
                            : '<span class="muted small">at prior</span>';
      rows+=`<tr><td>${t}${fail?' <span class="mockflag small">⚠failure</span>':''}</td>
        <td class="${dirClass(f.direction||tickerDirection(tk))}">${f.direction||tickerDirection(tk)||"—"}</td>
        <td>${inv}</td>
        <td>${fmt(pi,4)}</td><td class="${tilt>1e-6?'dir long':(tilt<-1e-6?'dir short':'')}">${fmt(mb,4)}</td>
        <td class="${tilt>1e-6?'dir long':(tilt<-1e-6?'dir short':'')}">${tilt==null?'—':(tilt>=0?'+':'')+fmt(tilt,4)}</td></tr>`;
    }
    const kinds=m.optimized.bl_inputs?.view_kinds||[], K=kinds.length, N=Object.keys(m.tickers).length;
    const nAbs=kinds.filter(k=>k==="absolute").length, nPair=kinds.filter(k=>k==="pairwise").length;
    const rr=m.optimized.risk_readout||{};
    vt.innerHTML=`<table><tr><th>ticker</th><th>consensus dir</th><th>BL view?</th><th>π (equil)</th><th>μ_BL (post)</th><th>tilt</th></tr>${rows}</table>
      <div class="small muted" style="margin-top:6px">${m.singleModelMeta?`${singleAgentLabel(m.singleModelMeta.agent_name || SINGLE_MODEL_AGENT)} pairwise views`:"Debate-gated views"}: K=${K} of N=${N} (${nAbs} absolute +1, ${nPair} pairwise +1/−1); neutral/unresolved stocks stay at the prior. cov κ=${fmt(m.optimized.covariance_condition_number,1)} · optimizer: ${m.optimized.status} ${m.optimized.used_fallback?'<span class="mockflag">(fallback)</span>':''} · vol=${pct(m.optimized.portfolio_vol)}</div>
      <div class="small muted">Risk readout (monthly): VaR₉₅ ${pct(rr.var)} · CVaR₉₅ ${pct(rr.cvar)} · σ ${pct(rr.sigma_monthly)}</div>
      <div class="small muted">${m.optimized.note||""}</div>`;
    renderWeights();
  } else { vt.innerHTML='<span class="muted">Run not finalized yet.</span>'; $("#weightschart").innerHTML=""; }
  renderBLMatrices();
  renderValidator();
}

function renderBLMatrices(){
  const m=STATE.model, host=$("#blmatrix");
  const bl=m.optimized && m.optimized.bl_inputs;
  if(!bl){ host.innerHTML='<span class="muted">Run not finalized.</span>'; return; }
  const order=bl.order, n=order.length;
  // P matrix grid (K views × N assets) + Q and Ω columns
  let head=`<tr><th>view ↓ / asset →</th>${order.map(t=>`<th>${t}</th>`).join("")}<th>Q (view)</th><th>Ω (var)</th></tr>`;
  const labels=bl.view_labels||order, kinds=bl.view_kinds||order.map(()=> "absolute");
  let rows="";
  for(let i=0;i<bl.P.length;i++){
    const kind=kinds[i];
    const cells=bl.P[i].map(x=>`<td>${Math.abs(x)>1e-9?('<b style="color:'+(x>0?'var(--long)':'var(--short)')+'">'+(x>0?'+':'')+x+'</b>'):'<span class="muted">0</span>'}</td>`).join("");
    const qd = bl.Q[i]>0.0005?"long":(bl.Q[i]<-0.0005?"short":"neutral");
    const tag = `<span class="pill" style="color:${kind==='pairwise'?'var(--economist)':'var(--accent)'}">${kind}</span>`;
    rows+=`<tr><th style="text-align:left">${labels[i]} ${tag}</th>${cells}<td class="${dirClass(qd)}">${fmt(bl.Q[i],4)}</td><td>${fmt(bl.omega_diag[i],6)}</td></tr>`;
  }
  // π vs μ_BL row block
  let pm=`<tr><th style="text-align:left">π equilibrium</th>${bl.pi.map(x=>`<td>${fmt(x,4)}</td>`).join("")}<td colspan="2" class="muted">prior (Π=δΣw)</td></tr>`;
  pm+=`<tr><th style="text-align:left">μ_BL posterior</th>${bl.mu_bl.map((x,i)=>`<td class="${x>bl.pi[i]+1e-6?'dir long':(x<bl.pi[i]-1e-6?'dir short':'')}">${fmt(x,4)}</td>`).join("")}<td colspan="2" class="muted">after views</td></tr>`;
  host.innerHTML=`
    <div class="small muted" style="margin-bottom:6px">E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q] &nbsp;·&nbsp; τ=${bl.tau} &nbsp; δ=${bl.risk_aversion} &nbsp;·&nbsp; absolute views ⇒ P is the asset-picking matrix (one view per ticker).</div>
    <table class="matrix">${head}${rows}</table>
    <div class="small muted" style="margin:8px 0 4px">prior vs posterior expected returns (annualized):</div>
    <table class="matrix">${pm}</table>`;
}

function renderWeights(){
  const m=STATE.model, o=m.optimized; const host=$("#weightschart");
  const w=o.weights||{}, eq=o.equilibrium_weights||{};
  const ts=Object.keys(m.tickers);
  const W=520,bh=22,gap=10,H=ts.length*(bh+gap)+10, mx=Math.max(0.0001,...ts.map(t=>Math.max(w[t]||0,eq[t]||0)));
  let svg=`<svg width="100%" viewBox="0 0 ${W} ${H}">`;
  ts.forEach((t,i)=>{const y=i*(bh+gap)+6; const ww=(w[t]||0)/mx*(W-90); const we=(eq[t]||0)/mx*(W-90);
    svg+=`<text x="0" y="${y+15}" fill="var(--mut)" font-size="11">${t}</text>`;
    svg+=`<rect x="46" y="${y+11}" width="${we}" height="8" fill="var(--moderator)" opacity="0.5"></rect>`;
    svg+=`<rect x="46" y="${y}" width="${ww}" height="10" fill="var(--accent)"></rect>`;
    svg+=`<text x="${46+Math.max(ww,we)+5}" y="${y+12}" fill="var(--txt)" font-size="10">${pct(w[t])}</text>`;});
  svg+=`</svg>`;
  const modelLabel = m.singleModelMeta ? `${singleAgentLabel(m.singleModelMeta.agent_name)} weight` : "multi-agent weight";
  host.innerHTML=`<div class="legend"><span><i style="background:var(--accent)"></i>${modelLabel}</span><span><i style="background:var(--moderator);opacity:.5"></i>equilibrium 1/N</span></div>`+svg;
}

function renderValidator(){
  const m=STATE.model, host=$("#valtable");
  if(!Object.keys(m.validator).length){host.innerHTML='<span class="muted">—</span>';return;}
  let rows="";
  Object.keys(m.validator).forEach(a=>{const v=m.validator[a]; if(!v)return;
    const hr=v.hallucination_rate; const col=hr>0?'var(--warn)':'var(--long)';
    rows+=`<tr><td style="color:${COLOR[a]||'var(--txt)'}">${a}${v.used_mock?' <span class="mockflag small">mock</span>':''}</td>
      <td style="color:${col}">${pct(hr)}</td><td>${v.n_hallucinated??"—"}/${v.n_cited_total??"—"}</td>
      <td>${v.verdict||"—"}</td><td class="small muted" style="text-align:left">${(v.shape_issues||[]).join("; ")||"none"}</td></tr>`;});
  host.innerHTML=`<table><tr><th>agent</th><th>hallucination rate</th><th>untraced/cited</th><th>verdict</th><th>shape issues</th></tr>${rows}</table>
   <div class="small muted" style="margin-top:6px">Each cited value is traced to that agent's input package within ${pct((m.validator.quant||m.validator.single_gemma||m.validator.single_gemini)?.tolerance||0.05)} tolerance.</div>`;
}

// ---------------------------------------------------------------- backtest
async function loadBacktest(name){
  const bt=await (await fetch(api("api/backtests/"+encodeURIComponent(name)))).json();
  STATE.backtest.name = name;
  const modeLabel = ` · ${runModeLabel(bt.run_mode)}`;
  $("#btmeta").textContent=`${bt.universe.join(", ")} · ${bt.start}→${bt.end} · ${bt.per_month.length} months · costs ${bt.txn_cost_bps}bps${modeLabel}`;
  renderEquity(bt); renderWeightEvo(bt); renderReturnsChart(bt); renderBtStrip(bt); renderBtTable(bt);
  await loadBacktestRunList(name);
}

async function loadBacktestRunList(name){
  const sel = $("#btrunsel");
  const meta = $("#btrunmeta");
  if(!sel || !meta) return;
  sel.innerHTML = "";
  meta.textContent = "loading saved run logs…";
  try{
    const data = await (await fetch(api("api/backtests/"+encodeURIComponent(name)+"/runs"))).json();
    const runs = data.runs || [];
    STATE.backtest.runs = runs;
    runs.forEach(r=>{
      const o=el("option");
      o.value=r.id;
      o.disabled=!r.available;
      o.textContent=`${r.label || r.name || r.id}${r.available?"":" (missing run log)"}${r.n_events?` · ${r.n_events} events`:""}`;
      sel.appendChild(o);
    });
    const available = runs.filter(r=>r.available).length;
    meta.textContent = runs.length ? `${available}/${runs.length} replayable` : "no saved run logs listed";
  }catch(e){
    meta.textContent = "could not read backtest run list";
  }
}

async function loadBacktestRun(){
  const name = STATE.backtest.name || $("#btsel")?.value;
  const runId = $("#btrunsel")?.value;
  if(!name || runId==null || runId===""){ return; }
  const res = await fetch(api("api/backtests/"+encodeURIComponent(name)+"/runs/"+encodeURIComponent(runId)));
  if(!res.ok){
    const msg = await res.text();
    alert("Could not load this run log. For older backtests, the standalone run JSON must still exist in runs/.\n\n" + msg);
    return;
  }
  const log = await res.json();
  const prov = (log.meta||{}).provider || "?";
  $("#runmeta").textContent = `${log.portfolio_id} · ${log.as_of_date} · ${prov} data · ${log.events.length} events`;
  STATE.logmeta = log.meta || {};
  startReplay(log.events);
  switchTab("debate");
}
const STRATS={
  debate:"var(--accent)",
  single_gemma:"var(--sentiment)",
  multi_gemini:"var(--economist)",
  equal_weight:"var(--moderator)",
  mvo_max_sharpe:"var(--sentiment)",
  risk_parity:"var(--economist)",
  single_gemini:"var(--quant)"
};

const STRAT_LABELS={
  debate:"Gemma multi",
  single_gemma:"Gemma single",
  multi_gemini:"Gemini multi",
  equal_weight:"Equal weight",
  mvo_max_sharpe:"MVO max Sharpe",
  risk_parity:"Risk parity",
  single_gemini:"Gemini single"
};

const MODEL_STRATS=["single_gemma","debate","single_gemini","multi_gemini"];
const BASELINE_STRATS=["equal_weight","mvo_max_sharpe","risk_parity"];

function selectedRunMode(){
  const sel=$("#runmode");
  if(sel && sel.value) return sel.value;
  return $("#geminic1")?.checked ? "gemini_c1" : "gemma_debate";
}

function runModeRunsGemini(mode){
  return ["gemini_c1","gemini_debate","compare","factorial"].includes(mode);
}

function runModeLabel(mode){
  if(mode === "gemma_single") return "Gemma single (SLM)";
  if(mode === "gemini_c1") return "Gemini single (LLM)";
  if(mode === "gemini_debate") return "Gemini multi (LLM)";
  if(mode === "compare") return "Legacy comparison: Gemma multi vs Gemini single";
  if(mode === "factorial") return "2x2 factorial: all four configurations";
  return "Gemma multi (SLM)";
}

function activeBacktestStrats(bt){
  return Object.keys(STRATS).filter(s =>
    bt?.performance?.[s] ||
    (bt?.per_month || []).some(m =>
      (m?.net_return && m.net_return[s] != null) ||
      (m?.weights && m.weights[s] != null)
    )
  );
}
function cumcurve(rets){const c=[1];rets.forEach(r=>c.push(c[c.length-1]*(1+r)));return c;}
function renderEquity(bt){
  const host=$("#equity");
  const W=560,H=240,pad=30;
  const months=bt.per_month.map(m=>m.as_of_date);
  const strats=activeBacktestStrats(bt);

  const curves={};
  strats.forEach(s=>{
    curves[s]=cumcurve(bt.per_month.map(m=>Number(m.net_return?.[s] || 0)));
  });

  const all=[].concat(...Object.values(curves));
  const mn=Math.min(...all);
  const mx=Math.max(...all);

  const x=i=>pad+i/(months.length)*(W-2*pad);
  const y=v=>H-pad-(v-mn)/(mx-mn||1)*(H-2*pad);

  let svg=`<svg width="100%" viewBox="0 0 ${W} ${H}">`;
  svg+=`<line x1="${pad}" y1="${y(1)}" x2="${W-pad}" y2="${y(1)}" stroke="var(--line)"/>`;

  strats.forEach(s=>{
    const c=curves[s];
    const d=c.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    svg+=`<path d="${d}" fill="none" stroke="${STRATS[s]}" stroke-width="2"/>`;
  });

  svg+=`</svg>`;

  host.innerHTML=
    `<div class="legend">${
      strats.map(s=>`<span><i style="background:${STRATS[s]}"></i>${STRAT_LABELS[s] || s}</span>`).join("")
    }</div>`+svg;
}
function backtestWeight(m, strat, ticker, bt){
  const exact = m?.weights?.[strat]?.[ticker];
  if(exact != null) return Number(exact) || 0;

  // Backward compatibility: some early Gemini-only files wrote the only model
  // portfolio into weights.debate even though the returns were labeled
  // single_gemini. Do not crash or render a blank panel for those runs.
  if(strat === SINGLE_MODEL_AGENT && bt?.run_mode === "gemini_c1"){
    const legacy = m?.weights?.debate?.[ticker];
    if(legacy != null) return Number(legacy) || 0;
  }

  return 0;
}

function backtestHasWeights(bt, strat){
  return (bt?.per_month || []).some(m => {
    if(m?.weights?.[strat]) return true;
    return strat === SINGLE_MODEL_AGENT && bt?.run_mode === "gemini_c1" && !!m?.weights?.debate;
  });
}

function renderOneWeightEvo(bt, strat){
  const W=560,H=240,pad=24; const months=bt.per_month || [];
  const U=bt.universe || []; const palette=["#3b82f6","#22c55e","#f59e0b","#a855f7","#ef4444","#14b8a6","#eab308","#ec4899"];
  if(!months.length) return '<span class="muted">No weights available.</span>';
  const hasWeights = backtestHasWeights(bt, strat);
  if(!hasWeights) return `<span class="muted">No ${esc(STRAT_LABELS[strat] || strat)} weights in this backtest.</span>`;

  const x=i=>pad+i/(months.length-1||1)*(W-2*pad);
  let svg=`<svg width="100%" viewBox="0 0 ${W} ${H}">`;
  const cum=months.map(()=>0);

  U.forEach((t,ti)=>{
    const top=months.map(m=>backtestWeight(m, strat, t, bt));
    let poly="";
    months.forEach((m,i)=>{const base=cum[i]; poly+=`${i?'L':'M'}${x(i).toFixed(1)},${(H-pad-base*(H-2*pad)).toFixed(1)} `;});
    for(let i=months.length-1;i>=0;i--){const v=cum[i]+top[i]; poly+=`L${x(i).toFixed(1)},${(H-pad-v*(H-2*pad)).toFixed(1)} `;}
    svg+=`<path d="${poly}Z" fill="${palette[ti%palette.length]}" opacity="0.8"/>`;
    months.forEach((m,i)=>cum[i]+=top[i]);
  });

  svg+=`</svg>`;
  return `<div class="small muted" style="margin-bottom:6px"><b>${esc(STRAT_LABELS[strat] || strat)}</b> weight evolution</div>`+
    `<div class="legend">${U.map((t,i)=>`<span><i style="background:${palette[i%palette.length]}"></i>${esc(t)}</span>`).join("")}</div>`+svg;
}

function renderWeightEvo(bt){
  const host=$("#weightevo");
  const title=$("#weightevo-title");
  const active=activeBacktestStrats(bt);
  const modelStrats=MODEL_STRATS.filter(s=>active.includes(s));
  const strats = modelStrats.length ? modelStrats : active.slice(0,1);

  if(title){
    title.textContent = strats.length > 1 ? "Weight evolution — model × architecture" : `Weight evolution — ${STRAT_LABELS[strats[0]] || "model portfolio"}`;
  }

  if(!strats.length){
    host.innerHTML='<span class="muted">No model weights available.</span>';
    return;
  }

  host.innerHTML = strats.map(s => `<div style="margin-bottom:14px">${renderOneWeightEvo(bt,s)}</div>`).join("");
}

function renderReturnsChart(bt){
  const host = $("#returnschart");
  const months = bt.per_month || [];

  if(!months.length){
    host.innerHTML = '<span class="muted">No monthly returns available.</span>';
    return;
  }

  const active = activeBacktestStrats(bt);
  const shown = MODEL_STRATS.filter(s => active.includes(s));
  if(active.includes("equal_weight")) shown.push("equal_weight");
  if(!shown.length) shown.push(...active.slice(0,3));

  const notional = 100000;
  const series = shown.map((s, si) => {
    const returns = months.map(m => Number(m.net_return?.[s] || 0));
    let value = notional;
    const values = [];
    const pnls = [];
    returns.forEach(r => {
      value *= (1 + r);
      values.push(value);
      pnls.push(value - notional);
    });
    return {s, si, returns, values, pnls};
  });

  const allReturns = series.flatMap(x => x.returns);
  const maxAbs = Math.max(0.001, ...allReturns.map(x => Math.abs(x)));

  const W = 1160;
  const rowH = 26 + shown.length * 18;
  const H = 58 + months.length * rowH;

  const dateX = 0;
  const padL = 120;
  const plotW = 390;
  const zeroX = padL + plotW / 2;

  const retX = padL + plotW + 32;
  const valueX = retX + 95;
  const pnlX = valueX + 130;
  const stratX = pnlX + 120;

  const x = v => padL + ((v + maxAbs) / (2 * maxAbs)) * plotW;

  let svg = `<svg width="100%" viewBox="0 0 ${W} ${H}">`;
  svg += `<line x1="${zeroX}" y1="36" x2="${zeroX}" y2="${H-12}" stroke="var(--line)"/>`;
  svg += `<text x="${dateX}" y="18" fill="var(--mut)" font-size="11">month</text>`;
  svg += `<text x="${padL}" y="18" fill="var(--mut)" font-size="11">monthly net return</text>`;
  svg += `<text x="${retX}" y="18" fill="var(--mut)" font-size="11">ret</text>`;
  svg += `<text x="${valueX}" y="18" fill="var(--mut)" font-size="11">$100k value</text>`;
  svg += `<text x="${pnlX}" y="18" fill="var(--mut)" font-size="11">P&L</text>`;
  svg += `<text x="${stratX}" y="18" fill="var(--mut)" font-size="11">strategy</text>`;

  months.forEach((m, i) => {
    const y0 = 44 + i * rowH;
    const label = m.as_of_date || "";
    svg += `<text x="${dateX}" y="${y0+18}" fill="var(--mut)" font-size="11">${esc(label)}</text>`;

    series.forEach((ser, j) => {
      const y = y0 + j * 18;
      const r = ser.returns[i];
      const bx = Math.min(zeroX, x(r));
      const bw = Math.max(2, Math.abs(x(r) - zeroX));
      const color = STRATS[ser.s] || (r >= 0 ? "var(--long)" : "var(--short)");
      svg += `<rect x="${bx}" y="${y}" width="${bw}" height="13" rx="4" fill="${color}" opacity="0.88"></rect>`;
      svg += `<text x="${retX}" y="${y+11}" fill="${r >= 0 ? "var(--long)" : "var(--short)"}" font-size="11">${pct(r)}</text>`;
      svg += `<text x="${valueX}" y="${y+11}" fill="var(--txt)" font-size="11">${money(ser.values[i])}</text>`;
      svg += `<text x="${pnlX}" y="${y+11}" fill="${ser.pnls[i] >= 0 ? "var(--long)" : "var(--short)"}" font-size="11">${money(ser.pnls[i])}</text>`;
      svg += `<text x="${stratX}" y="${y+11}" fill="${color}" font-size="11">${esc(STRAT_LABELS[ser.s] || ser.s)}</text>`;
    });
  });

  svg += `</svg>`;

  host.innerHTML =
    `<div class="small muted" style="margin-bottom:8px">
      Starts with $100k in month 1. Values compound each strategy's monthly net return through time.
    </div>
    <div class="legend" style="margin-bottom:8px">
      ${shown.map(s=>`<span><i style="background:${STRATS[s]||'var(--txt)'}"></i>${esc(STRAT_LABELS[s] || s)}</span>`).join("")}
    </div>
    ${svg}`;
}

function renderBtStrip(bt){
  const host=$("#btstrip");
  const active=activeBacktestStrats(bt);
  const modelStrats=MODEL_STRATS.filter(s=>active.includes(s));
  const strats=modelStrats.length ? modelStrats : active.slice(0,1);

  let head = `<tr><th>month</th>${strats.map(s=>`<th>${esc(STRAT_LABELS[s]||s)} turnover</th><th>${esc(STRAT_LABELS[s]||s)} ret</th>`).join("")}<th>rounds</th><th>consensus failures</th><th>source</th></tr>`;
  let rows="";

  bt.per_month.forEach(m=>{
    const cf=(m.consensus_failures||[]).length;
    const stratCells=strats.map(s=>`<td>${pct(m.turnover?.[s]||0)}</td><td>${pct(m.net_return?.[s]||0)}</td>`).join("");
    const source = m.any_mock ? '<span class="mockflag small">Gemma mock</span>' :
      m.gemini_any_mock ? '<span class="mockflag small">Gemini mock</span>' :
      '<span class="muted small">model</span>';
    rows+=`<tr><td>${esc(m.as_of_date)}</td>${stratCells}<td>${m.n_rounds ?? 0}</td>
      <td>${cf?('<span class="mockflag">'+m.consensus_failures.map(esc).join(", ")+'</span>'):'<span class="muted">none</span>'}</td>
      <td>${source}</td></tr>`;
  });

  host.innerHTML=`<table>${head}${rows}</table>`;
}

function renderBtTable(bt){
  const host=$("#bttable");
  const P=bt.performance || {};
  const strats=activeBacktestStrats(bt);
  let rows="";

  strats.forEach(s=>{
    const n=P[s]?.net;
    if(!n) return;

    rows+=`<tr>
      <td style="color:${STRATS[s]||'var(--txt)'}">${STRAT_LABELS[s] || s}</td>
      <td>${pct(n.cumulative_return)}</td>
      <td>${pct(n.annualized_return)}</td>
      <td>${fmt(n.sharpe,2)}</td>
      <td>${fmt(n.sortino,2)}</td>
      <td>${pct(n.max_drawdown)}</td>
      <td>${fmt(n.calmar,2)}</td>
      <td>${fmt(n.avg_turnover,2)}</td>
    </tr>`;
  });

  host.innerHTML=`<table>
    <tr>
      <th>strategy</th>
      <th>cum ret</th>
      <th>ann ret</th>
      <th>Sharpe</th>
      <th>Sortino</th>
      <th>maxDD</th>
      <th>Calmar</th>
      <th>turnover</th>
    </tr>
    ${rows}
  </table>`;
}

// ---------------------------------------------------------------- live polling
function stopLivePolling(){
  if(STATE.live.timer) clearInterval(STATE.live.timer);
  STATE.live.timer = null;
}

function setLiveChrome(status, stage, count){
  const bar = $("#livebar");
  const dot = $("#live-dot");
  const fill = $("#live-fill");

  if(!bar || !dot || !fill) return;

  bar.style.display = "flex";
  dot.className = "live-dot " + status;
  fill.className = "live-fill " + status;

  $("#live-title").textContent =
    status === "running" ? "Agents are thinking" :
    status === "done" ? "Run complete" :
    status === "error" ? "Run failed" :
    "Live run";

  $("#live-stage").textContent = stage || "Waiting for events…";
  $("#live-count").textContent = `${count || 0} events`;
}

function describeEvent(e){
  if(!e) return "Starting the rebalance…";

  const bits = [];

  if(e.phase) bits.push(e.phase);
  if(e.type) bits.push(e.type.replaceAll("_", " "));
  if(e.ticker) bits.push(e.ticker);
  if(e.round) bits.push("round " + e.round);
  if(e.agent) bits.push(e.agent);

  return bits.join(" · ");
}

function handleLivePollError(err){
  stopLivePolling();

  const btn = $("#livebtn");
  btn.disabled = false;
  btn.textContent = "▶ Start live run";

  const msg = err?.message || String(err);

  setLiveChrome("error", msg, STATE.events.length);

  $("#runmeta").textContent = "live error · polling failed";
  $("#narr").textContent = "ERROR: " + msg;
}

function liveRun(){
  const proto = location.protocol==="https:"?"wss":"ws";
  const ws = new WebSocket(`${proto}://${location.host}${BASE}ws/run`);
  let events = [];

  const start = $("#start").value.trim();
  const end = $("#end").value.trim();

  const universe = parseUniverseInput();

  if(!universe.length){
    alert("Please enter at least one ticker in the universe field.");
    return;
  }

  $("#livebtn").disabled = true;
  $("#livebtn").textContent = "running…";

  ws.onopen = () => {
    ws.send(JSON.stringify({
      provider: $("#provsel").value,
      start,
      end,
      portfolio_id: "backtest",
      universe,
      strict: $("#strict").checked,
      run_gemini: runModeRunsGemini(selectedRunMode()),
      run_mode: selectedRunMode()
    }));

    STATE.events = events;
    STATE.seek = 0;
    STATE.ticker = null;
    $("#scrubbar").style.display = "flex";
    $("#runmeta").textContent =
    `● LIVE BACKTEST · ${start} → ${end} · ${universeLabel(universe)} · ${runModeLabel(selectedRunMode())}`;
  };

  ws.onmessage = async ev => {
    const d = JSON.parse(ev.data);

    if(d._meta === "progress"){
      $("#runmeta").textContent = `● LIVE BACKTEST · ${d.message}`;
      return;
    }

    if(d._meta === "done"){
      $("#runmeta").textContent = `live backtest done · ${d.months || 0} months · ${d.summary_path || "saved"}`;
      $("#livebtn").disabled = false;
      $("#livebtn").textContent = "▶ Live backtest";

      await loadList();
      if(d.summary_path){
        $("#btsel").value = d.summary_path;
        await loadBacktest(d.summary_path);
      }
      return;
    }

    if(d._meta === "error"){
      $("#narr").textContent = "ERROR: " + d.error;
      $("#runmeta").textContent = "live backtest error";
      $("#livebtn").disabled = false;
      $("#livebtn").textContent = "▶ Live backtest";
      alert("Live backtest error:\n" + d.error);
      return;
    }

    // Each month-end rebalance starts its own event log with seq=1.
    // Reset the debate viewer when a new month starts so the UI shows the
    // currently-running portfolio decision instead of mixing months together.
    if(d.type === "run_started" && events.length){
      events = [];
      STATE.events = events;
      STATE.seek = 0;
      STATE.ticker = null;
    }

    events.push(d);
    STATE.seek = events.length;
    $("#seek").max = events.length;
    $("#seek").value = events.length;

    const asOf = d.payload?.as_of_date || d.ticker || "";

    if(d.type === "run_started"){
      const evUniverse = d.payload?.universe || universe;
      $("#runmeta").textContent =
        `● LIVE REBALANCE · ${asOf} · ${universeLabel(evUniverse)}`;
    }

    rebuild();
  };

  ws.onerror = () => {
    $("#narr").textContent = "websocket error";
    $("#runmeta").textContent = "live backtest websocket error";
    $("#livebtn").disabled = false;
    $("#livebtn").textContent = "▶ Live backtest";
  };
}

async function pollLiveRun(){
  if(!STATE.live.runId) return;

  const url = api(`api/live-runs/${STATE.live.runId}?after=${STATE.live.lastSeq}`);
  const res = await fetch(url);

  if(!res.ok) throw new Error(await res.text());

  const data = await res.json();

  if((data.events || []).length){
    STATE.events.push(...data.events);
    STATE.live.lastSeq = data.next_seq || STATE.live.lastSeq;

    STATE.seek = STATE.events.length;

    $("#seek").max = STATE.events.length;
    $("#seek").value = STATE.seek;

    rebuild();
  }

  STATE.live.status = data.status;

  const stage = data.error || describeEvent(data.last_event);

  setLiveChrome(data.status, stage, data.event_count);

  $("#runmeta").textContent =
    data.status === "running"
      ? `● LIVE · ${data.event_count} events`
      : data.status === "done"
        ? `live done · ${data.run_log_path || "saved"}`
        : `live error · ${data.error || "failed"}`;

  if(data.status === "done" || data.status === "error"){
    stopLivePolling();

    const btn = $("#livebtn");
    btn.disabled = false;
    btn.textContent = "▶ Start live run";

    if(data.status === "done") loadList();
  }
}

// ---------------------------------------------------------------- wiring
$("#loadbtn").onclick=()=>loadRun($("#runsel").value);
$("#refreshbtn").onclick=async()=>{ await loadList(); $("#runmeta").textContent=`${$("#runsel").options.length} runs found`; };
$("#btload").onclick=()=>loadBacktest($("#btsel").value);
$("#btrunload").onclick=loadBacktestRun;
$("#livebtn").onclick=liveRun;
// Prev/Next/Play step through the conversation one agent_message at a time (spec B5).
function stepMsg(dir){
  const seqs=msgSeqs(); if(!seqs.length){ return step(dir); }
  if(dir>0){ const nx=seqs.find(s=>s>STATE.seek); STATE.seek = nx==null?STATE.events.length:nx; }
  else { const pv=[...seqs].reverse().find(s=>s<STATE.seek); STATE.seek = pv==null?0:pv; }
  // follow the conversation: switch the council to whoever is speaking now
  const speaking=STATE.events.find(e=>e.seq===STATE.seek);
  if(speaking && speaking.type==="agent_message" && speaking.ticker) STATE.ticker=speaking.ticker;
  $("#seek").value=STATE.seek; rebuild();
}
$("#prev").onclick=()=>stepMsg(-1);
$("#next").onclick=()=>stepMsg(1);
$("#seek").oninput=e=>{STATE.seek=+e.target.value; rebuild();};
$("#play").onclick=function(){ STATE.playing=!STATE.playing; this.textContent=STATE.playing?"⏸ pause":"▶ play";
  const tick=()=>{ if(!STATE.playing)return; const seqs=msgSeqs(); const last=seqs.length?seqs[seqs.length-1]:STATE.events.length;
    if(STATE.seek>=last){STATE.playing=false;this.textContent="▶ play";return;} stepMsg(1); setTimeout(tick,650); }; tick(); };
function switchTab(name){
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active", x.dataset.tab===name));
  ["debate","inputs","outcome","backtest"].forEach(n=>$("#tab-"+n).style.display=(n===name)?"block":"none");
  if(name==="inputs") renderInputs();
}

document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>switchTab(t.dataset.tab));
loadConfigDefaults().then(loadList);