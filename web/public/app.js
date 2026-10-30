// The page: ask a question, stream the answer, show citations and per-sentence checks.
// Everything that comes from the model is inserted with textContent, never as HTML.

const $ = (id) => document.getElementById(id);
const form = $("form"), box = $("q"), askBtn = $("ask"), countEl = $("count");
const noticeEl = $("notice"), resultEl = $("result"), statusEl = $("status"), answerEl = $("answer"), sourcesEl = $("sources"), legendEl = $("legend");

let samples = [];
let turnstileId = null;
let turnstileToken = null;
let turnstileOn = false;
let busy = false;

// ---- state for one answer ---------------------------------------------------------------------

let state = fresh();
function fresh() { return { text: "", citations: [], sentences: null, timings: null, model: "", noAnswer: false }; }

const MARK = /\[(\d+(?:\s*[,;-]\s*\d+)*)\]/g;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** Append `text` to `parent`, turning [n] markers into buttons once the passage numbers are known. */
function appendWithCitations(parent, text) {
  const known = new Set(state.citations.map((c) => c.n));
  let last = 0;
  for (const m of text.matchAll(MARK)) {
    parent.append(text.slice(last, m.index));
    last = m.index + m[0].length;
    const nums = m[1].split(/\s*[,;]\s*/).flatMap((p) => {
      const r = p.split("-").map(Number);
      return r.length === 2 && r[1] >= r[0] && r[1] - r[0] < 20 ? Array.from({ length: r[1] - r[0] + 1 }, (_, i) => r[0] + i) : [Number(p)];
    });
    if (!nums.every((n) => known.has(n))) { parent.append(m[0]); continue; }
    parent.append("[");
    nums.forEach((n, i) => {
      if (i) parent.append(", ");
      const b = el("button", "cite", String(n));
      b.type = "button";
      b.title = "Show passage " + n;
      b.addEventListener("click", () => openSource(n));
      parent.append(b);
    });
    parent.append("]");
  }
  parent.append(text.slice(last));
}

function renderAnswer() {
  answerEl.replaceChildren();
  const text = state.text;
  if (!state.sentences || !state.sentences.length) { appendWithCitations(answerEl, text); return; }
  let pos = 0;
  for (const s of state.sentences) {
    const at = text.indexOf(s.text, pos);
    if (at < 0) continue;
    if (at > pos) answerEl.append(text.slice(pos, at)); // the whitespace between sentences, newlines included
    const span = el("span", "sentence");
    appendWithCitations(span, s.text);
    span.append(badge(s));
    answerEl.append(span);
    pos = at + s.text.length;
  }
  if (pos < text.length) appendWithCitations(answerEl, text.slice(pos));
}

function badge(s) {
  if (s.supported === null || s.supported === undefined) return el("span", "badge na", s.cited && s.cited.length ? "not checked" : "no citation");
  const p = s.p_entail.toFixed(2);
  const b = el("span", "badge " + (s.supported ? "ok" : "warn"), (s.supported ? "supported " : "not supported ") + p);
  b.title = "openjev-flash: P(entailment) = " + p + " for this sentence against passage" + (s.cited.length > 1 ? "s " : " ") + s.cited.join(", ");
  return b;
}

function renderSources() {
  sourcesEl.replaceChildren();
  for (const c of state.citations) {
    const d = el("details");
    d.id = "src-" + c.n;
    const sum = el("summary");
    sum.append(el("strong", "", "[" + c.n + "] "), document.createTextNode(c.document || c.file));
    sum.append(el("span", "meta", c.heading ? "  ·  " + c.heading : ""));
    d.append(sum);
    d.append(el("div", "passage", c.text || ""));
    const id = (c.file || "").replace(/\.md$/, "");
    if (/^\d{4}\.\d{4,5}$/.test(id)) {
      const a = el("a", "meta", "arXiv:" + id);
      a.href = "https://arxiv.org/abs/" + id; a.rel = "noopener"; a.target = "_blank";
      d.append(a);
    }
    sourcesEl.append(d);
  }
}

function openSource(n) {
  const d = $("src-" + n);
  if (!d) return;
  d.open = true;
  d.scrollIntoView({ behavior: "smooth", block: "nearest" });
  d.classList.add("flash");
  setTimeout(() => d.classList.remove("flash"), 1400);
}

// ---- event handling (shared by the live stream and the saved answers) ------------------------------

function onEvent(name, data) {
  switch (name) {
    case "retrieval":
      statusEl.textContent = data.relevant ? `Found ${data.count} passages. Asking the model...` : "Nothing relevant found in the papers.";
      break;
    case "token":
      if (!state.text) statusEl.textContent = "Writing the answer...";
      state.text += data.text;
      renderAnswer();
      break;
    case "citations":
      state.citations = data.citations || [];
      renderAnswer(); renderSources();
      statusEl.textContent = "Checking each cited sentence...";
      break;
    case "faithfulness":
      state.sentences = data.sentences || [];
      renderAnswer();
      legendEl.hidden = !state.sentences.some((s) => s.supported !== null && s.supported !== undefined);
      break;
    case "done": {
      state.timings = data.timings || null; state.model = data.model || ""; state.noAnswer = !!data.no_answer;
      const t = data.timings;
      const parts = [];
      if (data.model) parts.push("answered by " + data.model);
      if (t) parts.push(`${t.total_s.toFixed(1)} s (search ${t.retrieval_s.toFixed(1)}, model ${t.llm_s.toFixed(1)}, check ${t.faithfulness_s.toFixed(1)})`);
      statusEl.textContent = (data.no_answer ? "No answer: nothing relevant was found, so the model was not called. " : "") + parts.join(", ");
      break;
    }
    case "error":
      throw new LiveError(data.message || "The demo could not answer.", true);
  }
}

class LiveError extends Error {
  constructor(message, samples = true) { super(message); this.samples = samples; }
}

function showNotice(message, extraNode) {
  noticeEl.replaceChildren(document.createTextNode(message));
  if (extraNode) noticeEl.append(" ", extraNode);
  noticeEl.hidden = false;
}

function reset(question) {
  state = fresh();
  noticeEl.hidden = true;
  resultEl.hidden = false;
  answerEl.replaceChildren(); sourcesEl.replaceChildren();
  legendEl.hidden = true;
  statusEl.textContent = "Searching the papers...";
}

// ---- live ----------------------------------------------------------------------------------------------

async function readSse(res, handle) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    buf += dec.decode(value || new Uint8Array(), { stream: !done });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      let name = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) handle(name, JSON.parse(data));
    }
    if (done) return;
  }
}

async function askLive(question) {
  const body = { question };
  if (turnstileOn) body.turnstile = turnstileToken;
  let res;
  try {
    res = await fetch("/api/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new LiveError("Could not reach the demo server.", true);
  }
  if (!res.ok) {
    let info = {};
    try { info = await res.json(); } catch { /* not JSON */ }
    throw new LiveError(info.message || `The demo answered with HTTP ${res.status}.`, info.samples !== false && res.status !== 400 && res.status !== 413 && res.status !== 403);
  }
  await readSse(res, onEvent);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = box.value.trim();
  if (!question || busy) return;
  if (turnstileOn && !turnstileToken) { showNotice("Waiting for the bot check to finish. Try again in a moment."); return; }
  busy = true; askBtn.disabled = true;
  reset(question);
  try {
    await askLive(question);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const hit = samples.find((s) => s.question.toLowerCase() === question.toLowerCase());
    if (err instanceof LiveError && err.samples !== false) {
      if (hit) { showNotice(msg + " Showing the saved answer to this question instead."); replay(hit); }
      else { resultEl.hidden = true; showNotice(msg + " You can still read the saved answers below, recorded with the same pipeline."); }
    } else {
      resultEl.hidden = true; showNotice(msg);
    }
  } finally {
    busy = false; askBtn.disabled = false;
    if (turnstileOn && window.turnstile) { turnstileToken = null; window.turnstile.reset(turnstileId); }
  }
});

// ---- saved answers ---------------------------------------------------------------------------------------

function replay(sample) {
  reset(sample.question);
  box.value = sample.question; updateCount();
  const note = `Saved answer, recorded ${sample.recorded || ""}. `.replace(" , ", " ");
  for (const ev of sample.events) onEvent(ev.event, ev.data);
  statusEl.textContent = note + statusEl.textContent;
  if (noticeEl.hidden) showNotice("This is a saved answer, not a live one.");
}

function setupSamples() {
  const ex = $("examples"), saved = $("saved");
  if (!samples.length) { $("saved-wrap").hidden = true; return; }
  for (const s of samples) {
    const chip = el("button", "chip", s.question);
    chip.type = "button";
    chip.addEventListener("click", () => { box.value = s.question; updateCount(); box.focus(); });
    ex.append(chip);
    const row = el("button", "chip", s.question);
    row.type = "button";
    row.addEventListener("click", () => { replay(s); resultEl.scrollIntoView({ behavior: "smooth", block: "start" }); });
    saved.append(row);
  }
}

// ---- setup -----------------------------------------------------------------------------------------------

function updateCount() { countEl.textContent = `${box.value.length} / 500`; }
box.addEventListener("input", updateCount);
box.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit(); });

async function loadTurnstile(siteKey) {
  await new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    s.async = true; s.onload = resolve; s.onerror = reject;
    document.head.append(s);
  });
  turnstileOn = true;
  turnstileId = window.turnstile.render("#turnstile", {
    sitekey: siteKey, theme: "dark",
    callback: (t) => { turnstileToken = t; },
    "expired-callback": () => { turnstileToken = null; },
    "error-callback": () => { turnstileToken = null; },
  });
}

async function init() {
  updateCount();
  try {
    const r = await fetch("/samples.json");
    if (r.ok) samples = (await r.json()).samples || [];
  } catch { /* the page works without saved answers */ }
  setupSamples();
  try {
    const cfg = await (await fetch("/api/config")).json();
    if (cfg.turnstileSiteKey) await loadTurnstile(cfg.turnstileSiteKey);
  } catch {
    showNotice("The live demo could not load its settings. Saved answers still work.");
  }
}
init();
