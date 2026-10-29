'use strict';

const S = {
  samples: [],
  latest: [],
  proc: {},
  epoch: '',
  exe: '',
  logSeq: 0,
  metric: 'power',
  windowSec: 300,
  maxSamples: 3600,
  gpuCount: 0,
  es: null,
  reconnectTimer: null,
  autoFilled: false,
  gpuiDirty: false,
  sectionsDirty: false,
};

const COLORS = ['#ffb020', '#4cc9f0', '#f72585', '#7bd88f', '#b388ff', '#ff8fab', '#f9c74f', '#90e0ef'];

const METRICS = {
  power:  { label: 'power',      unit: 'W',   get: g => g.power_draw, fixed: null },
  temp:   { label: 'temperature', unit: '°C', get: g => g.temp,       fixed: [10, 100] },
  util:   { label: 'utilization', unit: '%',  get: g => g.util_gpu,   fixed: [0, 100] },
  clocks: { label: 'core clock',  unit: 'MHz', get: g => g.clock_sm,  fixed: null },
  mem:    { label: 'memory used', unit: 'MiB', get: g => g.mem_used,  fixed: null },
};

const THROTTLE_BITS = [
  [0x1, 'idle'], [0x2, 'app clocks'], [0x4, 'SW power cap'], [0x8, 'HW slowdown'],
  [0x10, 'sync boost'], [0x20, 'SW thermal'], [0x40, 'HW thermal'],
  [0x80, 'HW power brake'], [0x100, 'display clock'], [0x400, 'idle'],
];

const el = id => document.getElementById(id);
const nf = (v, d = 0) => (v == null || !isFinite(v)) ? '—' : Number(v).toFixed(d);
const colorFor = i => COLORS[i % COLORS.length];

function hms(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return (h ? h + ':' : '') + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}

function clockTime(t) {
  const d = new Date(t);
  return String(d.getHours()).padStart(2, '0') + ':' +
         String(d.getMinutes()).padStart(2, '0') + ':' +
         String(d.getSeconds()).padStart(2, '0');
}

function throttleText(hex) {
  if (!hex) return 'none';
  const v = parseInt(hex, 16);
  if (!isFinite(v) || v === 0) return 'none';
  const names = [];
  for (const [bit, name] of THROTTLE_BITS) if (v & bit) names.push(name);
  const uniq = [...new Set(names)];
  if (uniq.length === 0) return '0x' + v.toString(16);
  return uniq.join(', ');
}

// ------------------------------------------------------------------- canvas

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  if (w < 10 || h < 10) return null;
  const cw = Math.round(w * dpr), ch = Math.round(h * dpr);
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function niceStep(range, target) {
  const raw = range / Math.max(1, target);
  const pow = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const f = raw / pow;
  const n = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return n * pow;
}

function drawChart(canvas, series, opts) {
  const c = setupCanvas(canvas);
  if (!c) return;
  const { ctx, w, h } = c;
  const L = 56, R = 14, T = 12, B = 26;
  const pw = w - L - R, ph = h - T - B;
  const now = S.samples.length ? S.samples[S.samples.length - 1].t : Date.now();
  const span = opts.windowSec * 1000;
  const x0 = now - span, x1 = now;

  let lo = Infinity, hi = -Infinity, any = false;
  for (const s of series) {
    for (const p of s.pts) {
      if (p[0] < x0) continue;
      any = true;
      if (p[1] < lo) lo = p[1];
      if (p[1] > hi) hi = p[1];
    }
  }
  if (!any) { lo = 0; hi = 1; }
  if (opts.fixed) {
    lo = opts.fixed[0]; hi = opts.fixed[1];
  } else if (any) {
    const pad = (hi - lo) * 0.12 || Math.max(1, Math.abs(hi) * 0.08);
    lo = Math.max(0, lo - pad);
    hi = hi + pad;
  }
  if (hi <= lo) hi = lo + 1;

  const X = t => L + ((t - x0) / span) * pw;
  const Y = v => T + (1 - (v - lo) / (hi - lo)) * ph;

  ctx.font = '11px "Cascadia Mono", Consolas, monospace';
  ctx.textBaseline = 'middle';

  // horizontal grid + y labels
  const step = niceStep(hi - lo, 4);
  const start = Math.ceil(lo / step) * step;
  const dec = step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3;
  ctx.strokeStyle = '#1b2533';
  ctx.fillStyle = '#7d8a9c';
  ctx.lineWidth = 1;
  for (let v = start; v <= hi + 1e-9; v += step) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(L, y);
    ctx.lineTo(L + pw, y);
    ctx.stroke();
    const txt = Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + 'k' : v.toFixed(dec);
    ctx.textAlign = 'right';
    ctx.fillText(txt, L - 8, y);
  }

  // vertical grid + time labels
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const t = x0 + (span * i) / xTicks;
    const x = Math.round(X(t)) + 0.5;
    ctx.strokeStyle = '#141d28';
    ctx.beginPath();
    ctx.moveTo(x, T);
    ctx.lineTo(x, T + ph);
    ctx.stroke();
    ctx.fillStyle = '#7d8a9c';
    ctx.textAlign = i === 0 ? 'left' : i === xTicks ? 'right' : 'center';
    ctx.fillText(clockTime(t), x, T + ph + 14);
  }

  // series
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const s of series) {
    if (s.pts.length < 2) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let started = false;
    for (const p of s.pts) {
      if (p[0] < x0 - span * 0.1) continue;
      const x = X(p[0]), y = Y(p[1]);
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    }
    ctx.stroke();

    const last = s.pts[s.pts.length - 1];
    if (last && last[0] >= x0) {
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(X(last[0]), Y(last[1]), 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // axis lines
  ctx.strokeStyle = '#2a3646';
  ctx.beginPath();
  ctx.moveTo(L + 0.5, T);
  ctx.lineTo(L + 0.5, T + ph + 0.5);
  ctx.lineTo(L + pw, T + ph + 0.5);
  ctx.stroke();
}

function drawSpark(canvas, pts, color, fixed) {
  const c = setupCanvas(canvas);
  if (!c) return;
  const { ctx, w, h } = c;
  const pad = 3;
  if (pts.length < 2) return;

  let lo = Infinity, hi = -Infinity;
  for (const p of pts) {
    if (p[1] < lo) lo = p[1];
    if (p[1] > hi) hi = p[1];
  }
  if (fixed) { lo = fixed[0]; hi = fixed[1]; }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  if (fixed == null) {
    const p = (hi - lo) * 0.15;
    lo -= p; hi += p;
  }

  const t0 = pts[0][0], t1 = pts[pts.length - 1][0];
  const tsp = Math.max(1, t1 - t0);
  const X = t => pad + ((t - t0) / tsp) * (w - pad * 2);
  const Y = v => pad + (1 - (v - lo) / (hi - lo)) * (h - pad * 2);

  ctx.beginPath();
  ctx.moveTo(X(pts[0][0]), Y(pts[0][1]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(X(pts[i][0]), Y(pts[i][1]));
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // soft fill
  ctx.lineTo(X(pts[pts.length - 1][0]), h - pad);
  ctx.lineTo(X(pts[0][0]), h - pad);
  ctx.closePath();
  ctx.globalAlpha = 0.10;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
}

function seriesFor(metric) {
  const def = METRICS[metric];
  const n = S.gpuCount;
  const out = [];
  for (let i = 0; i < n; i++) {
    const pts = [];
    for (const s of S.samples) {
      const g = (s.gpus || []).find(x => x.index === i);
      if (!g) continue;
      const v = def.get(g);
      if (v == null) continue;
      pts.push([s.t, v]);
    }
    out.push({ name: 'GPU ' + i, color: colorFor(i), pts });
  }
  return out;
}

// ------------------------------------------------------------------ rendering

function render() {
  renderHeader();
  renderErr();
  renderCombined();
  ensureCards();
  renderCards();
  ensurePowerRows();
  renderPower();
  renderFormState();
}

function renderHeader() {
  const p = S.proc || {};
  const pill = el('pill');
  if (p.running) {
    pill.className = 'pill running';
    pill.textContent = 'running';
  } else if (p.stopped) {
    pill.className = 'pill idle';
    pill.textContent = 'stopped';
  } else if (p.exit_code != null) {
    pill.className = 'pill exited';
    pill.textContent = 'exited ' + p.exit_code;
  } else {
    pill.className = 'pill idle';
    pill.textContent = 'idle';
  }
  el('elapsed').textContent = p.started ? hms(p.elapsed) : '—';
  el('rate').textContent = p.rate || '';
  el('gpuCount').textContent = S.gpuCount ? S.gpuCount + (S.gpuCount === 1 ? ' GPU' : ' GPUs') : '';
  const fb = el('foundBadge');
  fb.classList.toggle('hidden', !p.match);
  fb.title = p.found ? 'scalar ' + p.found : 'match reported by the sweep';
}

function renderErr() {
  const bar = el('errBar');
  const msg = S.err || '';
  if (!msg) { bar.classList.add('hidden'); bar.textContent = ''; return; }
  bar.classList.remove('hidden');
  bar.textContent = 'nvidia-smi: ' + msg;
}

function renderCombined() {
  const series = seriesFor(S.metric);
  drawChart(el('combined'), series, { windowSec: S.windowSec, fixed: METRICS[S.metric].fixed });
  const legend = el('legend');
  legend.textContent = '';
  if (!series.length) {
    const s = document.createElement('span');
    s.className = 'muted';
    s.textContent = 'waiting for GPU data…';
    legend.appendChild(s);
    return;
  }
  for (const s of series) {
    const wrap = document.createElement('span');
    wrap.className = 'chip';
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = s.color;
    const name = document.createElement('span');
    name.textContent = s.name;
    const val = document.createElement('b');
    const last = s.pts.length ? s.pts[s.pts.length - 1][1] : null;
    val.textContent = last == null ? '—' : (Math.abs(last) >= 1000 ? (last / 1000).toFixed(2) + 'k' : last.toFixed(0));
    const unit = document.createElement('span');
    unit.textContent = ' ' + METRICS[S.metric].unit;
    wrap.append(sw, name, val, unit);
    legend.appendChild(wrap);
  }
}

let cardCount = 0;

function ensureCards() {
  const grid = el('gpuGrid');
  if (cardCount === S.gpuCount) return;
  grid.textContent = '';
  cardCount = S.gpuCount;

  for (let i = 0; i < S.gpuCount; i++) {
    const card = document.createElement('section');
    card.className = 'gpu';

    const head = document.createElement('div');
    head.className = 'gpu-head';
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = colorFor(i);
    const idx = document.createElement('span');
    idx.className = 'idx';
    idx.textContent = 'GPU ' + i;
    const name = document.createElement('span');
    name.className = 'name';
    name.id = 'g' + i + '-name';
    head.append(dot, idx, name);

    const stats = document.createElement('div');
    stats.className = 'gpu-stats';
    const statDefs = [
      ['temp', 'temp', '°C'],
      ['power', 'power', 'W'],
      ['util', 'util', '%'],
      ['mem', 'mem', 'GiB'],
    ];
    for (const [key, label] of statDefs) {
      const box = document.createElement('div');
      box.className = 'stat';
      const l = document.createElement('label');
      l.textContent = label;
      const b = document.createElement('b');
      b.id = 'g' + i + '-' + key;
      b.textContent = '—';
      const sm = document.createElement('small');
      sm.id = 'g' + i + '-' + key + '-sub';
      box.append(l, b, sm);
      stats.appendChild(box);
    }

    const sparks = document.createElement('div');
    sparks.className = 'sparks';
    const sparkDefs = [['temp', 'temp'], ['power', 'power'], ['util', 'util'], ['mem', 'mem']];
    for (const [key, label] of sparkDefs) {
      const box = document.createElement('div');
      box.className = 'spark-box';
      const lbl = document.createElement('div');
      lbl.className = 'lbl';
      const a = document.createElement('span');
      a.textContent = label;
      const b2 = document.createElement('span');
      b2.id = 'g' + i + '-spark-' + key + '-now';
      lbl.append(a, b2);
      const cv = document.createElement('canvas');
      cv.id = 'g' + i + '-spark-' + key;
      box.append(lbl, cv);
      sparks.appendChild(box);
    }

    const meta = document.createElement('div');
    meta.className = 'gpu-meta';
    for (const k of ['clock', 'pcie', 'throttle']) {
      const sp = document.createElement('span');
      sp.id = 'g' + i + '-' + k;
      meta.appendChild(sp);
    }

    card.append(head, stats, sparks, meta);
    grid.appendChild(card);
  }
}

function renderCards() {
  for (let i = 0; i < S.gpuCount; i++) {
    const g = (S.latest || []).find(x => x.index === i);
    if (!g) continue;

    const name = el('g' + i + '-name');
    if (name) name.textContent = g.name || '';

    const temp = el('g' + i + '-temp');
    if (temp) temp.textContent = nf(g.temp);
    const tempSub = el('g' + i + '-temp-sub');
    if (tempSub) tempSub.textContent = ' °C';

    const power = el('g' + i + '-power');
    if (power) power.textContent = nf(g.power_draw);
    const powerSub = el('g' + i + '-power-sub');
    if (powerSub) powerSub.textContent = g.power_limit != null ? '/ ' + nf(g.power_limit) + ' W' : 'W';

    const util = el('g' + i + '-util');
    if (util) util.textContent = nf(g.util_gpu);
    const utilSub = el('g' + i + '-util-sub');
    if (utilSub) utilSub.textContent = ' %';

    const mem = el('g' + i + '-mem');
    if (mem) mem.textContent = g.mem_used != null ? (g.mem_used / 1024).toFixed(1) : '—';
    const memSub = el('g' + i + '-mem-sub');
    if (memSub) memSub.textContent = g.mem_total != null ? ' / ' + (g.mem_total / 1024).toFixed(0) + ' GiB' : ' GiB';

    const clock = el('g' + i + '-clock');
    if (clock) clock.textContent = 'SM ' + nf(g.clock_sm) + ' / MEM ' + nf(g.clock_mem) + ' MHz';

    const pcie = el('g' + i + '-pcie');
    if (pcie) pcie.textContent = 'PCIe ' + (g.pcie_gen || '?') + '×' + (g.pcie_width || '?');

    const th = el('g' + i + '-throttle');
    if (th) {
      const txt = throttleText(g.throttle);
      th.textContent = 'throttle: ' + txt;
      th.className = 'throttle' + (txt === 'none' ? ' none' : '');
    }

    const defs = [
      ['temp', g.temp, null, '°C'],
      ['power', g.power_draw, null, 'W'],
      ['util', g.util_gpu, [0, 100], '%'],
      ['mem', g.mem_used, null, 'MiB'],
    ];
    for (const [key, val, fixed, unit] of defs) {
      const cv = el('g' + i + '-spark-' + key);
      const now = el('g' + i + '-spark-' + key + '-now');
      if (now) now.textContent = val == null ? '' : (key === 'mem' ? (val / 1024).toFixed(1) + ' GiB' : nf(val) + ' ' + unit);

      const pts = [];
      for (const s of S.samples) {
        const gg = (s.gpus || []).find(x => x.index === i);
        if (!gg) continue;
        const v = key === 'temp' ? gg.temp : key === 'power' ? gg.power_draw : key === 'util' ? gg.util_gpu : gg.mem_used;
        if (v == null) continue;
        pts.push([s.t, v]);
      }
      if (cv) drawSpark(cv, pts, colorFor(i), fixed);
    }
  }
}

let powerCount = -1;

function ensurePowerRows() {
  if (powerCount === S.gpuCount) return;
  powerCount = S.gpuCount;
  const rows = el('powerRows');
  rows.textContent = '';
  if (!S.gpuCount) {
    const s = document.createElement('span');
    s.className = 'muted';
    s.textContent = 'waiting for GPU data…';
    rows.appendChild(s);
    return;
  }
  for (let i = 0; i < S.gpuCount; i++) {
    const g = (S.latest || []).find(x => x.index === i) || {};
    const row = document.createElement('div');
    row.className = 'power-row';

    const name = document.createElement('span');
    name.className = 'gname';
    name.textContent = 'GPU ' + i + (g.name ? ' · ' + g.name : '');

    const input = document.createElement('input');
    input.type = 'number';
    input.min = '50';
    input.max = '1500';
    input.id = 'p' + i;
    input.value = g.power_limit != null ? String(Math.round(g.power_limit)) : '489';

    const cur = document.createElement('span');
    cur.className = 'cur';
    cur.id = 'p' + i + '-cur';
    cur.textContent = g.power_limit != null ? 'now ' + Math.round(g.power_limit) + ' W' : '';

    const w = document.createElement('span');
    w.className = 'cur';
    w.textContent = 'W';

    row.append(name, input, w, cur);
    rows.appendChild(row);
  }
}

function renderPower() {
  for (let i = 0; i < S.gpuCount; i++) {
    const g = (S.latest || []).find(x => x.index === i);
    const cur = el('p' + i + '-cur');
    if (cur && g) cur.textContent = g.power_limit != null ? 'now ' + Math.round(g.power_limit) + ' W' : '';
  }
}

function renderFormState() {
  const running = !!(S.proc && S.proc.running);
  el('startBtn').disabled = running;
  el('stopBtn').disabled = !running;
  const args = (S.proc && S.proc.args) || [];
  el('cmdLine').textContent = args.length ? ((S.exe ? S.exe + ' ' : '') + args.join(' ')) : '';
}

// --------------------------------------------------------------------- logs

function appendLog(lines) {
  if (!lines || !lines.length) return;
  const logEl = el('log');
  const frag = document.createDocumentFragment();
  for (const line of lines) {
    const span = document.createElement('span');
    span.textContent = line + '\n';
    if (line.startsWith('$ ')) span.className = 'l-cmd';
    else if (/FOUND|MATCH\s*:\s*YES/.test(line)) span.className = 'l-found';
    else if (/error|fatal|failed|cannot|invalid/i.test(line)) span.className = 'l-err';
    frag.appendChild(span);
  }
  logEl.appendChild(frag);
  let extra = logEl.childNodes.length - 4000;
  while (extra-- > 0 && logEl.firstChild) logEl.removeChild(logEl.firstChild);
  if (el('autoscroll').checked) logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  el('log').textContent = '';
}

function takeLog(m) {
  if (!m.log) return;
  appendLog(m.log.lines);
  if (typeof m.log.seq === 'number') S.logSeq = m.log.seq;
}

// ---------------------------------------------------------------------- sse

function connect() {
  if (S.es) { S.es.close(); S.es = null; }
  if (S.reconnectTimer) { clearTimeout(S.reconnectTimer); S.reconnectTimer = null; }

  const es = new EventSource('/api/events?logsince=' + encodeURIComponent(S.logSeq));
  S.es = es;

  es.onmessage = ev => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    if (!m || !m.type) return;

    if (S.epoch && m.epoch && m.epoch !== S.epoch) {
      // server restarted: its log/history counters are new
      S.epoch = m.epoch;
      S.logSeq = 0;
      S.samples = [];
      clearLog();
      connect();
      return;
    }
    if (m.epoch) S.epoch = m.epoch;

    if (m.type === 'hello') {
      S.samples = (m.samples || []).slice(-S.maxSamples);
      S.latest = m.gpus || [];
      S.proc = m.proc || {};
      S.err = m.error || '';
      S.gpuCount = m.gpu_count || (S.latest.length);
      if (m.exe) S.exe = m.exe;
      autoFillTargets();
      takeLog(m);
      render();
    } else if (m.type === 'tick') {
      if (m.gpus && m.gpus.length) {
        S.samples.push({ t: m.t, gpus: m.gpus });
        if (S.samples.length > S.maxSamples) S.samples.splice(0, S.samples.length - S.maxSamples);
        S.latest = m.gpus;
        S.gpuCount = m.gpu_count || m.gpus.length;
      }
      S.proc = m.proc || S.proc;
      S.err = m.error || '';
      takeLog(m);
      render();
    }
  };

  es.onerror = () => {
    es.close();
    if (S.es === es) S.es = null;
    if (!S.reconnectTimer) {
      S.reconnectTimer = setTimeout(() => { S.reconnectTimer = null; connect(); }, 2000);
    }
  };
}

function autoFillTargets() {
  if (S.autoFilled || !S.gpuCount) return;
  S.autoFilled = true;
  if (!S.gpuiDirty) {
    el('gpui').value = Array.from({ length: S.gpuCount }, (_, i) => i).join(',');
  }
  if (!S.sectionsDirty) {
    el('sections').value = String(S.gpuCount);
  }
}

// -------------------------------------------------------------------- actions

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  let j = {};
  try { j = await r.json(); } catch {}
  if (!r.ok) throw new Error((j && j.error) || ('HTTP ' + r.status));
  return j;
}

function setMsg(id, text, ok) {
  const m = el(id);
  m.textContent = text || '';
  m.className = 'msg' + (text ? (ok ? ' ok' : ' bad') : '');
}

async function onStart(ev) {
  ev.preventDefault();
  const body = {
    mode: el('mode').value,
    range: el('range').value.trim(),
    target: el('target').value.trim(),
    batch: parseInt(el('batch').value, 10) || 0,
    jobs: parseInt(el('jobs').value, 10) || 0,
    sections: parseInt(el('sections').value, 10) || 0,
    gpui: el('gpui').value.trim(),
    maxbatch: parseInt(el('maxbatch').value, 10) || 0,
    out: el('out').value.trim(),
    extra: el('extra').value.trim(),
  };
  if (!body.range || !body.target) {
    setMsg('runMsg', 'range and target are required', false);
    return;
  }
  el('startBtn').disabled = true;
  setMsg('runMsg', 'starting…', true);
  try {
    const p = await postJSON('/api/start', body);
    setMsg('runMsg', 'started, pid ' + (p.pid || '?'), true);
  } catch (e) {
    setMsg('runMsg', e.message, false);
    el('startBtn').disabled = false;
  }
}

async function onStop() {
  el('stopBtn').disabled = true;
  setMsg('runMsg', 'stopping…', true);
  try {
    await postJSON('/api/stop');
    setMsg('runMsg', 'stopped', true);
  } catch (e) {
    setMsg('runMsg', e.message, false);
    el('stopBtn').disabled = false;
  }
}

async function onApplyPower() {
  if (!S.gpuCount) return;
  const limits = [];
  for (let i = 0; i < S.gpuCount; i++) {
    const v = parseInt(el('p' + i).value, 10);
    if (!isFinite(v)) { setMsg('powerMsg', 'GPU ' + i + ': invalid value', false); return; }
    limits.push({ index: i, watts: v });
  }
  setMsg('powerMsg', 'applying…', true);
  try {
    const j = await postJSON('/api/power', { limits });
    const bad = (j.results || []).filter(r => !r.ok);
    if (bad.length === 0) {
      setMsg('powerMsg', 'applied to ' + limits.length + ' GPU(s)', true);
    } else {
      setMsg('powerMsg', bad.map(r => 'GPU ' + r.index + ': ' + r.output).join(' | '), false);
    }
  } catch (e) {
    setMsg('powerMsg', e.message, false);
  }
}

function fillAllPower() {
  const v = parseInt(el('allW').value, 10);
  if (!isFinite(v)) return;
  for (let i = 0; i < S.gpuCount; i++) {
    const inp = el('p' + i);
    if (inp) inp.value = String(v);
  }
}

// ---------------------------------------------------------------------- init

function init() {
  el('runForm').addEventListener('submit', onStart);
  el('stopBtn').addEventListener('click', onStop);
  el('applyPower').addEventListener('click', onApplyPower);
  el('fillPower').addEventListener('click', fillAllPower);

  el('gpui').addEventListener('input', () => { S.gpuiDirty = true; });
  el('sections').addEventListener('input', () => { S.sectionsDirty = true; });

  el('metricTabs').addEventListener('click', ev => {
    const b = ev.target.closest('button[data-metric]');
    if (!b) return;
    S.metric = b.dataset.metric;
    for (const t of el('metricTabs').querySelectorAll('.tab')) t.classList.toggle('active', t === b);
    renderCombined();
  });

  el('windowSel').addEventListener('change', () => {
    S.windowSec = parseInt(el('windowSel').value, 10) || 300;
    renderCombined();
  });

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    if (resizeTimer) return;
    resizeTimer = setTimeout(() => { resizeTimer = null; render(); }, 150);
  });

  connect();
}

document.addEventListener('DOMContentLoaded', init);
