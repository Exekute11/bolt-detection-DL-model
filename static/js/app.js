/* Godrej Fastener Vision - dashboard logic */
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const ICON_OK = '<svg viewBox="0 0 24 24"><path d="m5 12 5 5L20 7"/></svg>';
const ICON_WARN = '<svg viewBox="0 0 24 24"><path d="M12 3 2 21h20zM12 10v5M12 18v.5"/></svg>';
let lastEventId = -1, offline = false;

function toast(m) {
  const t = $('#toast');
  t.textContent = m;
  t.classList.add('show');
  clearTimeout(toast.t);
  toast.t = setTimeout(() => t.classList.remove('show'), 2400);
}

function setNum(el, v) {
  if (el.textContent !== String(v)) {
    el.textContent = v;
    el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop');
  }
}

setInterval(() => $('#clock').textContent = new Date().toLocaleTimeString(), 1000);

function render(d) {
  $$('[data-k]').forEach(e => setNum(e, d[e.dataset.k]));

  const alert = d.status === 'ALERT';
  const st = $('#statusKpi');
  st.textContent = d.running ? d.status : 'PAUSED';
  $('#missCard').classList.toggle('alert', alert && d.running);

  const b = $('#banner');
  b.className = 'banner ' + (!d.running ? 'idle' : alert ? 'alert' : 'ok');
  $('#bTag').textContent = !d.running ? 'PAUSED' : d.status;
  $('#bTitle').textContent = !d.running ? 'Monitoring Paused'
    : alert ? 'Attention: Fastener Missing' : 'All Systems Nominal';
  $('#bText').textContent = !d.running ? 'Press Start to resume live inspection.'
    : alert ? `${d.missing_bolts} fastener(s) below the reference of ${d.reference_bolts}.`
    : d.reference_locked ? `All ${d.reference_bolts} reference fasteners are present.`
    : 'Tracking live. Capture a reference to enable missing-fastener alerts.';
  $('#bIcon').innerHTML = alert && d.running ? ICON_WARN : ICON_OK;

  $('#refNote').textContent = d.reference_locked ? 'Locked reference' : 'Auto-following';
  $('#srcNote').textContent = 'Source: ' + d.source;
  $('#fps').textContent = Math.round(d.fps);
  $('#liveText').textContent = offline ? 'OFFLINE' : d.running ? 'LIVE' : 'PAUSED';
  $('#livePill').classList.toggle('off', offline || !d.running);
  $('#pausedOv').classList.toggle('show', !d.running);
  $('#btnStart').disabled = d.running;
  $('#btnStop').disabled = !d.running;
  $('#btnClr').disabled = !d.reference_locked;
  $$('#srcSeg button').forEach(x => x.classList.toggle('on', x.dataset.s === d.source));

  $('#actCount').textContent = d.active_ids.length;
  /* Re-render chips ONLY when the ID list really changes (stops flicker) */
  const idKey = d.active_ids.join(',');
  if (idKey !== render.idKey) {
    const old = new Set((render.idKey || '').split(',').filter(Boolean));
    render.idKey = idKey;
    $('#chips').innerHTML = d.active_ids.length
      ? d.active_ids.map(i => `<span class="chip${old.has(String(i)) ? ' keep' : ''}"><i></i>ID ${i}</span>`).join('')
      : '<p class="empty">Waiting for detections…</p>';
  }

  if (window.cycleRender) cycleRender(d);
  if (window.stageRender) stageRender(d);
  if (window.cameraRender) cameraRender(d);

  const topId = d.events.length ? d.events[0].id : 0;
  if (topId !== lastEventId) {
    const firstLoad = lastEventId === -1;
    lastEventId = topId;
    $('#timeline').innerHTML = d.events.length
      ? d.events.map(e => `<li class="${e.level}"><i></i><div><b>${e.msg}</b><small>${e.t}</small></div></li>`).join('')
      : '<li class="empty">No events yet</li>';
    if (!firstLoad) { const f = document.querySelector('#timeline li'); if (f) f.classList.add('new'); }
  }
}

async function poll() {
  try {
    const r = await fetch('/status', { cache: 'no-store' });
    render(await r.json());
    offline = false;
  } catch (e) {
    offline = true;
    $('#liveText').textContent = 'OFFLINE';
    $('#livePill').classList.add('off');
  }
  setTimeout(poll, 150);
}
poll();

async function act(action, body, okMsg) {
  try {
    const r = await fetch('/control/' + action, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
    const d = await r.json();
    render(d);
    toast(d.message || okMsg);
  } catch (e) {
    toast('Backend not reachable');
  }
}

$('#btnStart').onclick = () => act('start', null, 'Monitoring started');
$('#btnStop').onclick = () => act('stop', null, 'Monitoring paused');
$('#btnRef').onclick = () => act('capture_reference', null, 'Reference captured');
$('#btnClr').onclick = () => act('clear_reference', null, 'Reference cleared');
$$('#srcSeg button').forEach(b => b.onclick = () => act('source', { source: b.dataset.s }, 'Source: ' + b.dataset.s));

/* Fullscreen (with webkit fallback) */
function toggleFs() {
  const s = $('#stage');
  const active = document.fullscreenElement || document.webkitFullscreenElement;
  if (active) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  else (s.requestFullscreen || s.webkitRequestFullscreen).call(s);
}
$('#btnFs').onclick = toggleFs;
$('#stage').ondblclick = toggleFs;

/* Reconnect the MJPEG stream only if it actually drops */
$('#feed').onerror = () => setTimeout(() => { $('#feed').src = '/video_feed?t=' + Date.now(); }, 1500);
