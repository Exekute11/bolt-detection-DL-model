/* Video stage: exact aspect ratio (no side bars), FPS tooltip, detection-zone drawing */
(function () {
  const $ = s => document.querySelector(s);
  const stage = $('#stage'), layer = document.createElement('div'), rect = document.createElement('div');
  layer.className = 'zone-layer'; rect.className = 'zone-rect'; rect.style.display = 'none';
  layer.innerHTML = '<div class="zone-hint">Drag on the video to draw the detection zone &middot; Esc to cancel</div>';
  layer.appendChild(rect); stage.appendChild(layer);
  const post = (a, b) => fetch('/control/' + a, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b || {}) }).then(r => r.json());
  const toast = m => { const t = $('#toast'); t.textContent = m; t.classList.add('show'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove('show'), 2400); };

  const om = document.createElement('div'); om.className = 'open-msg';
  om.innerHTML = '<i class="spin"></i><b id="openTxt">Opening camera…</b><small>This can take up to 10 seconds</small>';
  stage.appendChild(om);

  window.stageRender = function (d) {
    om.classList.toggle('show', !!d.opening);                 /* shown while the new source is opening */
    stage.classList.toggle('opening', !!d.opening);
    if (d.opening) $('#openTxt').textContent = d.opening === 'webcam' ? 'Opening camera…' : 'Opening video file…';
    if (d.frame_w && d.frame_h) {                       /* stage takes the camera's exact shape */
      const ar = d.frame_w + ' / ' + d.frame_h;
      if (stage.dataset.ar !== ar) { stage.dataset.ar = ar; stage.style.aspectRatio = ar; }
    }
    const pill = $('#fps').parentElement;
    pill.title = 'Video ' + Math.round(d.fps) + ' FPS · AI detection ' + d.det_fps + ' FPS';
    $('#btnZoneClr').disabled = !d.roi;
    $('#btnZone').classList.toggle('on', !!d.roi);
  };

  let drawing = false, start = null;
  const norm = e => { const r = stage.getBoundingClientRect();
    return { x: Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1), y: Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1) }; };
  const stop = () => { layer.classList.remove('on'); rect.style.display = 'none'; drawing = false; start = null; };

  $('#btnZone').onclick = () => { layer.classList.add('on'); toast('Drag on the video to draw the zone'); };
  $('#btnZoneClr').onclick = () => post('roi', { clear: true }).then(d => toast(d.message));
  layer.onmousedown = e => { start = norm(e); drawing = true; rect.style.display = 'block'; e.preventDefault(); };
  layer.onmousemove = e => {
    if (!drawing) return; const p = norm(e);
    rect.style.left = Math.min(start.x, p.x) * 100 + '%'; rect.style.top = Math.min(start.y, p.y) * 100 + '%';
    rect.style.width = Math.abs(p.x - start.x) * 100 + '%'; rect.style.height = Math.abs(p.y - start.y) * 100 + '%';
  };
  window.addEventListener('mouseup', e => {
    if (!drawing) return; const p = norm(e), s = start; stop();
    post('roi', { x1: s.x, y1: s.y, x2: p.x, y2: p.y }).then(d => toast(d.message));
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') stop(); });
})();
