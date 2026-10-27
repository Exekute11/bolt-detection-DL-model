/* Inspection Cycle panel: timed mode, ring timer, part library, yield stats, result sound */
(function () {
  const $ = s => document.querySelector(s);
  const post = (a, b) => fetch('/control/' + a, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b || {}) }).then(r => r.json());
  const C = 2 * Math.PI * 34;
  const bar = $('#ringBar'); bar.style.strokeDasharray = C;
  let lastResult = -1, partsKey = '', typing = false;

  function beep(pass) {
    if (!$('#sndOn').checked) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const tone = (f, t0, d) => { const o = ctx.createOscillator(), g = ctx.createGain(); o.frequency.value = f; o.type = 'sine';
        g.gain.setValueAtTime(.2, ctx.currentTime + t0); g.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + t0 + d);
        o.connect(g); g.connect(ctx.destination); o.start(ctx.currentTime + t0); o.stop(ctx.currentTime + t0 + d); };
      if (pass) { tone(880, 0, .18); tone(1320, .2, .25); } else { tone(300, 0, .3); tone(300, .4, .3); tone(300, .8, .3); }
    } catch (e) {}
  }

  window.cycleRender = function (d) {
    const c = d.cycle; if (!c) return;
    const on = c.enabled, st = c.state, r = c.result;
    document.querySelectorAll('#cycSeg button').forEach(b => b.classList.toggle('on', (b.dataset.m === '1') === on));
    $('#cycPanel').dataset.state = on ? st : 'LIVE';
    if (document.activeElement !== $('#cycDur')) $('#cycDur').value = c.duration;

    let big = '--', small = '', title = 'Live mode', hint = 'Counting continuously. Switch to Timed to inspect one part for a fixed time.', prog = 0;
    if (on) {
      if (!d.running) { title = 'Paused'; hint = 'Press Start to continue.'; big = '||'; }
      else if (st === 'IDLE') { title = 'Waiting for part'; hint = `Place a part in front of the camera. It will be inspected for ${c.duration} s.`; big = '⏳'; }
      else if (st === 'INSPECTING') { title = 'Inspecting…'; hint = 'Hold the part steady until the timer finishes.'; big = c.remaining.toFixed(1); small = 'sec'; prog = c.progress; }
      else if (st === 'RESULT' && r) {
        prog = 1; big = r.verdict === 'PASS' ? '✓' : r.verdict === 'FAIL' ? '✕' : '?';
        title = r.verdict === 'PASS' ? 'PASS' : r.verdict === 'FAIL' ? 'FAIL' : 'No reference';
        hint = r.verdict === 'NO_REF' ? `Counted ${r.count}. Capture a reference to get PASS/FAIL.`
          : `${r.count} of ${r.ref} fasteners · ${r.stability}% stable${r.verdict === 'FAIL' ? ' · ' + r.missing + ' missing' : ''}. Remove the part to continue.`;
      }
    }
    $('#ringBig').textContent = big; $('#ringSmall').textContent = small;
    $('#cycState').textContent = title; $('#cycHint').textContent = hint;
    bar.style.strokeDashoffset = C * (1 - prog);
    const rs = $('#ring'); rs.className = 'ring ' + (on && st === 'RESULT' && r ? 'res-' + r.verdict : '');

    /* banner override while a verdict is on screen */
    if (on && st === 'RESULT' && r && r.verdict !== 'NO_REF') {
      const bn = $('#banner'), ok = r.verdict === 'PASS';
      bn.className = 'banner ' + (ok ? 'ok' : 'alert');
      $('#bTag').textContent = r.verdict;
      $('#bTitle').textContent = ok ? 'Part Passed Inspection' : 'Part Failed Inspection';
      $('#bText').textContent = ok ? `All ${r.ref} fasteners present.` : `${r.missing} fastener(s) missing (found ${r.count} of ${r.ref}).`;
    }

    if (r && r.id !== lastResult) { if (lastResult !== -1 && r.verdict !== 'NO_REF') beep(r.verdict === 'PASS'); lastResult = r.id; }
    if (r === null && lastResult === -1) lastResult = 0;

    const p = c.stats.pass, f = c.stats.fail, tot = p + f;
    $('#stTot').textContent = tot; $('#stPass').textContent = p; $('#stFail').textContent = f;
    $('#stYield').textContent = tot ? Math.round(100 * p / tot) + '%' : '--';

    const key = JSON.stringify(d.parts) + '|' + d.active_part;
    if (key !== partsKey && document.activeElement !== $('#partSel')) {
      partsKey = key;
      $('#partSel').innerHTML = '<option value="">— none —</option>' +
        Object.entries(d.parts).map(([n, v]) => `<option value="${n.replace(/"/g, '&quot;')}"${n === d.active_part ? ' selected' : ''}>${n} (${v})</option>`).join('');
    }
  };

  const toast = m => { const t = $('#toast'); t.textContent = m; t.classList.add('show'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove('show'), 2400); };
  const done = d => { if (d.message) toast(d.message); window.cycleRender(d); };

  document.querySelectorAll('#cycSeg button').forEach(b => b.onclick = () => post('cycle_mode', { enabled: b.dataset.m === '1' }).then(done));
  $('#cycDur').onchange = () => post('cycle_config', { duration: $('#cycDur').value }).then(done);
  $('#partSel').onchange = () => post('part_select', { name: $('#partSel').value }).then(d => { partsKey = ''; done(d); });
  $('#btnSavePart').onclick = () => { const n = prompt('Part name / number (current reference will be saved):'); if (n) post('part_save', { name: n }).then(d => { partsKey = ''; done(d); }); };
  $('#btnDelPart').onclick = () => { const n = $('#partSel').value; if (n && confirm('Delete part ' + n + '?')) post('part_delete', { name: n }).then(d => { partsKey = ''; done(d); }); };
})();
