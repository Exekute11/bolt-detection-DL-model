/* Camera & Image panel: HD resolution, FPS, exposure meter, auto-fix, software + hardware controls */
(function () {
  const $ = s => document.querySelector(s);
  const post = (a, b) => fetch('/control/' + a, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b || {}) }).then(r => r.json());
  const toast = m => { const t = $('#toast'); t.textContent = m; t.classList.add('show'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove('show'), 2400); };
  let busy = {};                                           // sliders being dragged are not overwritten by polling

  const set = (id, v) => { const e = $(id); if (e && !busy[id] && document.activeElement !== e) e.value = v; };
  window.cameraRender = function (d) {
    const c = d.camera; if (!c) return;
    const m = c.mean, bad = c.clip > 8 || m > 72, dark = m < 25;
    $('#mFill').style.width = Math.min(m, 100) + '%';
    $('#expBadge').textContent = bad ? 'Too bright' : dark ? 'Too dark' : 'Exposure OK';
    $('#expBadge').className = 'exp ' + (bad || dark ? 'warn' : 'ok');
    $('#mText').textContent = `Brightness ${m}% · blown-out pixels ${c.clip}%` + (bad ? ' → press Auto-fix' : '');
    set('#camRes', c.width + 'x' + c.height); set('#camFps', String(c.fps));
    set('#swGamma', c.sw_gamma); set('#swBri', c.sw_brightness); set('#swCon', c.sw_contrast);
    $('#vGamma').textContent = c.sw_gamma; $('#vBri').textContent = c.sw_brightness; $('#vCon').textContent = c.sw_contrast;
    $('#swEnh').checked = c.sw_enhance;
    $('#hwAuto').checked = c.auto_exposure; $('#hwAf').checked = c.autofocus;
    set('#hwExp', c.exposure); set('#hwFoc', c.focus); $('#vExp').textContent = c.exposure; $('#vFoc').textContent = c.focus;
    $('#expRow').style.display = c.auto_exposure ? 'none' : '';
    $('#focRow').style.display = c.autofocus ? 'none' : '';
    $('#hwBox').style.opacity = c.webcam ? 1 : .45;
    set('#camIdx', String(c.index)); set('#camDrv', c.backend);
    const i = c.info, low = c.webcam && i.width && i.width < c.width * 0.95;
    $('#camNow').textContent = c.webcam && i.width
      ? `Camera #${i.index} · ${i.backend} · ${i.width}×${i.height} ${i.fourcc} · camera delivers ${i.cam_fps} FPS · video ${Math.round(d.fps)} FPS`
      : (c.webcam ? '' : 'Resolution/FPS apply to the webcam source');
    $('#camWarn').textContent = i.note ? i.note : low ? `This camera/driver only gives ${i.width}×${i.height}. Try another Driver or Camera #.`
      : (c.webcam && i.fourcc && i.fourcc !== 'MJPG' && i.width >= 1280 ? 'Camera is not in MJPG mode - FPS will be low. Try another Driver.' : '');
  };

  const send = (b, msg) => post('camera', b).then(d => { if (msg) toast(msg); window.cameraRender(d); });
  let timer = {};
  function slider(id, key, num) {
    const e = $(id);
    e.oninput = () => { busy[id] = true; clearTimeout(timer[id]);
      timer[id] = setTimeout(() => { send({ [key]: num ? +e.value : e.value }); }, 120); };
    e.onchange = () => { busy[id] = false; };
    e.onmouseup = e.ontouchend = () => setTimeout(() => { busy[id] = false; }, 400);
  }
  slider('#swGamma', 'sw_gamma', 1); slider('#swBri', 'sw_brightness', 1); slider('#swCon', 'sw_contrast', 1);
  slider('#hwExp', 'exposure', 1); slider('#hwFoc', 'focus', 1);
  $('#camRes').onchange = e => { const [w, h] = e.target.value.split('x').map(Number); send({ width: w, height: h }, 'Applying ' + w + '×' + h + '…'); };
  $('#camFps').onchange = e => send({ fps: +e.target.value }, 'Applying ' + e.target.value + ' FPS…');
  $('#camIdx').onchange = e => send({ index: +e.target.value }, 'Switching camera…');
  $('#camDrv').onchange = e => send({ backend: e.target.value }, 'Switching driver…');
  $('#swEnh').onchange = e => send({ sw_enhance: e.target.checked });
  $('#hwAuto').onchange = e => send({ auto_exposure: e.target.checked });
  $('#hwAf').onchange = e => send({ autofocus: e.target.checked });
  $('#btnSharp').onclick = () => post('camera_sharp').then(d => { toast(d.message); window.cameraRender(d); });
  $('#btnAutoFix').onclick = () => post('camera_autotune').then(d => { toast(d.message); window.cameraRender(d); });
  $('#btnCamReset').onclick = () => post('camera_reset').then(d => { toast('Camera reset'); window.cameraRender(d); });
})();
