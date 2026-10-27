/* Shared UI animations: sliding toggle indicator, page transitions, video switch effect */
(function () {
  const $$ = s => document.querySelectorAll(s);

  /* 1. Sliding pill indicator for every .seg toggle (follows the button with class "on") */
  function place(seg) {
    let ind = seg.querySelector('.ind');
    if (!ind) { ind = document.createElement('i'); ind.className = 'ind'; seg.prepend(ind); }
    const on = seg.querySelector('button.on');
    if (!on) return;
    ind.style.width = on.offsetWidth + 'px';
    ind.style.transform = 'translateX(' + on.offsetLeft + 'px)';
  }
  function initSeg(seg) {
    place(seg);
    new MutationObserver(() => place(seg)).observe(seg, { subtree: true, attributes: true, attributeFilter: ['class'] });
  }
  function setup() {
    $$('.seg').forEach(initSeg);
    requestAnimationFrame(() => $$('.seg').forEach(s => s.classList.add('ready'))); /* enable transition after first placement */
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setup); else setup();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => $$('.seg').forEach(place));
  window.addEventListener('resize', () => $$('.seg').forEach(place));

  /* 2. Page transition: content fades/slides out, next page animates in */
  document.addEventListener('click', e => {
    const a = e.target.closest('a[href]');
    if (!a || a.target === '_blank' || a.hasAttribute('download') || e.ctrlKey || e.metaKey || e.shiftKey || e.button) return;
    const u = new URL(a.href, location.href);
    if (u.origin !== location.origin || u.pathname === location.pathname || u.pathname.endsWith('.csv')) return;
    e.preventDefault();
    document.body.classList.add('leaving');
    setTimeout(() => { location.href = u.href; }, 260);
  });
  window.addEventListener('pageshow', () => document.body.classList.remove('leaving')); /* back-button cache */

  /* 3. Video stage dims + blurs briefly when the source is switched */
  document.addEventListener('click', e => {
    const b = e.target.closest('#srcSeg button');
    const stage = document.getElementById('stage');
    if (!b || !stage || b.classList.contains('on')) return;
    stage.classList.add('switching');
    setTimeout(() => stage.classList.remove('switching'), 900);
  });
})();
