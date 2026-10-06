(function () {
  var doc = window.parent.document;
  function measure() {
    var rail = doc.querySelector('[data-testid="stSidebar"]');
    // Collapsed rails are translated off-screen rather than removed,
    // so the right edge -- clamped at 0 -- is what the chat area
    // actually starts at, in both states.
    var w = rail ? Math.max(0, rail.getBoundingClientRect().right) : 0;
    doc.documentElement.style.setProperty('--kc-rail-w', w + 'px');
  }
  function track() {
    var start = null;
    function step(t) {
      if (start === null) start = t;
      measure();
      if (t - start < 600) window.requestAnimationFrame(step);
    }
    window.requestAnimationFrame(step);
  }
  measure();
  var rail = doc.querySelector('[data-testid="stSidebar"]');
  if (rail) {
    new window.parent.ResizeObserver(measure).observe(rail);
    rail.addEventListener('transitionstart', track);
  }
  // The collapse control lives outside the rail, and clicking it is
  // the one case where the rail is about to move before any observer
  // on it has fired.
  doc.addEventListener('click', function (e) {
    if (e.target.closest('[data-testid="stSidebarCollapseButton"], '
                         + '[data-testid="stSidebarCollapsedControl"]')) {
      track();
    }
  }, true);
  window.parent.addEventListener('resize', measure);
})();
