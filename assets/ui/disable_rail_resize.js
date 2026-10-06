(function () {
  var doc = window.parent.document;
  function sweep() {
    var all = doc.querySelectorAll('div');
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if (el.dataset.kcResizeOff) continue;
      var cur = window.parent.getComputedStyle(el).cursor;
      if (cur === 'col-resize' || cur === 'ew-resize') {
        el.dataset.kcResizeOff = '1';
        el.style.setProperty('display', 'none', 'important');
        el.style.setProperty('pointer-events', 'none', 'important');
        el.style.setProperty('cursor', 'default', 'important');
      }
    }
  }
  sweep();
  var pending = null;
  new window.parent.MutationObserver(function () {
    if (pending) return;
    pending = window.setTimeout(function () { pending = null; sweep(); }, 80);
  }).observe(doc.body, {childList: true, subtree: true});
})();
