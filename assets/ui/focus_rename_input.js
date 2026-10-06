(function () {
  var doc = window.parent.document;
  var tries = 0;
  function grab() {
    var el = doc.querySelector(
      '[data-testid="stSidebar"] [class*="st-key-kc_rail_editing_"] input'
    );
    if (el) { el.focus(); el.select(); return; }
    if (tries++ < 20) window.setTimeout(grab, 50);
  }
  grab();
})();
