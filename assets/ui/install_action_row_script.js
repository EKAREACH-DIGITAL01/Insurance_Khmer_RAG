(function () {
  var doc = window.parent.document;
  if (doc.body.dataset.kcActionsWired) return;
  doc.body.dataset.kcActionsWired = '1';
  doc.addEventListener('click', function (e) {
    var copy = e.target.closest('.kc-act-copy');
    if (copy) {
      var copyText = '';
      try {
        var raw = window.parent.atob(copy.dataset.copyB64 || '');
        var bytes = Uint8Array.from(raw, function (c) { return c.charCodeAt(0); });
        copyText = new TextDecoder('utf-8').decode(bytes);
      } catch (err) {
        copyText = '';
      }
      window.parent.navigator.clipboard
        .writeText(copyText)
        .then(function () {
          copy.classList.add('is-done');
          window.setTimeout(function () {
            copy.classList.remove('is-done');
          }, 1200);
        });
      return;
    }
    var thumb = e.target.closest('.kc-act-thumb');
    if (thumb) {
      var wasOn = thumb.classList.contains('is-on');
      var row = thumb.closest('.kc-actions');
      if (row) {
        row.querySelectorAll('.kc-act-thumb').forEach(function (t) {
          t.classList.remove('is-on');
        });
      }
      if (!wasOn) thumb.classList.add('is-on');
    }
  }, false);
})();
