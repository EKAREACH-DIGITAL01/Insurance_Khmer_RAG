(function() {
    const doc = window.parent.document;
    const win = window.parent;
    const candidates = [
        doc.querySelector('[data-testid="stAppViewContainer"]'),
        doc.querySelector('[data-testid="stMain"]'),
        doc.querySelector('.main'),
        doc.scrollingElement,
        doc.body,
    ];
    candidates.forEach(function(el) {
        if (el) {
            try { el.scrollTo({ top: el.scrollHeight, behavior: "auto" }); }
            catch (e) {}
        }
    });
    try { win.scrollTo({ top: doc.body.scrollHeight, behavior: "auto" }); }
    catch (e) {}
})();
