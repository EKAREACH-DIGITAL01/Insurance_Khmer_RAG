window.parent.document.addEventListener(
    'keydown',
    function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') {
            e.stopImmediatePropagation();
        }
    },
    true
);
