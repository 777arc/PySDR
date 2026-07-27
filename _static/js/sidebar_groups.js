// Groups the hardware SDRs chapters into a collapsible section in the left sidebar
// Expanded by default; the reader can expand/collapse in place without navigating away
// State is remembered across pages via localStorage.
(function () {
  var GROUP_LABEL = 'Specific SDR Hardware';
  var FILES = ['pluto.html', 'usrp.html', 'bladerf.html', 'rtlsdr.html', 'hackrf.html'];

  function fileName(url) {
    return (url || '').split('#')[0].split('?')[0].split('/').pop();
  }

  function init() {
    var sidebar = document.querySelector('.sphinxsidebarwrapper') || document.body;

    // Collect the SDR chapter <li> items, keeping the order in FILES.
    var byFile = {};
    sidebar.querySelectorAll('li.toctree-l1 > a[href]').forEach(function (a) {
      var f = fileName(a.getAttribute('href'));
      if (FILES.indexOf(f) !== -1) byFile[f] = a.parentElement;
    });

    // The chapter you're currently on is rendered as the "current" item with an
    // href of "#" (plus a nested list of its sections), so it needs its own
    // lookup rather than matching by filename.
    var currentFile = fileName(window.location.pathname);
    if (FILES.indexOf(currentFile) !== -1) {
      var currentLi = sidebar.querySelector('li.toctree-l1.current');
      if (currentLi) byFile[currentFile] = currentLi;
    }

    var lis = [];
    FILES.forEach(function (f) { if (byFile[f]) lis.push(byFile[f]); });
    if (!lis.length) return;

    var firstLi = lis[0];
    var parentUl = firstLi.parentElement;

    // Build the collapsible group header and its (initially empty) child list.
    var groupLi = document.createElement('li');
    groupLi.className = 'toctree-l1 sdr-group';

    var toggle = document.createElement('a');
    toggle.href = '#';
    toggle.className = 'sdr-group-toggle';
    toggle.setAttribute('role', 'button');
    toggle.innerHTML = '<span class="sdr-caret"></span>' + GROUP_LABEL;

    var childUl = document.createElement('ul');
    childUl.className = 'sdr-group-children';

    groupLi.appendChild(toggle);
    groupLi.appendChild(childUl);
    parentUl.insertBefore(groupLi, firstLi);

    // Keep them as toctree-l1 so they retain the full-size chapter text;
    // indentation under the group is handled by .sdr-group-children CSS.
    lis.forEach(function (li) {
      childUl.appendChild(li);
    });

    function setOpen(open) {
      groupLi.classList.toggle('open', open);
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    // Expanded by default. Stay open unless the reader explicitly collapsed it
    // last time (stored === '0'); always open when on one of these pages.
    var onSdrPage = FILES.indexOf(currentFile) !== -1;
    var stored = null;
    try { stored = window.localStorage.getItem('sdrGroupOpen'); } catch (e) {}
    setOpen(onSdrPage || stored !== '0');

    toggle.addEventListener('click', function (e) {
      e.preventDefault();
      var open = !groupLi.classList.contains('open');
      setOpen(open);
      try { window.localStorage.setItem('sdrGroupOpen', open ? '1' : '0'); } catch (e2) {}
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
