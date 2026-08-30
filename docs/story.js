/* LatentMesh explainer — scroll choreography.
   One IntersectionObserver drives which step is active; the active step's
   data-viz id decides which visual is shown and animated. No dependencies. */
(function () {
  "use strict";

  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- top progress bar ---- */
  var bar = document.querySelector(".progress");
  if (bar) {
    var ticking = false;
    var onScroll = function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        var h = document.documentElement;
        var max = h.scrollHeight - h.clientHeight;
        bar.style.width = max > 0 ? (h.scrollTop / max) * 100 + "%" : "0%";
        ticking = false;
      });
    };
    addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  /* ---- prepare stroke-draw lengths so dashes animate correctly ---- */
  document.querySelectorAll(".draw").forEach(function (el) {
    try {
      var len = el.getTotalLength ? el.getTotalLength() : 400;
      el.style.setProperty("--len", Math.ceil(len + 2));
    } catch (e) {
      el.style.setProperty("--len", 400);
    }
  });

  /* ---- scrollytelling ---- */
  document.querySelectorAll(".scrolly").forEach(function (scrolly) {
    var steps = Array.prototype.slice.call(scrolly.querySelectorAll(".step"));
    var vizzes = Array.prototype.slice.call(scrolly.querySelectorAll(".viz"));
    if (!steps.length || !vizzes.length) return;

    var show = function (id) {
      vizzes.forEach(function (v) {
        var on = v.dataset.viz === id;
        v.style.display = on ? "grid" : "none";
        // Re-trigger entry animations each time a visual becomes current.
        if (on) {
          if (reduce) { v.classList.add("on"); return; }
          v.classList.remove("on");
          void v.offsetWidth; // force reflow so transitions restart
          requestAnimationFrame(function () { v.classList.add("on"); });
        }
      });
    };

    var current = null;
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          steps.forEach(function (s) { s.classList.toggle("active", s === e.target); });
          var id = e.target.dataset.viz;
          if (id && id !== current) { current = id; show(id); }
          // Drive the 3D scene's state from the step, when it is the active panel.
          var sc = e.target.dataset.scene;
          if (sc && window.__lmScene) { window.__lmScene.setState(sc); }
        });
      },
      // A band across the middle of the viewport: a step activates when it
      // reaches the centre, which is where the reader's eye actually is.
      { rootMargin: "-45% 0px -45% 0px", threshold: 0 }
    );
    steps.forEach(function (s) { io.observe(s); });

    // Show the first visual immediately so the panel is never blank.
    show(steps[0].dataset.viz);
    steps[0].classList.add("active");
    if (steps[0].dataset.scene && window.__lmScene) window.__lmScene.setState(steps[0].dataset.scene);
  });

  /* ---- one-shot reveals for non-scrolly panels ---- */
  var revealIO = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("on"); revealIO.unobserve(e.target); }
      });
    },
    { rootMargin: "0px 0px -12% 0px", threshold: 0.15 }
  );
  document.querySelectorAll(".viz.standalone").forEach(function (v) { revealIO.observe(v); });


})();
