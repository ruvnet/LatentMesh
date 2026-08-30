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

  /* ---- count-up numbers ---- */
  var fmt = function (v, dp) { return dp ? v.toFixed(dp) : Math.round(v).toLocaleString(); };
  var countIO = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var el = e.target;
        countIO.unobserve(el);
        var target = parseFloat(el.dataset.count);
        var dp = parseInt(el.dataset.dp || "0", 10);
        var suffix = el.dataset.suffix || "";
        if (reduce || isNaN(target)) { el.textContent = fmt(target, dp) + suffix; return; }
        var t0 = null, dur = 900;
        var tick = function (t) {
          if (t0 === null) t0 = t;
          var p = Math.min((t - t0) / dur, 1);
          var eased = 1 - Math.pow(1 - p, 3);
          el.textContent = fmt(target * eased, dp) + suffix;
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
    },
    { threshold: 0.5 }
  );
  document.querySelectorAll("[data-count]").forEach(function (el) { countIO.observe(el); });

  /* ---- theme toggle (respects system default until clicked) ---- */
  var btn = document.querySelector("[data-theme-toggle]");
  if (btn) {
    var stored = null;
    try { stored = localStorage.getItem("lm-theme"); } catch (e) { /* private mode */ }
    if (stored) document.documentElement.setAttribute("data-theme", stored);
    btn.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme");
      if (!cur) {
        cur = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      }
      var next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("lm-theme", next); } catch (e) { /* ignore */ }
    });
  }
})();
