/* =====================================================================
   EmotionScope v3, Full app.js
   Foundation (routing, health, tabs) + Analyse page (all 8 features)
   ===================================================================== */

(() => {
  "use strict";

  const API_URL = window.location.origin;
  const VALID_ROUTES = [
    "home", "analyse", "cast", "compare", "timeline",
    "scenes", "trailer", "score", "shifts", "persuasion", "history", "about", "dashboard",
  ];

  // -------------------------------------------------------------------
  // ROUTING
  // -------------------------------------------------------------------
  function setRoute(name) {
    if (!VALID_ROUTES.includes(name)) name = "home";
    document.querySelectorAll(".page").forEach((p) => {
      p.hidden = p.dataset.page !== name;
    });
    document.querySelectorAll("[data-route]").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.route === name);
    });
    document.querySelector(".topnav")?.classList.remove("is-open");
    window.scrollTo({ top: 0, behavior: "instant" });
    // The Monitor page is admin-only. guardDashboard() decides whether to
    // show the login card or the real dashboard. Routing through here covers
    // nav clicks, direct #dashboard URLs and browser back/forward equally.
    if (name === "dashboard") guardDashboard();
  }

  function initRouting() {
    document.body.addEventListener("click", (e) => {
      const a = e.target.closest("[data-route]");
      if (!a) return;
      if (a.target === "_blank") return;
      e.preventDefault();
      const name = a.dataset.route;
      history.pushState(null, "", `#${name}`);
      setRoute(name);
    });
    window.addEventListener("popstate", () => {
      const hash = location.hash.replace("#", "") || "home";
      setRoute(hash);
    });
    const initial = location.hash.replace("#", "") || "home";
    setRoute(initial);
  }

  // -------------------------------------------------------------------
  // BURGER (mobile)
  // -------------------------------------------------------------------
  function initBurger() {
    const burger = document.getElementById("burger");
    const topnav = document.querySelector(".topnav");
    if (!burger || !topnav) return;
    burger.addEventListener("click", (e) => {
      e.stopPropagation();
      topnav.classList.toggle("is-open");
    });
    document.addEventListener("click", (e) => {
      if (!topnav.classList.contains("is-open")) return;
      if (e.target.closest(".topnav") || e.target.closest("#burger")) return;
      topnav.classList.remove("is-open");
    });
  }

  // -------------------------------------------------------------------
  // HEALTH
  // -------------------------------------------------------------------
  let healthTimer = null;
  async function checkHealth() {
    const card = document.getElementById("navStatus");
    const txt  = document.getElementById("navStatusText");
    if (!card || !txt) return;
    try {
      const r = await fetch(`${API_URL}/health`);
      if (!r.ok) throw new Error("status " + r.status);
      const d = await r.json();
      if (d.model_loaded) {
        card.dataset.state = "ok";
        txt.textContent = "ready";
        modelVersion = d.model_version || modelVersion;
        if (healthTimer) clearTimeout(healthTimer);
      } else {
        card.dataset.state = "warn";
        txt.textContent = "loading models…";
        healthTimer = setTimeout(checkHealth, 3000);
      }
    } catch {
      card.dataset.state = "error";
      txt.textContent = "offline";
      healthTimer = setTimeout(checkHealth, 5000);
    }
  }

  // -------------------------------------------------------------------
  // HOME TAB GROUP
  // -------------------------------------------------------------------
  const TAB_CTA = {
    film:    "Analyse a TV episode in seconds",
    podcast: "Profile a podcast interview",
    news:    "Score a political clip",
    reality: "Detect drama in reality TV",
  };
  function initTabGroup() {
    const wrap = document.getElementById("tabGroupTabs");
    const cta  = document.getElementById("tabCtaText");
    if (!wrap || !cta) return;
    wrap.addEventListener("click", (e) => {
      const tab = e.target.closest(".tabgroup-tab");
      if (!tab) return;
      wrap.querySelectorAll(".tabgroup-tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      cta.textContent = TAB_CTA[tab.dataset.tab] || TAB_CTA.film;
    });
  }

  // -------------------------------------------------------------------
  // HOME TEMPLATES -> preload Analyse page
  // -------------------------------------------------------------------
  function initTemplates() {
    document.querySelectorAll(".template").forEach((t) => {
      t.addEventListener("click", () => {
        const demo = t.dataset.demo;
        if (demo && DEMO_SAMPLES[demo]) {
          // Switch to Analyse, fill the text, switch tab to text
          history.pushState(null, "", "#analyse");
          setRoute("analyse");
          setTimeout(() => {
            const tabs = document.querySelectorAll("#page-analyse .input-tab");
            const bodies = document.querySelectorAll("#page-analyse .input-body");
            tabs.forEach((x) => x.classList.toggle("is-active", x.dataset.tab === "text"));
            bodies.forEach((b) => b.classList.toggle("is-active", b.dataset.body === "text"));
            const ta = document.getElementById("analyseText");
            if (ta) {
              ta.value = DEMO_SAMPLES[demo];
              ta.dispatchEvent(new Event("input"));
            }
          }, 60);
        }
      });
    });
  }

  // -------------------------------------------------------------------
  // ===================== ANALYSE PAGE ==============================
  // -------------------------------------------------------------------
  const EMOTIONS = ["happiness","neutral","anger","sadness","surprise","fear","disgust"];
  const DEMO_SAMPLES = {
    trump: "Donald Trump shouted at the journalists, furious about the questions. He cursed loudly and slammed his fist on the podium. Trump then stormed off the stage in rage, refusing to answer anything. The reporters were left stunned by the outburst. The press secretary tried to calm the situation but failed.",
    bake: "Mary Berry smiled warmly as she tasted Susan's tart. The crowd cheered when the contestants presented their cakes. Paul was nervous but his bake came out perfectly. Mary laughed at Paul's joke about soggy bottoms. Susan beamed with pride and joy when she won star baker.",
    bigbrother: "John was angry at Sarah for breaking the rules. Sarah cried alone in the bedroom. The other contestants were shocked. Then suddenly everyone laughed when the joke task was announced. Mark felt afraid he might be evicted next.",
    podcast: "The host calmly asked about the new book. The guest spoke slowly about her childhood. They discussed philosophy and ideas. Both were thoughtful and engaged. There were a few quiet moments of reflection.",
    surprise: "Wow, I can't believe this is happening! That's amazing news! What a shock!",
    happiness: "I'm so happy today! Everything is going wonderfully! This is the best day ever!",
    sadness: "I feel so down today. Everything seems hopeless. I miss them so much.",
    anger: "This is absolutely outrageous! How dare you! I am furious right now!",
  };

  let analyseRadar = null;
  let analyseTimeline = null;
  let lastResult = null;
  let modelVersion = "unknown";

  function getActiveAnalyseTab() {
    const t = document.querySelector("#page-analyse .input-tab.is-active");
    return t ? t.dataset.tab : "text";
  }

  function initAnalyseTabs() {
    const tabs = document.querySelectorAll("#page-analyse .input-tab");
    const bodies = document.querySelectorAll("#page-analyse .input-body");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const t = tab.dataset.tab;
        tabs.forEach((x) => x.classList.toggle("is-active", x === tab));
        bodies.forEach((b) => b.classList.toggle("is-active", b.dataset.body === t));
      });
    });
  }

  function initAnalyseCounter() {
    const ta = document.getElementById("analyseText");
    const c  = document.getElementById("analyseCounter");
    if (!ta || !c) return;
    const update = () => {
      const n = ta.value.length;
      c.textContent = `${n} / 1000`;
      c.style.color = n > 950 ? "var(--c-red)" : "";
    };
    ta.addEventListener("input", update);
    update();
  }

  function initQuickButtons() {
    document.querySelectorAll(".quick-btn").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.quick;
        const ta = document.getElementById("analyseText");
        if (ta && DEMO_SAMPLES[k]) {
          ta.value = DEMO_SAMPLES[k];
          ta.dispatchEvent(new Event("input"));
        }
      });
    });
  }

  function initFileInputs() {
    const a = document.getElementById("analyseAudioFile");
    const c = document.getElementById("analyseCsvFile");
    if (a) a.addEventListener("change", () => {
      document.getElementById("analyseAudioName").textContent = a.files[0] ? `✓ ${a.files[0].name}` : "";
    });
    if (c) c.addEventListener("change", () => {
      document.getElementById("analyseCsvName").textContent = c.files[0] ? `✓ ${c.files[0].name}` : "";
    });
  }

  function initRandomSample() {
    document.getElementById("btnRandomSample")?.addEventListener("click", () => {
      const keys = ["trump", "bake", "bigbrother", "podcast"];
      const pick = keys[Math.floor(Math.random() * keys.length)];
      const tabs = document.querySelectorAll("#page-analyse .input-tab");
      const bodies = document.querySelectorAll("#page-analyse .input-body");
      tabs.forEach((t) => t.classList.toggle("is-active", t.dataset.tab === "text"));
      bodies.forEach((b) => b.classList.toggle("is-active", b.dataset.body === "text"));
      const ta = document.getElementById("analyseText");
      ta.value = DEMO_SAMPLES[pick];
      ta.dispatchEvent(new Event("input"));
    });
  }

  function splitSentences(text) {
    return text.replace(/\s+/g," ").split(/(?<=[.!?])\s+(?=[A-Z\d"'])/g).map((s)=>s.trim()).filter(Boolean);
  }
  function hhmmss(s) {
    const h = String(Math.floor(s/3600)).padStart(2,"0");
    const m = String(Math.floor((s%3600)/60)).padStart(2,"0");
    const x = String(Math.floor(s%60)).padStart(2,"0");
    return `${h}:${m}:${x}`;
  }
  function esc(s) {
    return String(s||"")
      .replace(/&/g,"&amp;")
      .replace(/</g,"&lt;")
      .replace(/>/g,"&gt;")
      .replace(/"/g,"&quot;");
  }

  async function runAnalyse() {
    const tab = getActiveAnalyseTab();
    const btn = document.getElementById("btnAnalyse");
    const status = document.getElementById("analyseStatus");
    const results = document.getElementById("analyseResults");

    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Running pipeline…";

    let pipelineResult;
    try {
      if (tab === "text") {
        const text = document.getElementById("analyseText").value.trim();
        if (!text) throw new Error("Empty text.");
        const sentences = splitSentences(text);
        const segments = [];
        for (let i = 0; i < sentences.length; i++) {
          const r = await fetch(`${API_URL}/predict/text/full`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: sentences[i] }),
          });
          if (!r.ok) throw new Error(`text classify ${r.status}`);
          const d = await r.json();
          modelVersion = d.model_version || modelVersion;
          const t0 = i * 4;
          segments.push({
            Start: hhmmss(t0), End: hhmmss(t0+3),
            Sentence: sentences[i],
            Emotion: d.top_emotion,
            Confidence: d.top_confidence,
          });
        }
        const castR = await fetch(`${API_URL}/predict/cast`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, top_n: 10 }),
        });
        const castD = castR.ok ? await castR.json() : { cast: [] };
        const tlR = await fetch(`${API_URL}/predict/timeline`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ segments }),
        });
        const timeline = tlR.ok ? await tlR.json() : {};
        pipelineResult = { segments, cast: castD.cast || [], timeline };

      } else if (tab === "audio" || tab === "csv") {
        const fileEl = document.getElementById(tab === "audio" ? "analyseAudioFile" : "analyseCsvFile");
        const file = fileEl.files && fileEl.files[0];
        if (!file) throw new Error("No file selected.");
        const form = new FormData();
        form.append("file", file);
        form.append("is_transcript", tab === "csv" ? "true" : "false");
        form.append("whisper_model", document.getElementById("whisper-model-audio")?.value || "base");
        const r = await fetch(`${API_URL}/predict/pipeline`, { method: "POST", body: form });
        if (!r.ok) throw new Error(`pipeline ${r.status}`);
        pipelineResult = await r.json();

      } else if (tab === "youtube") {
        const url = document.getElementById("analyseYouTube").value.trim();
        if (!url) throw new Error("No URL.");
        const form = new FormData();
        form.append("url", url);
        form.append("whisper_model", document.getElementById("whisper-model-youtube")?.value || "base");
        const r = await fetch(`${API_URL}/predict/youtube`, { method: "POST", body: form });
        if (!r.ok) throw new Error(`youtube ${r.status}`);
        pipelineResult = await r.json();
      }
    } catch (err) {
      btn.disabled = false;
      status.classList.add("is-error");
      status.textContent = `❌ Failed: ${err.message}`;
      return;
    }

    status.textContent = "⏳ Computing scenes, score, persuasion…";

    const segments = pipelineResult.segments || [];
    let scenes = [], trailer = { clips: [] }, score = null, persuasion = null, shifts = { shifts: [] };
    try {
      const sR = await fetch(`${API_URL}/predict/scenes`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments, min_segments_per_scene: 2, generate_titles: true }),
      });
      if (sR.ok) { const d = await sR.json(); scenes = d.scenes || []; }

      const [trR, scR, peR, shR] = await Promise.all([
        scenes.length ? fetch(`${API_URL}/predict/trailer`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scenes, generate_taglines: true }),
        }).then((r) => r.ok ? r.json() : { clips: [] }) : Promise.resolve({ clips: [] }),
        fetch(`${API_URL}/predict/score`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ segments, cast: pipelineResult.cast || [], scenes, use_ai_classification: true }),
        }).then((r) => r.ok ? r.json() : null),
        fetch(`${API_URL}/predict/persuasion`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ segments, use_ai_assessment: true }),
        }).then((r) => r.ok ? r.json() : null),
        fetch(`${API_URL}/predict/shifts`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ segments }),
        }).then((r) => r.ok ? r.json() : { shifts: [] }),
      ]);
      trailer = trR; score = scR; persuasion = peR; shifts = shR;
    } catch (e) { console.warn("Enrich failed:", e); }
    // AI Summary (final synthesis)
    status.textContent = "⏳ AI generating verdict…";
    let aiSummary = null;
    try {
      const sumR = await fetch(`${API_URL}/predict/summary`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments, cast: pipelineResult.cast || [], scenes, score, persuasion }),
      });
      if (sumR.ok) aiSummary = await sumR.json();
    } catch (e) { console.warn("Summary failed:", e); }

    lastResult = { ...pipelineResult, scenes, trailer, score, persuasion, shifts, aiSummary };
// Save to history
    const txt = getActiveAnalyseTab() === "text"
      ? document.getElementById("analyseText").value
      : "";
    addHistoryEntry({
      type: "analyse",
      title: (lastResult.score && lastResult.score.classification) || "Untitled analysis",
      snippet: txt.slice(0, 120),
      text: txt,
      grade: (lastResult.score && lastResult.score.grade) || "-",
      segmentCount: (lastResult.segments || []).length,
      t_total:     lastResult.t_total     || 0,
      t_whisper:   lastResult.t_whisper   || 0,
      t_translate: lastResult.t_translate || 0,
      t_classify:  lastResult.t_classify  || 0,
    });

    renderAnalyse(lastResult);
    resetQaPanel(); // new analysis means a fresh Q&A conversation
    status.classList.remove("is-error");
    status.classList.add("is-ok");
    status.textContent = "✓ Done";
    btn.disabled = false;
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderAnalyse(r) {
    const segs = r.segments || [];

    // AI Verdict
    const aiCard = document.getElementById("resultAiVerdict");
    const aiText = document.getElementById("resultAiText");
    const aiBadge = document.getElementById("resultAiBadge");
    if (r.aiSummary && r.aiSummary.summary) {
      aiCard.hidden = false;
      aiText.textContent = r.aiSummary.summary;
      // Show the actual model that wrote the verdict (backend sends it).
      aiBadge.textContent = r.aiSummary.ai_generated
        ? `Groq · ${r.aiSummary.model || "LLM"}`
        : "Fallback";
    } else if (aiCard) {
      aiCard.hidden = true;
    }

    document.getElementById("resultTitle").textContent = (r.score && r.score.classification) || "Untitled content";
    document.getElementById("resultMeta").textContent = `${segs.length} segments · ${(r.cast||[]).length} characters · ${(r.scenes||[]).length} scenes`;
    document.getElementById("resultGrade").textContent = (r.score && r.score.grade) || "-";

    // Radar
    const dist = {};
    EMOTIONS.forEach((e) => dist[e] = 0);
    segs.forEach((s) => {
      const e = s.Emotion || s.emotion;
      if (dist.hasOwnProperty(e)) dist[e]++;
    });
    const tot = Object.values(dist).reduce((a,b)=>a+b,0) || 1;
    EMOTIONS.forEach((e) => dist[e] = dist[e] / tot);
    const ctx = document.getElementById("analyseRadarChart");
    if (analyseRadar) analyseRadar.destroy();
    analyseRadar = new Chart(ctx, {
      type: "radar",
      data: {
        labels: EMOTIONS.map((e) => e.charAt(0).toUpperCase()+e.slice(1)),
        datasets: [{
          label: "Distribution",
          data: EMOTIONS.map((e) => dist[e]),
          backgroundColor: "rgba(0,212,255,0.18)",
          borderColor: "#00d4ff",
          borderWidth: 2,
          pointBackgroundColor: "#00d4ff",
          pointRadius: 3,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { r: {
          angleLines: { color: "rgba(255,255,255,0.08)" },
          grid: { color: "rgba(255,255,255,0.08)" },
          pointLabels: { color: "#a1a1a1", font: { family: "Geist Mono, monospace", size: 11 } },
          ticks: { display: false },
          suggestedMin: 0,
          suggestedMax: Math.max(0.5, ...Object.values(dist)) * 1.1,
        } },
      },
    });

    // Score axes
    const axesEl = document.getElementById("resultScoreAxes");
    axesEl.innerHTML = "";
    document.getElementById("resultClass").textContent = (r.score && r.score.classification) || "-";
    document.getElementById("resultAudience").textContent = (r.score && r.score.target_audience) || "";
    if (r.score && r.score.scores) {
      Object.entries(r.score.scores).forEach(([k,v]) => {
        const pct = Math.round((v||0)*100);
        const row = document.createElement("div");
        row.className = "score-axis";
        row.innerHTML = `
          <span class="score-axis-label">${k.replace(/_/g," ")}</span>
          <span class="score-axis-val">${pct}%</span>
          <div class="score-axis-bar"><span style="width:${pct}%;"></span></div>
        `;
        axesEl.appendChild(row);
      });
    }

    // Timeline
    const tl = r.timeline || {};
    const times = tl.times || [];
    const confs = times.map((_, i) => {
      const lbl = (tl.labels || [])[i];
      return lbl ? (tl.series?.[lbl]?.[i] || 0) : 0;
    });
    const tctx = document.getElementById("analyseTimelineChart");
    if (analyseTimeline) analyseTimeline.destroy();
    analyseTimeline = new Chart(tctx, {
      type: "line",
      data: {
        datasets: [{
          label: "Intensity",
          data: times.map((t, i) => ({ x: t, y: confs[i] })),
          borderColor: "#00d4ff",
          backgroundColor: "rgba(0,212,255,0.15)",
          borderWidth: 2, tension: 0.35, fill: true,
          pointBackgroundColor: "#00d4ff", pointRadius: 3,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { intersect: false, mode: "index" },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `${Math.round(items[0].parsed.x)}s`,
              label: (item) => {
                const s = (tl.sentences || [])[item.dataIndex] || "";
                return s.length > 80 ? s.slice(0,80)+"…" : s;
              },
            },
          },
        },
        scales: {
          x: { type: "linear", grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#666", callback: (v) => `${v}s` } },
          y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#666" } },
        },
      },
    });

    // Cast
    const castEl = document.getElementById("resultCast");
    const cast = (r.cast || []).slice(0, 6);
    castEl.innerHTML = cast.length ? cast.map((c) => `
      <div class="cast-row">
        <div class="cast-avatar">${esc((c.name||"?").charAt(0).toUpperCase())}</div>
        <div class="cast-info">
          <p class="cast-name">${esc(c.name)}</p>
          <p class="cast-detail">${c.mentions} mentions · ${esc(c.dominant_emotion)}</p>
        </div>
        <span class="cast-arch" data-arch="${esc(c.archetype)}">${esc(c.archetype)}</span>
      </div>
    `).join("") : `<div class="empty-state">No named characters detected.</div>`;

    // Persuasion
    if (r.persuasion) {
      const p = r.persuasion;
      const pct = Math.round((p.persuasion_score || 0) * 100);
      document.getElementById("resultPersLevel").textContent = p.confidence_level || "-";
      document.getElementById("resultPersNum").textContent = `${pct}%`;
      document.getElementById("resultPersFill").style.width = `${pct}%`;
      document.getElementById("resultPersText").textContent = p.assessment || "";
    }

    // Scenes
    const sceneEl = document.getElementById("resultScenes");
    const scenes = r.scenes || [];
    sceneEl.innerHTML = scenes.length ? scenes.map((s) => `
      <div class="scene-row">
        <span class="scene-mood" data-mood="${esc(s.mood)}">${esc(s.mood)}</span>
        <span class="scene-title">${esc(s.ai_title || s.dominant_emotion || "Scene")}</span>
        <span class="scene-time">${esc(s.start_label || "0:00")} -> ${esc(s.end_label || "")}</span>
      </div>
    `).join("") : `<div class="empty-state">Not enough segments to detect scenes.</div>`;

    // Clips
    const clipEl = document.getElementById("resultClips");
    const clips = (r.trailer && r.trailer.clips) || [];
    clipEl.innerHTML = clips.length ? clips.map((c, i) => {
      const color  = MOOD_COLORS[c.mood] || "#9ca3af";
      const color2 = MOOD_COLORS_2[c.mood] || "#7928ca";
      const icon   = MOOD_ICONS_SVG[c.mood] || MOOD_ICONS_SVG["Mixed"];
      return `
        <div class="trailer-card" style="--tc-color: ${color}; --tc-color-2: ${color2};">
          <div class="trailer-card-preview" style="--mood-icon: url(&quot;${icon}&quot;);">
            <div class="trailer-card-play"></div>
            <span class="trailer-card-time">${esc(c.start_label || "0:00")} · ${Math.round(c.duration||0)}s</span>
          </div>
          <div class="trailer-card-meta">
            <span class="trailer-card-num">Clip ${i+1}</span>
            <span class="trailer-card-mood" style="background: ${color}26; color: ${color};">${esc(c.mood)}</span>
          </div>
          <h4 class="trailer-card-tagline">${esc(c.tagline || c.mood || "Highlight")}</h4>
          <p class="trailer-card-hook">"${esc(c.hook || "")}"</p>
        </div>
      `;
    }).join("") : `<div class="empty-state">No trailer clips yet.</div>`;;

    // Shifts
    const shiftEl = document.getElementById("resultShifts");
    const shifts = (r.shifts && r.shifts.shifts) || [];
    shiftEl.innerHTML = shifts.length ? shifts.slice(0, 6).map((s) => `
      <div class="shift-row">
        <span class="shift-time-tag">${esc(s.label || "0:00")}</span>
        <div>
          <div class="shift-arrow">
            <span class="shift-from">${esc(s.from_emotion)}</span>
            <span>-></span>
            <span class="shift-to">${esc(s.to_emotion)}</span>
            <span class="shift-mag">${Math.round((s.magnitude||0)*100)}%</span>
          </div>
          <p class="shift-sentence">"${esc(s.to_sentence || "")}"</p>
        </div>
      </div>
    `).join("") : `<div class="empty-state">No major turning points detected.</div>`;

    // Segments with feedback
    const segEl = document.getElementById("resultSegments");
    segEl.innerHTML = segs.map((s, i) => `
      <div class="segment-row" data-idx="${i}">
        <span class="segment-emotion" data-emotion="${esc(s.Emotion||s.emotion)}">${esc(s.Emotion||s.emotion)}</span>
        <span class="segment-text">${esc(s.Sentence||s.sentence)}</span>
        <span class="segment-conf">${(((s.Confidence||s.confidence)||0)*100).toFixed(0)}%</span>
        <div class="segment-feedback">
          <button class="fb-btn fb-yes" title="Correct">👍</button>
          <button class="fb-btn fb-no" title="Wrong">👎</button>
        </div>
        <select class="fb-correct-input">
          <option value="">- Correct label -</option>
          ${EMOTIONS.map((e) => `<option value="${e}">${e}</option>`).join("")}
        </select>
      </div>
    `).join("");

    segEl.querySelectorAll(".segment-row").forEach((row) => {
      const idx = parseInt(row.dataset.idx, 10);
      const seg = segs[idx];
      const yes = row.querySelector(".fb-yes");
      const no  = row.querySelector(".fb-no");
      const sel = row.querySelector(".fb-correct-input");

      yes.addEventListener("click", () => {
        if (yes.classList.contains("is-correct")) return;
        sendFeedback(seg, true, null);
        yes.classList.add("is-correct");
        no.classList.remove("is-wrong");
        row.classList.remove("is-correcting");
      });
      no.addEventListener("click", () => {
        no.classList.add("is-wrong");
        yes.classList.remove("is-correct");
        row.classList.add("is-correcting");
      });
      sel.addEventListener("change", () => {
        if (!sel.value) return;
        sendFeedback(seg, false, sel.value);
        row.classList.remove("is-correcting");
      });
    });
  }

  async function sendFeedback(seg, isCorrect, correctLabel) {
    try {
      await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sentence: seg.Sentence || seg.sentence,
          predicted_label: seg.Emotion || seg.emotion,
          predicted_conf: seg.Confidence || seg.confidence || 0,
          is_correct: isCorrect,
          user_label: correctLabel,
          model_version: modelVersion,
          source: "text",
        }),
      });
    } catch (e) { console.warn("Feedback failed:", e); }
  }

  // -------------------------------------------------------------------
  // ===================== CAST PAGE =================================
  // -------------------------------------------------------------------
  const CAST_DEMOS = {
    hamlet: "Hamlet stared at the ghost in terror. Claudius poured the poison while smiling. Ophelia sang sadly by the river. Polonius hid behind the curtain, nervous. Hamlet's anger grew as he watched his uncle Claudius. The queen Gertrude pleaded with her son. Laertes returned furious to avenge his father. Horatio comforted Hamlet in his grief.",
    press: "Donald Trump shouted at the reporters in fury. Joe Biden calmly defended his policy. Kamala Harris smiled while answering questions. Trump cursed and stormed off the stage. The press secretary tried to calm everyone. Biden then addressed the nation with hope.",
    bake: "Mary Berry tasted Susan's lemon tart with delight. Paul Hollywood frowned at Tom's soggy bottom. Prue laughed at the jokes from the contestants. Susan beamed proudly as she presented her cake. Tom looked nervous before serving his dish. Mary praised Susan's perfect bake.",
  };
  const ARCH_COLORS = {
    "Hero":         "#10b981",
    "Antagonist":   "#ef4444",
    "Tragic Hero":  "#3b82f6",
    "Comic Relief": "#f5c518",
    "Mysterious":   "#7928ca",
    "Supporting":   "#9ca3af",
  };

  function initCastPage() {
    const ta = document.getElementById("castText");
    const c  = document.getElementById("castCounter");
    if (ta && c) {
      const update = () => {
        const n = ta.value.length;
        c.textContent = `${n} / 3000`;
        c.style.color = n > 2900 ? "var(--c-red)" : "";
      };
      ta.addEventListener("input", update);
      update();
    }
    document.querySelectorAll("[data-cast-quick]").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.castQuick;
        if (ta && CAST_DEMOS[k]) {
          ta.value = CAST_DEMOS[k];
          ta.dispatchEvent(new Event("input"));
        }
      });
    });
    document.getElementById("btnCast")?.addEventListener("click", runCast);
  }

  async function runCast() {
    const btn = document.getElementById("btnCast");
    const status = document.getElementById("castStatus");
    const results = document.getElementById("castResults");
    const ta = document.getElementById("castText");
    const text = (ta?.value || "").trim();
    if (!text) {
      status.classList.add("is-error");
      status.textContent = "❌ Empty text.";
      return;
    }
    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Detecting cast…";

    try {
      const r = await fetch(`${API_URL}/predict/cast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, top_n: 10 }),
      });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const d = await r.json();
      // Save to history
    addHistoryEntry({
      type: "cast",
      title: `${d.total_characters} characters detected`,
      snippet: text.slice(0, 120),
      text: text,
      characterCount: d.total_characters,
    });
      renderCast(d);
      status.classList.add("is-ok");
      status.textContent = `✓ Found ${d.total_characters} characters`;
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderCast(d) {
    const cast = d.cast || [];
    const summary = document.getElementById("castSummary");
    summary.textContent = cast.length
      ? `${cast.length} characters detected · sorted by mentions`
      : "No named characters found";

    const grid = document.getElementById("castGrid");
    if (!cast.length) {
      grid.innerHTML = `<div class="empty-state">Try a passage with more names (Hamlet, Trump, Mary Berry…).</div>`;
      return;
    }
    grid.innerHTML = cast.map((c) => {
      const color = ARCH_COLORS[c.archetype] || "#9ca3af";
      const initial = (c.name || "?").charAt(0).toUpperCase();
      const topLine = (c.top_lines && c.top_lines[0]) || null;
      return `
        <div class="cast-card" style="--accent: ${color};">
          <div class="cast-card-head">
            <div class="cast-card-avatar" style="background: ${color};">${esc(initial)}</div>
            <div class="cast-card-info">
              <p class="cast-card-name">${esc(c.name)}</p>
              <p class="cast-card-meta">${c.mentions} mentions · ${esc(c.dominant_emotion)} (${Math.round((c.score||0)*100)}%)</p>
            </div>
            <span class="cast-card-arch" style="background: ${color}26; color: ${color};">${esc(c.archetype)}</span>
          </div>
          ${topLine ? `<p class="cast-card-quote">"${esc(topLine.sentence)}"</p>` : ""}
          <div class="cast-card-stats">
            ${Object.entries(c.emotions || {}).slice(0,3).map(([k,v]) => `
              <span class="cast-card-stat">${k}: ${Math.round((v||0)*100)}%</span>
            `).join("")}
          </div>
        </div>
      `;
    }).join("");
  }

  // -------------------------------------------------------------------
  // ===================== COMPARE PAGE ==============================
  // -------------------------------------------------------------------
  const COMPARE_DEMO_LEFT  = "Donald Trump shouted at the journalists, furious about the questions. He cursed loudly and slammed his fist on the podium. Trump then stormed off the stage in rage. The reporters were left stunned by the outburst. The press secretary tried to calm the situation but failed.";
  const COMPARE_DEMO_RIGHT = "Mary Berry smiled warmly as she tasted Susan's tart. The crowd cheered when the contestants presented their cakes. Paul laughed at the joke about soggy bottoms. Susan beamed with pride and joy when she won star baker. The atmosphere in the tent was full of warmth and friendship.";

  const WINNER_LABELS = {
    more_happy:      "more happy",
    more_sad:        "more sad",
    more_angry:      "more angry",
    more_fearful:    "more fearful",
    more_surprised:  "more surprised",
    more_disgusted:  "more disgusted",
    more_intense:    "more intense",
    more_diverse:    "more diverse",
    more_characters: "more characters",
    longer:          "longer",
  };

  let cmpRadar = null;

  function initComparePage() {
    document.getElementById("btnCompare")?.addEventListener("click", runCompare);
    document.getElementById("btnCompareDemo")?.addEventListener("click", () => {
      const left  = document.querySelector('.twin-panel[data-side="left"]');
      const right = document.querySelector('.twin-panel[data-side="right"]');
      left.querySelector('[data-field="text"]').value  = COMPARE_DEMO_LEFT;
      right.querySelector('[data-field="text"]').value = COMPARE_DEMO_RIGHT;
      left.querySelector('[data-field="label"]').value  = "";
      right.querySelector('[data-field="label"]').value = "";
    });
  }

  async function analyseOneSide(panel) {
    const text = panel.querySelector('[data-field="text"]').value.trim();
    const label = panel.querySelector('[data-field="label"]').value.trim();
    if (!text) throw new Error("Empty side.");
    const sentences = splitSentences(text);
    const segments = [];
    for (let i = 0; i < sentences.length; i++) {
      const r = await fetch(`${API_URL}/predict/text/full`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sentences[i] }),
      });
      if (!r.ok) throw new Error(`classify ${r.status}`);
      const d = await r.json();
      const t0 = i * 4;
      segments.push({
        Start: hhmmss(t0), End: hhmmss(t0+3),
        Sentence: sentences[i],
        Emotion: d.top_emotion,
        Confidence: d.top_confidence,
      });
    }
    const castR = await fetch(`${API_URL}/predict/cast`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_n: 10 }),
    });
    const castD = castR.ok ? await castR.json() : { cast: [] };
    return { segments, cast: castD.cast || [], label };
  }

  async function runCompare() {
    const btn = document.getElementById("btnCompare");
    const status = document.getElementById("compareStatus");
    const results = document.getElementById("compareResults");
    const leftPanel  = document.querySelector('.twin-panel[data-side="left"]');
    const rightPanel = document.querySelector('.twin-panel[data-side="right"]');

    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Analysing both sides in parallel…";

    let leftR, rightR;
    try {
      [leftR, rightR] = await Promise.all([
        analyseOneSide(leftPanel),
        analyseOneSide(rightPanel),
      ]);
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
      btn.disabled = false;
      return;
    }

    status.textContent = "⏳ Computing comparison + AI titles…";
    let cmp;
    try {
      const r = await fetch(`${API_URL}/compare`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left:  { type: "text", label: leftR.label || null,  result: leftR },
          right: { type: "text", label: rightR.label || null, result: rightR },
        }),
      });
      if (!r.ok) throw new Error(`compare ${r.status}`);
      cmp = await r.json();
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
      btn.disabled = false;
      return;
    }
// Save to history
    const leftTxt  = leftPanel.querySelector('[data-field="text"]').value;
    const rightTxt = rightPanel.querySelector('[data-field="text"]').value;
    addHistoryEntry({
      type: "compare",
      title: `${cmp.left_label || "Side A"} vs ${cmp.right_label || "Side B"}`,
      snippet: leftTxt.slice(0, 80) + " · " + rightTxt.slice(0, 60),
      leftText: leftTxt,
      rightText: rightTxt,
      leftLabel:  leftPanel.querySelector('[data-field="label"]').value,
      rightLabel: rightPanel.querySelector('[data-field="label"]').value,
      similarity: Math.round((cmp.comparison.similarity || 0) * 100),
    });
    renderCompare(cmp);
    status.classList.add("is-ok");
    status.textContent = "✓ Done";
    btn.disabled = false;
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderCompare(cmp) {
    // Dynamic colors based on dominant emotion per side
    const leftDist  = cmp.comparison.emotion_distribution.left;
    const rightDist = cmp.comparison.emotion_distribution.right;
    const leftDom  = Object.entries(leftDist).sort((a,b) => b[1]-a[1])[0][0];
    const rightDom = Object.entries(rightDist).sort((a,b) => b[1]-a[1])[0][0];

    const leftColor  = EMOTION_COLORS_RGB[leftDom]  || "#0070f3";
    let   rightColor = EMOTION_COLORS_RGB[rightDom] || "#ff0080";
    if (leftColor === rightColor) {
      rightColor = leftColor === "#3b82f6" ? "#ff0080" : "#3b82f6";
    }

    const results = document.getElementById("compareResults");
    results.style.setProperty("--cmp-left-color",  leftColor);
    results.style.setProperty("--cmp-right-color", rightColor);

    document.getElementById("cmpLeftName").textContent  = cmp.left_label  || "Source A";
    document.getElementById("cmpRightName").textContent = cmp.right_label || "Source B";

    const st = cmp.comparison.stats;
    document.getElementById("cmpLeftMeta").textContent  =
      `${st.left_segment_count} segments · ${st.left_character_count} characters · ${leftDom}`;
    document.getElementById("cmpRightMeta").textContent =
      `${st.right_segment_count} segments · ${st.right_character_count} characters · ${rightDom}`;
    document.getElementById("cmpSimilarity").textContent = `${Math.round((cmp.comparison.similarity||0)*100)}%`;

    // Winners
    const wEl = document.getElementById("cmpWinners");
    wEl.innerHTML = Object.entries(cmp.comparison.winners).map(([k, side]) => {
      const label = WINNER_LABELS[k] || k.replace(/_/g, " ");
      const display = side === "left" ? "CH 01" : side === "right" ? "CH 02" : "TIE";
      return `
        <div class="winner-pill">
          <span class="winner-pill-label">${label}</span>
          <span class="winner-pill-side" data-side="${side}">${display}</span>
        </div>
      `;
    }).join("");

    // Radar overlay with dynamic colors
    const ctx = document.getElementById("cmpRadarChart");
    if (cmpRadar) cmpRadar.destroy();
    const leftAlpha  = leftColor  + "33";
    const rightAlpha = rightColor + "33";
    cmpRadar = new Chart(ctx, {
      type: "radar",
      data: {
        labels: EMOTIONS.map((e) => e.charAt(0).toUpperCase()+e.slice(1)),
        datasets: [
          {
            label: cmp.left_label || "Side A",
            data: EMOTIONS.map((e) => leftDist[e] || 0),
            backgroundColor: leftAlpha,
            borderColor: leftColor,
            borderWidth: 2,
            pointBackgroundColor: leftColor,
            pointRadius: 3,
          },
          {
            label: cmp.right_label || "Side B",
            data: EMOTIONS.map((e) => rightDist[e] || 0),
            backgroundColor: rightAlpha,
            borderColor: rightColor,
            borderWidth: 2,
            pointBackgroundColor: rightColor,
            pointRadius: 3,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom", labels: { color: "#a1a1a1", font: { family: "Geist Mono, monospace", size: 11 } } },
        },
        scales: { r: {
          angleLines: { color: "rgba(255,255,255,0.08)" },
          grid: { color: "rgba(255,255,255,0.08)" },
          pointLabels: { color: "#a1a1a1", font: { family: "Geist Mono, monospace", size: 11 } },
          ticks: { display: false },
        } },
      },
    });

    // Delta bars with dynamic colors
    const dEl = document.getElementById("cmpDeltaBars");
    dEl.innerHTML = "";
    EMOTIONS.forEach((emo) => {
      const v = cmp.comparison.emotion_deltas[emo] || 0;
      const pct = Math.min(Math.abs(v) * 100, 100);
      const side = v >= 0 ? "left" : "right";
      const fillColor = side === "left" ? leftColor : rightColor;
      const style = side === "left"
        ? `right: 50%; width: ${pct/2}%; background: linear-gradient(90deg, ${fillColor}, transparent);`
        : `left: 50%; width: ${pct/2}%; background: linear-gradient(270deg, ${fillColor}, transparent);`;
      const row = document.createElement("div");
      row.className = "delta-row";
      row.innerHTML = `
        <span class="delta-label">${emo}</span>
        <div class="delta-bar"><div class="delta-bar-fill" style="${style}"></div></div>
        <span class="delta-value">${v >= 0 ? "+" : ""}${(v*100).toFixed(0)}%</span>
      `;
      dEl.appendChild(row);
    });
  }
// -------------------------------------------------------------------
  // ===================== TIMELINE PAGE =============================
  // -------------------------------------------------------------------
  const TIMELINE_DEMOS = {
    trump: "Donald Trump shouted at the journalists, furious about the questions. He cursed loudly and slammed his fist on the podium. Trump then stormed off the stage in rage. The reporters were left stunned by the outburst. The press secretary tried to calm everyone. Eventually the room fell silent and tense.",
    bake: "Mary Berry smiled warmly as she tasted Susan's tart. The crowd cheered when the contestants presented their cakes. Paul laughed at the joke about soggy bottoms. Susan beamed with pride when she won star baker. Tom was disappointed but cheered for Susan. Mary praised everyone for their delicious bakes.",
    mixed: "John was happy at the surprise party. Then he was shocked when his ex walked in. He felt sad as memories rushed back. Anger flashed across his face briefly. He sighed deeply and finally laughed it off. Everyone joined him in laughter and joy.",
  };

  let timelineChartObj = null;

  function initTimelinePage() {
    const ta = document.getElementById("timelineText");
    const c  = document.getElementById("timelineCounter");
    if (ta && c) {
      const update = () => {
        c.textContent = `${ta.value.length} / 3000`;
        c.style.color = ta.value.length > 2900 ? "var(--c-red)" : "";
      };
      ta.addEventListener("input", update);
      update();
    }
    document.querySelectorAll("[data-tl-quick]").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.tlQuick;
        if (ta && TIMELINE_DEMOS[k]) {
          ta.value = TIMELINE_DEMOS[k];
          ta.dispatchEvent(new Event("input"));
        }
      });
    });
    document.getElementById("btnTimeline")?.addEventListener("click", runTimelinePage);
  }

  async function runTimelinePage() {
    const btn = document.getElementById("btnTimeline");
    const status = document.getElementById("timelineStatus");
    const results = document.getElementById("timelineResults");
    const text = (document.getElementById("timelineText").value || "").trim();
    if (!text) { status.classList.add("is-error"); status.textContent = "❌ Empty text."; return; }

    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Building timeline…";

    try {
      const sentences = splitSentences(text);
      const segments = [];
      for (let i = 0; i < sentences.length; i++) {
        const r = await fetch(`${API_URL}/predict/text/full`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sentences[i] }),
        });
        if (!r.ok) throw new Error(`classify ${r.status}`);
        const d = await r.json();
        const t0 = i * 4;
        segments.push({
          Start: hhmmss(t0), End: hhmmss(t0+3),
          Sentence: sentences[i],
          Emotion: d.top_emotion,
          Confidence: d.top_confidence,
        });
      }
      const tlR = await fetch(`${API_URL}/predict/timeline`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments }),
      });
      if (!tlR.ok) throw new Error(`timeline ${tlR.status}`);
      const tl = await tlR.json();
      renderTimelinePage(tl);
      status.classList.add("is-ok");
      status.textContent = `✓ ${tl.segment_count} segments · ${(tl.peaks||[]).length} peaks`;
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderTimelinePage(tl) {
    document.getElementById("timelineSummary").textContent =
      `${tl.segment_count} segments · ${Math.round(tl.duration || 0)}s total · ${(tl.peaks||[]).length} peaks`;

    // Chart
    const times = tl.times || [];
    const confs = times.map((_, i) => {
      const lbl = (tl.labels || [])[i];
      return lbl ? (tl.series?.[lbl]?.[i] || 0) : 0;
    });
    const ctx = document.getElementById("timelineChart");
    if (timelineChartObj) timelineChartObj.destroy();
    timelineChartObj = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [{
          label: "Intensity",
          data: times.map((t, i) => ({ x: t, y: confs[i] })),
          borderColor: "#00d4ff",
          backgroundColor: "rgba(0,212,255,0.15)",
          borderWidth: 2, tension: 0.35, fill: true,
          pointBackgroundColor: "#00d4ff",
          pointRadius: 4,
          pointHoverRadius: 6,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { intersect: false, mode: "index" },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `${Math.round(items[0].parsed.x)}s · ${(tl.labels||[])[items[0].dataIndex] || ""}`,
              label: (item) => {
                const s = (tl.sentences || [])[item.dataIndex] || "";
                return s.length > 90 ? s.slice(0, 90) + "…" : s;
              },
            },
          },
        },
        scales: {
          x: { type: "linear", grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#666", callback: (v) => `${v}s` } },
          y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#666" } },
        },
      },
    });

    // Peaks
    const peakEl = document.getElementById("peakList");
    const peaks = tl.peaks || [];
    peakEl.innerHTML = peaks.length ? peaks.slice(0, 5).map((p, i) => `
      <div class="peak-row">
        <span class="peak-rank">${i+1}</span>
        <span class="peak-time">${Math.round(p.time)}s · ${esc(p.emotion)}</span>
        <span class="peak-sentence">${esc(p.sentence)}</span>
        <span class="peak-score">${Math.round((p.score||0)*100)}%</span>
      </div>
    `).join("") : `<div class="empty-state">No strong peaks detected (try a more emotional passage).</div>`;
  }

  // -------------------------------------------------------------------
  // ===================== SCENES PAGE ===============================
  // -------------------------------------------------------------------
  const SCENE_DEMOS = {
    drama: "John walked happily into the room. He smiled at his friends. Then suddenly he saw the betrayal note. His face turned red with anger. He shouted at everyone in fury. Sarah cried in the corner alone. Mark tried to calm them down. Eventually they all sat in silence. The room felt heavy with sadness.",
    news: "The president walked confidently to the podium. The crowd cheered with joy. He began his speech with strong words. Then the protesters started shouting angrily. Police moved in quickly. The president looked nervous and afraid. He cut his speech short. The crowd watched in shocked silence.",
  };
// Primary color per mood
  const MOOD_COLORS = {
    "Tension":      "#ef4444",
    "Climax":       "#fb923c",
    "Comic Relief": "#f5c518",
    "Joyful":       "#f5c518",
    "Sadness":      "#3b82f6",
    "Mystery":      "#7928ca",
    "Disgust":      "#10b981",
    "Calm":         "#9ca3af",
    "Mixed":        "#a1a1a1",
  };

  // Secondary color per mood (creates the 2-tone halo)
  const MOOD_COLORS_2 = {
    "Tension":      "#7928ca",  // red + violet
    "Climax":       "#f5c518",  // orange + yellow
    "Comic Relief": "#ff0080",  // yellow + pink
    "Joyful":       "#ff0080",  // yellow + pink
    "Sadness":      "#7928ca",  // blue + violet
    "Mystery":      "#3b82f6",  // violet + blue
    "Disgust":      "#fb923c",  // green + orange
    "Calm":         "#3b82f6",  // grey + blue
    "Mixed":        "#10b981",  // mixed
  };

  const MOOD_ICONS_SVG = {
    "Tension":      "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ef4444'><path d='M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z'/></svg>",
    "Climax":       "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23fb923c'><path d='M13 2L3 14h7v8l10-12h-7z'/></svg>",
    "Comic Relief": "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23f5c518'><path d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-3 8c.83 0 1.5-.67 1.5-1.5S9.83 7 9 7s-1.5.67-1.5 1.5S8.17 10 9 10zm6 0c.83 0 1.5-.67 1.5-1.5S15.83 7 15 7s-1.5.67-1.5 1.5.67 1.5 1.5 1.5zm-3 8c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z'/></svg>",
    "Joyful":       "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23f5c518'><path d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-3 8c.83 0 1.5-.67 1.5-1.5S9.83 7 9 7s-1.5.67-1.5 1.5S8.17 10 9 10zm6 0c.83 0 1.5-.67 1.5-1.5S15.83 7 15 7s-1.5.67-1.5 1.5.67 1.5 1.5 1.5zm-3 8c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z'/></svg>",
    "Sadness":      "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%233b82f6'><path d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-3.5 6c.83 0 1.5.67 1.5 1.5S9.33 11 8.5 11 7 10.33 7 9.5 7.67 8 8.5 8zm7 0c.83 0 1.5.67 1.5 1.5S16.33 11 15.5 11 14 10.33 14 9.5s.67-1.5 1.5-1.5zM12 14c2.33 0 4.31 1.46 5.11 3.5H6.89C7.69 15.46 9.67 14 12 14z'/></svg>",
    "Mystery":      "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%237928ca'><path d='M11.07 12.85c.77-1.39 2.25-2.21 3.11-3.44.91-1.29.4-3.7-2.18-3.7-1.69 0-2.52 1.28-2.87 2.34L6.54 6.96C7.25 4.83 9.18 3 11.99 3c2.35 0 3.96 1.07 4.78 2.41.7 1.15 1.11 3.3.03 4.9-1.2 1.77-2.35 2.31-2.97 3.45-.25.46-.35.76-.35 2.24h-2.89c-.01-.78-.13-2.05.48-3.15zM14 20c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2z'/></svg>",
    "Disgust":      "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2310b981'><path d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-3.5 6c.83 0 1.5.67 1.5 1.5S9.33 11 8.5 11 7 10.33 7 9.5 7.67 8 8.5 8zm7 0c.83 0 1.5.67 1.5 1.5S16.33 11 15.5 11 14 10.33 14 9.5s.67-1.5 1.5-1.5zM12 17.5c-1.9 0-3.5-1.04-4-2.5h8c-.5 1.46-2.1 2.5-4 2.5z'/></svg>",
    "Calm":         "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%239ca3af'><path d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z'/></svg>",
    "Mixed":        "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23a1a1a1'><path d='M12 2L1 21h22L12 2zm0 4l8.5 13H3.5L12 6z'/></svg>",
  };

  function initScenesPage() {
    const ta = document.getElementById("scenesText");
    const c  = document.getElementById("scenesCounter");
    if (ta && c) {
      const update = () => {
        c.textContent = `${ta.value.length} / 3000`;
        c.style.color = ta.value.length > 2900 ? "var(--c-red)" : "";
      };
      ta.addEventListener("input", update);
      update();
    }
    document.querySelectorAll("[data-sc-quick]").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.scQuick;
        if (ta && SCENE_DEMOS[k]) {
          ta.value = SCENE_DEMOS[k];
          ta.dispatchEvent(new Event("input"));
        }
      });
    });
    document.getElementById("btnScenes")?.addEventListener("click", runScenesPage);
  }

  async function runScenesPage() {
    const btn = document.getElementById("btnScenes");
    const status = document.getElementById("scenesStatus");
    const results = document.getElementById("scenesResults");
    const text = (document.getElementById("scenesText").value || "").trim();
    if (!text) { status.classList.add("is-error"); status.textContent = "❌ Empty text."; return; }

    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Classifying segments…";

    try {
      const sentences = splitSentences(text);
      const segments = [];
      for (let i = 0; i < sentences.length; i++) {
        const r = await fetch(`${API_URL}/predict/text/full`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sentences[i] }),
        });
        if (!r.ok) throw new Error(`classify ${r.status}`);
        const d = await r.json();
        const t0 = i * 4;
        segments.push({
          Start: hhmmss(t0), End: hhmmss(t0+3),
          Sentence: sentences[i],
          Emotion: d.top_emotion,
          Confidence: d.top_confidence,
        });
      }
      status.textContent = "⏳ Detecting scenes with AI…";
      const sR = await fetch(`${API_URL}/predict/scenes`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments, min_segments_per_scene: 2, generate_titles: true }),
      });
      if (!sR.ok) throw new Error(`scenes ${sR.status}`);
      const d = await sR.json();
      renderScenesPage(d);
      status.classList.add("is-ok");
      status.textContent = `✓ ${(d.scenes||[]).length} scenes detected`;
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderScenesPage(d) {
    const scenes = d.scenes || [];
    document.getElementById("scenesSummary").textContent =
      scenes.length ? `${scenes.length} scenes detected` : "No scenes detected";

    // Strip (visual flow)
    const strip = document.getElementById("sceneStrip");
    strip.innerHTML = scenes.map((s, i) => {
      const color = MOOD_COLORS[s.mood] || "#9ca3af";
      const dur = Math.round(s.duration || 0);
      return `
        <div class="scene-segment" style="--seg-color: ${color};">
          <div class="scene-segment-num">SCENE ${i+1}</div>
          <div class="scene-segment-mood" style="color: ${color};">${esc(s.mood)}</div>
          <div class="scene-segment-duration">${dur}s</div>
        </div>
      `;
    }).join("");

    // Cards
    const cards = document.getElementById("sceneCards");
    cards.innerHTML = scenes.map((s, i) => {
      const color = MOOD_COLORS[s.mood] || "#9ca3af";
      const firstSentence = (s.segments && s.segments[0] && (s.segments[0].Sentence || s.segments[0].sentence)) || "";
      return `
        <div class="scene-card" style="--sc-color: ${color};">
          <div class="scene-card-head">
            <span class="scene-card-num">Scene ${i+1}</span>
            <span class="scene-card-mood" style="background: ${color}26; color: ${color};">${esc(s.mood)}</span>
            <span class="scene-card-time">${esc(s.start_label || "0:00")} -> ${esc(s.end_label || "")}</span>
          </div>
          <h4 class="scene-card-title">${esc(s.ai_title || s.dominant_emotion || "Untitled scene")}</h4>
          <p class="scene-card-quote">"${esc(firstSentence)}"</p>
        </div>
      `;
    }).join("");
  }

  // -------------------------------------------------------------------
  // ===================== TRAILER PAGE ==============================
  // -------------------------------------------------------------------
  const TRAILER_DEMOS = {
    drama: "John walked happily into the room with great joy. He smiled brightly at his friends. Then suddenly he saw the shocking betrayal note. His face turned red with rage and fury. He shouted angrily at everyone present. Sarah cried sadly in the corner alone. Mark tried to calm them with care. Eventually they all sat in heavy silence. The atmosphere felt deeply sad and broken.",
    reality: "Susan presented her perfect cake with joy. Mary Berry tasted it and beamed with delight. The crowd cheered loudly with happiness. Tom was nervous about his own bake. Paul frowned at Tom's soggy bottom. Tom looked devastated and sad. The judges then surprised everyone with results. Susan won star baker to huge applause.",
  };

  function initTrailerPage() {
    const ta = document.getElementById("trailerText");
    const c  = document.getElementById("trailerCounter");
    if (ta && c) {
      const update = () => {
        c.textContent = `${ta.value.length} / 3000`;
        c.style.color = ta.value.length > 2900 ? "var(--c-red)" : "";
      };
      ta.addEventListener("input", update);
      update();
    }
    document.querySelectorAll("[data-tr-quick]").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.trQuick;
        if (ta && TRAILER_DEMOS[k]) {
          ta.value = TRAILER_DEMOS[k];
          ta.dispatchEvent(new Event("input"));
        }
      });
    });
    document.getElementById("btnTrailer")?.addEventListener("click", runTrailerPage);
  }

  async function runTrailerPage() {
    const btn = document.getElementById("btnTrailer");
    const status = document.getElementById("trailerStatus");
    const results = document.getElementById("trailerResults");
    const text = (document.getElementById("trailerText").value || "").trim();
    if (!text) { status.classList.add("is-error"); status.textContent = "❌ Empty text."; return; }

    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Classifying segments…";

    try {
      const sentences = splitSentences(text);
      const segments = [];
      for (let i = 0; i < sentences.length; i++) {
        const r = await fetch(`${API_URL}/predict/text/full`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sentences[i] }),
        });
        if (!r.ok) throw new Error(`classify ${r.status}`);
        const d = await r.json();
        const t0 = i * 4;
        segments.push({
          Start: hhmmss(t0), End: hhmmss(t0+3),
          Sentence: sentences[i],
          Emotion: d.top_emotion,
          Confidence: d.top_confidence,
        });
      }

      status.textContent = "⏳ Detecting scenes…";
      const sR = await fetch(`${API_URL}/predict/scenes`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments, min_segments_per_scene: 2, generate_titles: false }),
      });
      if (!sR.ok) throw new Error(`scenes ${sR.status}`);
      const sceneData = await sR.json();
      const scenes = sceneData.scenes || [];

      if (!scenes.length) throw new Error("Not enough content to build a trailer.");

      status.textContent = "⏳ AI writing taglines…";
      const trR = await fetch(`${API_URL}/predict/trailer`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenes, generate_taglines: true, max_clips: 5 }),
      });
      if (!trR.ok) throw new Error(`trailer ${trR.status}`);
      const tr = await trR.json();
      renderTrailerPage(tr);
      status.classList.add("is-ok");
      status.textContent = `✓ ${(tr.clips||[]).length} clips · ${Math.round(tr.total_duration||0)}s`;
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

 function renderTrailerPage(tr) {
    const clips = tr.clips || [];
    document.getElementById("trailerClipCount").textContent = clips.length;
    document.getElementById("trailerDuration").textContent = `${Math.round(tr.total_duration||0)}s`;
    document.getElementById("trailerMoods").textContent = (tr.moods_used||[]).length;

    const cards = document.getElementById("trailerCards");
    cards.innerHTML = clips.length ? clips.map((c, i) => {
      const color  = MOOD_COLORS[c.mood] || "#9ca3af";
      const color2 = MOOD_COLORS_2[c.mood] || "#7928ca";
      const icon   = MOOD_ICONS_SVG[c.mood] || MOOD_ICONS_SVG["Mixed"];
      return `
        <div class="trailer-card" style="--tc-color: ${color}; --tc-color-2: ${color2};">
          <div class="trailer-card-preview" style="--mood-icon: url(&quot;${icon}&quot;);">
            <div class="trailer-card-play"></div>
            <span class="trailer-card-time">${esc(c.start_label || "0:00")} · ${Math.round(c.duration||0)}s</span>
          </div>
          <div class="trailer-card-meta">
            <span class="trailer-card-num">Clip ${i+1}</span>
            <span class="trailer-card-mood" style="background: ${color}26; color: ${color};">${esc(c.mood)}</span>
          </div>
          <h4 class="trailer-card-tagline">${esc(c.tagline || c.mood || "Highlight")}</h4>
          <p class="trailer-card-hook">"${esc(c.hook || "")}"</p>
        </div>
      `;
    }).join("") : `<div class="empty-state">No trailer clips generated.</div>`;
  }
  
  // -------------------------------------------------------------------
  // ===================== SCORE PAGE ================================
  // -------------------------------------------------------------------
  const SCORE_DEMOS = {
    drama: "John walked angrily into the room. His face was twisted with fury. Sarah cried softly in the corner alone. Mark tried to calm everyone down. The tension was unbearable in that moment. Then suddenly Sarah laughed nervously. Everyone felt confused by her reaction.",
    comedy: "Mary Berry laughed at the soggy bottom joke. The contestants giggled together happily. Paul made another silly comment that everyone enjoyed. Susan beamed with joy as she presented her tart. The atmosphere was warm and full of joy. Everyone celebrated together with smiles.",
    political: "The president shouted angrily at the journalists. He cursed and slammed his fist. The reporters were stunned and silent. Then he stormed off stage in fury. The press secretary tried desperately to calm everyone. The room felt heavy with tension and shock.",
  };

  function initScorePage() {
    const ta = document.getElementById("scoreText");
    const c  = document.getElementById("scoreCounter");
    if (ta && c) {
      const update = () => {
        c.textContent = `${ta.value.length} / 3000`;
        c.style.color = ta.value.length > 2900 ? "var(--c-red)" : "";
      };
      ta.addEventListener("input", update);
      update();
    }
    document.querySelectorAll("[data-sco-quick]").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.scoQuick;
        if (ta && SCORE_DEMOS[k]) {
          ta.value = SCORE_DEMOS[k];
          ta.dispatchEvent(new Event("input"));
        }
      });
    });
    document.getElementById("btnScore")?.addEventListener("click", runScorePage);
  }

  async function runScorePage() {
    const btn = document.getElementById("btnScore");
    const status = document.getElementById("scoreStatus");
    const results = document.getElementById("scoreResults");
    const text = (document.getElementById("scoreText").value || "").trim();
    if (!text) { status.classList.add("is-error"); status.textContent = "❌ Empty text."; return; }

    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Classifying segments…";

    try {
      const sentences = splitSentences(text);
      const segments = [];
      for (let i = 0; i < sentences.length; i++) {
        const r = await fetch(`${API_URL}/predict/text/full`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sentences[i] }),
        });
        if (!r.ok) throw new Error(`classify ${r.status}`);
        const d = await r.json();
        const t0 = i * 4;
        segments.push({
          Start: hhmmss(t0), End: hhmmss(t0+3),
          Sentence: sentences[i],
          Emotion: d.top_emotion,
          Confidence: d.top_confidence,
        });
      }

      status.textContent = "⏳ AI generating scorecard…";
      const sR = await fetch(`${API_URL}/predict/score`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments, use_ai_classification: true }),
      });
      if (!sR.ok) throw new Error(`score ${sR.status}`);
      const score = await sR.json();
      renderScorePage(score);
      status.classList.add("is-ok");
      status.textContent = `✓ Grade: ${score.grade}`;
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderScorePage(score) {
    document.getElementById("scoreGradeBig").textContent = score.grade || "-";
    document.getElementById("scoreClassBig").textContent = score.classification || "Untitled";
    document.getElementById("scoreAudienceBig").textContent = score.target_audience || "";

    const sEl = document.getElementById("scoreSuitable");
    sEl.innerHTML = (score.suitable_for || []).map((f) => `<span class="score-suitable-pill">${esc(f)}</span>`).join("");

    const axEl = document.getElementById("scoreAxesBig");
    axEl.innerHTML = Object.entries(score.scores || {}).map(([k,v]) => {
      const pct = Math.round((v||0)*100);
      return `
        <div class="score-axis-big">
          <div class="score-axis-big-head">
            <span class="score-axis-big-label">${k.replace(/_/g," ")}</span>
            <span class="score-axis-big-val">${pct}%</span>
          </div>
          <div class="score-axis-big-bar"><span style="width:${pct}%;"></span></div>
        </div>
      `;
    }).join("");
  }

  // -------------------------------------------------------------------
  // ===================== SHIFTS PAGE ================================
  // -------------------------------------------------------------------
  const SHIFT_DEMOS = {
    rollercoaster: "I was so incredibly happy today. Everything was going wonderfully. Then suddenly I got the terrible news. My heart sank with sadness. I felt furious at the injustice. I shouted at everyone in anger. But then my friend made me laugh. The room filled with joy again.",
    news: "The crowd cheered with happiness at the victory. They celebrated together with joy. Then the protesters began shouting angrily. The atmosphere turned tense and fearful. Police arrived and calmed the situation. Eventually peace was restored quietly.",
  };
  const EMOTION_COLORS_RGB = {
    happiness: "#f5c518",
    neutral:   "#9ca3af",
    anger:     "#ef4444",
    sadness:   "#3b82f6",
    surprise:  "#fb923c",
    fear:      "#7928ca",
    disgust:   "#10b981",
  };

  function initShiftsPage() {
    const ta = document.getElementById("shiftsText");
    const c  = document.getElementById("shiftsCounter");
    if (ta && c) {
      const update = () => {
        c.textContent = `${ta.value.length} / 3000`;
        c.style.color = ta.value.length > 2900 ? "var(--c-red)" : "";
      };
      ta.addEventListener("input", update);
      update();
    }
    document.querySelectorAll("[data-sh-quick]").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.shQuick;
        if (ta && SHIFT_DEMOS[k]) {
          ta.value = SHIFT_DEMOS[k];
          ta.dispatchEvent(new Event("input"));
        }
      });
    });
    document.getElementById("btnShifts")?.addEventListener("click", runShiftsPage);
  }

  async function runShiftsPage() {
    const btn = document.getElementById("btnShifts");
    const status = document.getElementById("shiftsStatus");
    const results = document.getElementById("shiftsResults");
    const text = (document.getElementById("shiftsText").value || "").trim();
    if (!text) { status.classList.add("is-error"); status.textContent = "❌ Empty text."; return; }

    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Looking for turning points…";

    try {
      const sentences = splitSentences(text);
      const segments = [];
      for (let i = 0; i < sentences.length; i++) {
        const r = await fetch(`${API_URL}/predict/text/full`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sentences[i] }),
        });
        if (!r.ok) throw new Error(`classify ${r.status}`);
        const d = await r.json();
        const t0 = i * 4;
        segments.push({
          Start: hhmmss(t0), End: hhmmss(t0+3),
          Sentence: sentences[i],
          Emotion: d.top_emotion,
          Confidence: d.top_confidence,
        });
      }
      const sR = await fetch(`${API_URL}/predict/shifts`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments }),
      });
      if (!sR.ok) throw new Error(`shifts ${sR.status}`);
      const d = await sR.json();
      renderShiftsPage(d);
      status.classList.add("is-ok");
      status.textContent = `✓ ${d.shift_count} shifts detected`;
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderShiftsPage(d) {
    const shifts = d.shifts || [];
    document.getElementById("shiftsSummary").textContent =
      shifts.length ? `${shifts.length} turning points detected` : "No major shifts found";
    const el = document.getElementById("shiftsDetailed");
    el.innerHTML = shifts.length ? shifts.map((s) => {
      const fromColor = EMOTION_COLORS_RGB[s.from_emotion] || "#a1a1a1";
      const toColor   = EMOTION_COLORS_RGB[s.to_emotion]   || "#fff";
      return `
        <div class="shift-detail" style="--from-color: ${fromColor}; --to-color: ${toColor};">
          <span class="shift-detail-time">${esc(s.label || "0:00")}</span>
          <div class="shift-detail-body">
            <div class="shift-detail-arrow">
              <span class="shift-detail-from">${esc(s.from_emotion)}</span>
              <span class="shift-detail-arr">-></span>
              <span class="shift-detail-to">${esc(s.to_emotion)}</span>
            </div>
            <p class="shift-detail-sentence">"${esc(s.to_sentence || "")}"</p>
          </div>
          <span class="shift-detail-mag">${Math.round((s.magnitude||0)*100)}%</span>
        </div>
      `;
    }).join("") : `<div class="empty-state">No major emotion shifts detected.</div>`;
  }

  // -------------------------------------------------------------------
  // ===================== PERSUASION PAGE ===========================
  // -------------------------------------------------------------------
  const PERS_DEMOS = {
    trump: "I will make America great again. The opposition has destroyed everything. Only I can fix this disaster. We will win like never before. Our enemies fear our strength. The people will rise up with me. Together we will achieve incredible victories.",
    ted: "Today I want to share a powerful idea with you. Imagine a world where every problem has a solution. We have the tools and the knowledge to change everything. The future belongs to those who dare to dream. Every great achievement started with a single step. Let us take that step together right now.",
    ad: "This product will transform your life completely. Thousands of customers love it already. You deserve the very best quality possible. Order now before the limited offer ends. Join a community of happy users worldwide. Don't wait, your better life starts today.",
  };

  function initPersuasionPage() {
    const ta = document.getElementById("persuasionText");
    const c  = document.getElementById("persuasionCounter");
    if (ta && c) {
      const update = () => {
        c.textContent = `${ta.value.length} / 3000`;
        c.style.color = ta.value.length > 2900 ? "var(--c-red)" : "";
      };
      ta.addEventListener("input", update);
      update();
    }
    document.querySelectorAll("[data-pe-quick]").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.peQuick;
        if (ta && PERS_DEMOS[k]) {
          ta.value = PERS_DEMOS[k];
          ta.dispatchEvent(new Event("input"));
        }
      });
    });
    document.getElementById("btnPersuasion")?.addEventListener("click", runPersuasionPage);
  }

  async function runPersuasionPage() {
    const btn = document.getElementById("btnPersuasion");
    const status = document.getElementById("persuasionStatus");
    const results = document.getElementById("persuasionResults");
    const text = (document.getElementById("persuasionText").value || "").trim();
    if (!text) { status.classList.add("is-error"); status.textContent = "❌ Empty text."; return; }

    btn.disabled = true;
    status.classList.remove("is-error","is-ok");
    status.textContent = "⏳ Analysing rhetorical signals…";

    try {
      const sentences = splitSentences(text);
      const segments = [];
      for (let i = 0; i < sentences.length; i++) {
        const r = await fetch(`${API_URL}/predict/text/full`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sentences[i] }),
        });
        if (!r.ok) throw new Error(`classify ${r.status}`);
        const d = await r.json();
        const t0 = i * 4;
        segments.push({
          Start: hhmmss(t0), End: hhmmss(t0+3),
          Sentence: sentences[i],
          Emotion: d.top_emotion,
          Confidence: d.top_confidence,
        });
      }

      status.textContent = "⏳ AI writing verdict…";
      const pR = await fetch(`${API_URL}/predict/persuasion`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments, use_ai_assessment: true }),
      });
      if (!pR.ok) throw new Error(`persuasion ${pR.status}`);
      const p = await pR.json();
      renderPersuasionPage(p);
      status.classList.add("is-ok");
      status.textContent = `✓ ${p.confidence_level}`;
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      status.classList.add("is-error");
      status.textContent = `❌ ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderPersuasionPage(p) {
    const pct = Math.round((p.persuasion_score || 0) * 100);
    document.getElementById("persGaugeNum").textContent = `${pct}%`;
    document.getElementById("persGaugeLabel").textContent = (p.confidence_level || "").toUpperCase();

    // Animate gauge arc, circumference is 2*pi*r = 2*pi*86 ≈ 540
    const arc = document.getElementById("persGaugeArc");
    const circumference = 540;
    const offset = circumference * (1 - pct / 100);
    setTimeout(() => { arc.style.strokeDashoffset = offset; }, 100);

    document.getElementById("persAssessment").textContent = p.assessment || "-";

    const axesEl = document.getElementById("persAxes");
    axesEl.innerHTML = Object.entries(p.scores || {}).map(([k, v]) => {
      const valPct = Math.round((v||0)*100);
      return `
        <div class="persuasion-axis">
          <div class="persuasion-axis-label">${k.replace(/_/g," ")}</div>
          <div class="persuasion-axis-num">${valPct}%</div>
          <div class="persuasion-axis-bar"><span style="width:${valPct}%;"></span></div>
        </div>
      `;
    }).join("");
  }



// -------------------------------------------------------------------
  // ===================== HISTORY SYSTEM ============================
  // -------------------------------------------------------------------
  const HISTORY_KEY = "emotionscope_history_v1";
  const HISTORY_MAX = 50;

  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  }
  function saveHistory(arr) {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(arr.slice(0, HISTORY_MAX))); }
    catch (e) { console.warn("History save failed:", e); }
  }
  function addHistoryEntry(entry) {
    const arr = loadHistory();
    arr.unshift({ ...entry, id: Date.now() + "_" + Math.random().toString(36).slice(2,7), ts: Date.now() });
    saveHistory(arr);
    if (document.querySelector('[data-page="history"]:not([hidden])')) renderHistory();
  }
  function deleteHistoryEntry(id) {
    saveHistory(loadHistory().filter((e) => e.id !== id));
    renderHistory();
  }
  function clearHistory() {
    if (!confirm("Clear all history? This cannot be undone.")) return;
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  }
  function formatTimeAgo(ts) {
    const s = Math.floor((Date.now() - ts) / 1000);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s/60)}m ago`;
    if (s < 86400) return `${Math.floor(s/3600)}h ago`;
    const d = Math.floor(s/86400);
    if (d < 7) return `${d}d ago`;
    return new Date(ts).toLocaleDateString();
  }

  let historyFilter = "all";

  function initHistoryPage() {
    document.getElementById("historyTabs")?.addEventListener("click", (e) => {
      const tab = e.target.closest(".history-tab");
      if (!tab) return;
      historyFilter = tab.dataset.htab;
      document.querySelectorAll(".history-tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      renderHistory();
    });
    document.getElementById("btnClearHistory")?.addEventListener("click", clearHistory);
  }

  function renderHistory() {
    const list = document.getElementById("historyList");
    const empty = document.getElementById("historyEmpty");
    const count = document.getElementById("historyCount");
    if (!list) return;

    const all = loadHistory();
    const filtered = historyFilter === "all" ? all : all.filter((e) => e.type === historyFilter);
    count.textContent = `${all.length} saved`;

    if (!filtered.length) {
      list.innerHTML = "";
      empty.hidden = false;
      return;
    }
    empty.hidden = true;

    list.innerHTML = filtered.map((e) => {
      const icon = e.type === "analyse" ? "📝" : e.type === "compare" ? "⚖️" : "🎭";
      const snapshot = renderHistorySnapshot(e);
      return `
        <div class="history-item">
          <div class="history-item-icon" data-type="${e.type}">${icon}</div>
          <div class="history-item-body">
            <div class="history-item-meta">
              <span class="history-item-type" data-type="${e.type}">${e.type}</span>
              <span class="history-item-time">${formatTimeAgo(e.ts)}</span>
            </div>
            <h4 class="history-item-title">${esc(e.title || "Untitled")}</h4>
            <p class="history-item-snippet">${esc(e.snippet || "")}</p>
          </div>
          ${snapshot}
          <div class="history-item-actions">
            <button class="history-item-btn" title="Reopen" data-action="open" data-id="${e.id}">↗</button>
            <button class="history-item-btn history-item-btn--delete" title="Delete" data-action="delete" data-id="${e.id}">×</button>
          </div>
        </div>
      `;
    }).join("");

    list.querySelectorAll("[data-action]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.id;
        if (btn.dataset.action === "delete") {
          deleteHistoryEntry(id);
        } else if (btn.dataset.action === "open") {
          reopenHistory(id);
        }
      });
    });
  }

  function renderHistorySnapshot(e) {
    if (e.type === "analyse" && e.grade) {
      return `
        <div class="history-item-snapshot">
          <span class="history-item-stat-num">${esc(e.grade)}</span>
          <span class="history-item-stat-label">grade</span>
        </div>`;
    }
    if (e.type === "compare" && e.similarity != null) {
      return `
        <div class="history-item-snapshot">
          <span class="history-item-stat-num">${e.similarity}%</span>
          <span class="history-item-stat-label">similar</span>
        </div>`;
    }
    if (e.type === "cast" && e.characterCount != null) {
      return `
        <div class="history-item-snapshot">
          <span class="history-item-stat-num">${e.characterCount}</span>
          <span class="history-item-stat-label">chars</span>
        </div>`;
    }
    return `<div class="history-item-snapshot"></div>`;
  }

  function reopenHistory(id) {
    const item = loadHistory().find((e) => e.id === id);
    if (!item) return;
    const route = item.type === "cast" ? "cast" : item.type === "compare" ? "compare" : "analyse";
    history.pushState(null, "", `#${route}`);
    setRoute(route);
    setTimeout(() => {
      if (item.type === "analyse" && item.text) {
        const ta = document.getElementById("analyseText");
        if (ta) { ta.value = item.text; ta.dispatchEvent(new Event("input")); }
      } else if (item.type === "cast" && item.text) {
        const ta = document.getElementById("castText");
        if (ta) { ta.value = item.text; ta.dispatchEvent(new Event("input")); }
      } else if (item.type === "compare" && item.leftText && item.rightText) {
        const left  = document.querySelector('.twin-panel[data-side="left"]');
        const right = document.querySelector('.twin-panel[data-side="right"]');
        left.querySelector('[data-field="text"]').value = item.leftText;
        right.querySelector('[data-field="text"]').value = item.rightText;
        if (item.leftLabel)  left.querySelector('[data-field="label"]').value  = item.leftLabel;
        if (item.rightLabel) right.querySelector('[data-field="label"]').value = item.rightLabel;
      }
    }, 60);
  }

  // -------------------------------------------------------------------
  // BACK BUTTON + PAGE FOOTER (auto-inject on sub-pages)
  // -------------------------------------------------------------------
  function injectBackAndFooter() {
    const subPages = document.querySelectorAll('.page[data-page]:not([data-page="home"])');
    subPages.forEach((page) => {
      // Back button at top
      if (!page.querySelector(".back-home")) {
        const back = document.createElement("button");
        back.className = "back-home";
        back.innerHTML = `<span class="back-home-arrow"><-</span> Home`;
        back.addEventListener("click", () => {
          history.pushState(null, "", "#home");
          setRoute("home");
        });
        page.insertBefore(back, page.firstChild);
      }
      // Footer at bottom
      if (!page.querySelector(".page-footer")) {
        const footer = document.createElement("footer");
        footer.className = "page-footer";
        footer.innerHTML = `
          <div class="page-footer-col">
            <span class="page-footer-head">Product</span>
            <a href="#analyse" data-route="analyse">Analyse</a>
            <a href="#compare" data-route="compare">Compare</a>
            <a href="#history" data-route="history">History</a>
          </div>
          <div class="page-footer-col">
            <span class="page-footer-head">Features</span>
            <a href="#cast" data-route="cast">Cast</a>
            <a href="#scenes" data-route="scenes">Scenes</a>
            <a href="#trailer" data-route="trailer">Trailer</a>
          </div>
          <div class="page-footer-col">
            <span class="page-footer-head">Project</span>
            <a href="#about" data-route="about">About</a>
            <a href="https://www.contentintelligence.nl/" target="_blank" rel="noopener">Client (CIA)</a>
            <a href="https://www.buas.nl/" target="_blank" rel="noopener">BUas</a>
          </div>
          <div class="page-footer-col">
            <span class="page-footer-head">Tech</span>
            <a href="/docs" target="_blank">API docs</a>
            <a href="https://groq.com" target="_blank" rel="noopener">Groq</a>
            <a href="https://fastapi.tiangolo.com" target="_blank" rel="noopener">FastAPI</a>
          </div>
          <div class="page-footer-meta">
            Made by <span class="page-footer-brand">NLP Group 1</span> · ▲ EmotionScope · Block D Sprint 4 · BUas 2026
          </div>
        `;
        page.appendChild(footer);
      }
    });
  }

  // -------------------------------------------------------------------
  // TOPNAV SCROLL EFFECT
  // -------------------------------------------------------------------
  function initTopnavScroll() {
    const topnav = document.querySelector(".topnav");
    if (!topnav) return;
    const onScroll = () => {
      topnav.classList.toggle("is-scrolled", window.scrollY > 20);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  // -------------------------------------------------------------------
  // ASK THE AI, Q&A over the transcript of the last analysis.
  // Calls the backend /ask endpoint (Groq LLM). The model only sees the
  // transcript segments, so answers stay grounded in the actual content
  // instead of being invented.
  // -------------------------------------------------------------------
  function resetQaPanel() {
    const thread = document.getElementById("qaThread");
    const status = document.getElementById("qaStatus");
    if (thread) thread.innerHTML = "";
    if (status) { status.textContent = ""; status.classList.remove("is-error", "is-ok"); }
  }

  function appendQaMessage(kind, text) {
    // kind is "q" (user question) or "a" (AI answer) or "err" (failure)
    const thread = document.getElementById("qaThread");
    if (!thread) return null;
    const el = document.createElement("div");
    el.className = kind === "q" ? "qa-msg qa-msg--q" : "qa-msg qa-msg--a";
    if (kind === "err") el.classList.add("qa-msg--err");
    el.textContent = text;
    thread.appendChild(el);
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return el;
  }

  function initQaPanel() {
    const btn = document.getElementById("btnQa");
    const input = document.getElementById("qaInput");
    const status = document.getElementById("qaStatus");
    if (!btn || !input) return;

    async function ask() {
      const question = input.value.trim();
      if (!question) return;
      if (!lastResult || !(lastResult.segments || []).length) {
        status.classList.add("is-error");
        status.textContent = "❌ Run an analysis first, the AI answers from its transcript.";
        return;
      }
      btn.disabled = true;
      status.classList.remove("is-error", "is-ok");
      status.textContent = "⏳ Thinking…";
      appendQaMessage("q", question);
      input.value = "";

      try {
        const r = await fetch(`${API_URL}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, segments: lastResult.segments }),
        });
        if (r.status === 503) throw new Error("AI service is not configured on the server (GROQ_API_KEY).");
        if (!r.ok) throw new Error(`status ${r.status}`);
        const d = await r.json();
        appendQaMessage("a", d.answer || "(no answer)");
        status.classList.add("is-ok");
        status.textContent = `✓ Answered from ${d.context_segments} transcript lines`;
      } catch (err) {
        appendQaMessage("err", err.message);
        status.classList.add("is-error");
        status.textContent = `❌ ${err.message}`;
      } finally {
        btn.disabled = false;
        input.focus();
      }
    }

    btn.addEventListener("click", ask);
    // Enter in the input asks too, feels like chat.
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
  }

  // -------------------------------------------------------------------
  // ADMIN GATE for the Monitor page.
  // The backend checks credentials (POST /admin/login) and returns a
  // signed, expiring token. We keep it in sessionStorage so closing the
  // tab signs the admin out again. The dashboard content stays hidden
  // until the backend confirms the token (GET /admin/verify).
  // -------------------------------------------------------------------
  const ADMIN_TOKEN_KEY = "emotionscope_admin_token";

  async function guardDashboard() {
    const login = document.getElementById("dashLogin");
    const content = document.getElementById("dashContent");
    if (!login || !content) { loadDashboard(); return; } // markup missing, behave as before

    // Ask the backend whether the stored token is still good. The check is
    // server-side on purpose, hiding a div in JS alone would not be a gate.
    let ok = false;
    const token = sessionStorage.getItem(ADMIN_TOKEN_KEY) || "";
    if (token) {
      try {
        const r = await fetch(`${API_URL}/admin/verify`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        ok = r.ok;
      } catch (_) { ok = false; }
    }
    if (!ok) sessionStorage.removeItem(ADMIN_TOKEN_KEY);

    login.hidden = ok;
    content.hidden = !ok;
    if (ok) loadDashboard();
  }

  function initAdminGate() {
    const btn = document.getElementById("dashLoginBtn");
    const status = document.getElementById("dashLoginStatus");
    const user = document.getElementById("dashLoginUser");
    const pass = document.getElementById("dashLoginPass");
    if (!btn) return;

    async function submit() {
      const username = (user?.value || "").trim();
      const password = pass?.value || "";
      if (!username || !password) {
        status.classList.add("is-error");
        status.textContent = "❌ Enter both a username and a password.";
        return;
      }
      btn.disabled = true;
      status.classList.remove("is-error", "is-ok");
      status.textContent = "⏳ Checking…";
      try {
        const r = await fetch(`${API_URL}/admin/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        if (r.status === 401) throw new Error("Wrong username or password.");
        if (!r.ok) throw new Error(`status ${r.status}`);
        const d = await r.json();
        sessionStorage.setItem(ADMIN_TOKEN_KEY, d.token);
        pass.value = "";
        status.classList.add("is-ok");
        status.textContent = "✓ Signed in";
        guardDashboard();
      } catch (err) {
        status.classList.add("is-error");
        status.textContent = `❌ ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    }

    btn.addEventListener("click", submit);
    pass?.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
    document.getElementById("dashLogout")?.addEventListener("click", () => {
      sessionStorage.removeItem(ADMIN_TOKEN_KEY);
      guardDashboard();
    });
  }

  // -------------------------------------------------------------------
  // DASHBOARD, feedback flywheel + model monitoring + pipeline perf
  // -------------------------------------------------------------------
  async function loadDashboard() {
    try {
      const [stats, check, endpointInfo] = await Promise.all([
        fetch(`${API_URL}/feedback/stats`).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`${API_URL}/feedback/auto-check`).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`${API_URL}/endpoint/info`).then(r => r.ok ? r.json() : null).catch(() => null),
      ]);

      // --- KPIs ---
      if (stats) {
        document.getElementById("dash-total").textContent = stats.total ?? "-";
        document.getElementById("dash-agreement").textContent =
          stats.agreement_rate != null ? (stats.agreement_rate * 100).toFixed(1) + "%" : "-";
      }
      if (check) {
        document.getElementById("dash-corrections").textContent = check.new_corrections ?? "-";
        document.getElementById("dash-remaining").textContent = check.remaining ?? "-";
        const pct = Math.min(100, Math.round(((check.new_corrections || 0) / (check.threshold || 20)) * 100));
        document.getElementById("dash-flywheel-fill").style.width = pct + "%";
        document.getElementById("dash-flywheel-text").textContent =
          `${check.new_corrections ?? 0} / ${check.threshold ?? 20}`;
        document.getElementById("dash-flywheel-note").textContent =
          check.ready
            ? "✓ Threshold reached, retrain can be triggered."
            : `${check.remaining ?? "?"} more corrections needed to trigger auto-retrain.`;
      }

      // --- Pipeline perf from localStorage ---
      const hist = loadHistory().filter(e => e.t_total);
      document.getElementById("dash-runs").textContent = hist.length || loadHistory().length;
      if (hist.length) {
        const avg = (arr) => (arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(2) + "s";
        document.getElementById("dash-avg-latency").textContent = avg(hist.map(e=>e.t_total));
        document.getElementById("dash-t-whisper").textContent   = avg(hist.map(e=>e.t_whisper||0));
        document.getElementById("dash-t-translate").textContent = avg(hist.map(e=>e.t_translate||0));
        document.getElementById("dash-t-classify").textContent  = avg(hist.map(e=>e.t_classify||0));
        document.getElementById("dash-t-total").textContent     = avg(hist.map(e=>e.t_total||0));
      } else {
        document.getElementById("dash-avg-latency").textContent = "-";
      }

      // --- Endpoint info ---
      if (endpointInfo) {
        document.getElementById("dash-endpoint-name").textContent    = endpointInfo.endpoint_name || "-";
        document.getElementById("dash-endpoint-uri").textContent     = endpointInfo.scoring_uri || "-";
        document.getElementById("dash-endpoint-status").textContent  = endpointInfo.provisioning_state || "-";
        const traffic = endpointInfo.traffic || {};
        document.getElementById("dash-endpoint-traffic").textContent =
          Object.entries(traffic).map(([k,v]) => `${k}: ${v}%`).join(" · ") || "-";
      }

      // --- Per-model cards ---
      const modelCards = document.getElementById("dash-model-cards");
      if (stats && stats.per_model && stats.per_model.length) {
        modelCards.innerHTML = stats.per_model.map(m => {
          const pct = m.agreement_pct != null ? m.agreement_pct.toFixed(1) : "-";
          const color = m.agreement_pct >= 70 ? "#22c55e" : m.agreement_pct >= 40 ? "#f59e0b" : "#ef4444";
          return `
          <div class="dash-model-card">
            <div class="dash-model-card-name">${m.model_version}</div>
            <div class="dash-model-card-stats">
              <div class="dash-model-stat"><span>${m.n}</span><small>feedbacks</small></div>
              <div class="dash-model-stat" style="color:${color}"><span>${pct}%</span><small>agreement</small></div>
            </div>
            <div class="dash-model-bar-track"><div class="dash-model-bar-fill" style="width:${pct === '-' ? 0 : pct}%;background:${color}"></div></div>
          </div>`;
        }).join("");
      }

      // --- Retrain history ---
      const histEl = document.getElementById("dash-retrain-history");
      if (check && check.history && check.history.length) {
        histEl.innerHTML = check.history.map(h => `
          <div class="dash-retrain-row">
            <span>${h.started_at ? h.started_at.slice(0,16).replace("T"," ") : "-"}</span>
            <span class="dash-retrain-status" data-s="${h.status}">${h.status}</span>
            <span>${h.corrections_count ?? 0} corrections</span>
            <span>${h.new_model_version || "-"}</span>
          </div>`).join("");
      }
    } catch (e) {
      console.warn("[Dashboard] Failed to load:", e);
    }
  }

  // -------------------------------------------------------------------
  // BOOT
  // -------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initRouting();
    initTopnavScroll();
    initBurger();
    initTabGroup();
    initTemplates();
    initAnalyseTabs();
    initAnalyseCounter();
    initQuickButtons();
    initFileInputs();
    initRandomSample();
    initCastPage();
    initComparePage();
    initTimelinePage();
    initScenesPage();
    initTrailerPage();
    initScorePage();
    initShiftsPage();
    initPersuasionPage();
    initHistoryPage();
    renderHistory();
    initQaPanel();
    initAdminGate();
    document.getElementById("btnAnalyse")?.addEventListener("click", runAnalyse);
    try { injectBackAndFooter(); } catch (e) { console.error("injectBackAndFooter failed:", e); }
    checkHealth();
    console.log("[EmotionScope v3] Foundation + Analyse loaded.");
  });

})();
