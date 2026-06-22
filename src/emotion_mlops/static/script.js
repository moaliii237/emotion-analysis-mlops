/* ===================================
   EmotionScope - Frontend Logic
   Premium sade, emoji'siz
   =================================== */

const API_URL = window.location.origin;
const STORAGE_KEY_RUNS = "emotionscope-runs";
const STORAGE_KEY_THEME = "emotionscope-theme";

// Emotion metadata, only color + description (no emoji)
const EMOTION_META = {
    anger:     { color: "#FF5454", desc: "Strong displeasure or hostility" },
    disgust:   { color: "#A3E635", desc: "Strong aversion or revulsion" },
    fear:      { color: "#C084FC", desc: "Anxiety, worry, or apprehension" },
    happiness: { color: "#FFC93C", desc: "Joy, contentment, or pleasure" },
    neutral:   { color: "#94A3B8", desc: "No strong emotional signal" },
    sadness:   { color: "#5BA3F8", desc: "Sorrow, grief, or unhappiness" },
    surprise:  { color: "#FF8A4C", desc: "Astonishment or unexpected reaction" },
};

const EMOTIONS = Object.keys(EMOTION_META);

const SAMPLES = [
    "I cannot believe this is happening to me!",
    "I love spending time with my family.",
    "I feel so lonely and empty inside.",
    "That smells absolutely revolting.",
    "Listen to me right now!",
    "What a wonderful surprise!",
    "I am terrified of what comes next.",
    "Everything is perfectly fine, I guess.",
    "This is the best day of my entire life!",
    "Why does this always happen to me?",
];

// State
let currentMode = "text";
let selectedAudioFile = null;
let selectedCsvFile = null;
let runs = [];
let lastResult = null;
let lastResultIsPipeline = false;

let latencyChart = null;
let stepChart = null;

// ========== Helpers ==========
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getThemeColors() {
    const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
    return {
        text: isDark ? "#FAFAFA" : "#18181B",
        textSoft: isDark ? "#A1A1AA" : "#52525B",
        grid: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)",
    };
}

// ========== TOAST ==========
function showToast(message, type = "info", title = null, duration = 4000) {
    const container = document.getElementById("toast-container");
    if (!container) return;
    
    const icons = { success: "✓", error: "!", warning: "!", info: "i" };
    const defaultTitles = { success: "Done", error: "Error", warning: "Warning", info: "Info" };
    
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <div class="toast-content">
            <div class="toast-title">${title || defaultTitles[type]}</div>
            <div class="toast-message">${escapeHtml(message)}</div>
        </div>
        <button class="toast-close" aria-label="Close">×</button>
    `;
    
    const close = () => {
        toast.classList.add("removing");
        setTimeout(() => toast.remove(), 250);
    };
    
    toast.querySelector(".toast-close").addEventListener("click", close);
    if (duration > 0) setTimeout(close, duration);
    container.appendChild(toast);
}

// ========== THEME ==========
(function initTheme() {
    function getInitialTheme() {
        const stored = localStorage.getItem(STORAGE_KEY_THEME);
        if (stored) return stored;
        return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }
    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem(STORAGE_KEY_THEME, theme);
    }
    applyTheme(getInitialTheme());
    
    document.addEventListener("DOMContentLoaded", () => {
        const btn = document.getElementById("theme-toggle");
        if (!btn) return;
        btn.addEventListener("click", () => {
            const current = document.documentElement.getAttribute("data-theme") || "dark";
            const next = current === "dark" ? "light" : "dark";
            applyTheme(next);
            showToast(`Switched to ${next} mode`, "info", "Theme", 2000);
            if (latencyChart || stepChart) renderDashboard();
        });
    });
})();

// ========== PAGE NAVIGATION ==========
function switchPage(pageName) {
    document.querySelectorAll(".nav-link[data-page]").forEach(link => {
        link.classList.toggle("active", link.dataset.page === pageName);
    });
    document.querySelectorAll("[data-page-content]").forEach(page => {
        page.hidden = page.dataset.pageContent !== pageName;
    });
    
    if (pageName === "dashboard") renderDashboard();
    if (pageName === "history") renderHistory();
    
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ========== HEALTH ==========
async function checkApiHealth() {
    const status = document.getElementById("api-status");
    const statusText = document.querySelector(".status-text");
    if (!status || !statusText) return;
    
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        if (data.model_loaded) {
            status.classList.add("connected");
            status.classList.remove("error");
            statusText.textContent = `Connected · ${data.model_version}`;
        } else {
            status.classList.add("error");
            statusText.textContent = "Model not loaded";
        }
    } catch {
        status.classList.add("error");
        statusText.textContent = "API offline";
    }
}

// ========== MODE SWITCH ==========
function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.mode === mode));
    document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.dataset.panel === mode));
    
    document.getElementById("pipeline-settings").hidden = mode === "text";
    
    const btnText = document.getElementById("run-btn-text");
    if (btnText) btnText.textContent = mode === "text" ? "Analyse" : "Run pipeline";
    
    // Hide results
    document.getElementById("text-result").hidden = true;
    document.getElementById("pipeline-result").hidden = true;
    document.getElementById("error-card").hidden = true;
}

// ========== DROPZONE ==========
function setupDropzone(zoneId, inputId, infoId, nameId, sizeId, removeId, onFile) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    const info = document.getElementById(infoId);
    const nameEl = document.getElementById(nameId);
    const sizeEl = document.getElementById(sizeId);
    const removeBtn = document.getElementById(removeId);
    const content = zone.querySelector(".dropzone-content");
    
    function showFile(file) {
        content.hidden = true;
        info.hidden = false;
        info.style.display = "inline-flex";
        nameEl.textContent = file.name;
        sizeEl.textContent = formatBytes(file.size);
        onFile(file);
    }
    function clearFile() {
        content.hidden = false;
        info.hidden = true;
        input.value = "";
        onFile(null);
    }
    
    zone.addEventListener("click", (e) => {
        if (e.target === removeBtn || removeBtn.contains(e.target)) return;
        if (!info.hidden) return;
        input.click();
    });
    input.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) showFile(file);
    });
    removeBtn.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });
    
    ["dragenter", "dragover"].forEach(ev => {
        zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragging"); });
    });
    ["dragleave", "drop"].forEach(ev => {
        zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragging"); });
    });
    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) { input.files = e.dataTransfer.files; showFile(file); }
    });
}

// ========== PROGRESS ==========
function setProgress(percent, title) {
    const card = document.getElementById("progress-card");
    card.hidden = false;
    document.getElementById("progress-title").textContent = title || "Processing";
    document.getElementById("progress-fill").style.width = `${percent}%`;
}
function hideProgress() {
    document.getElementById("progress-card").hidden = true;
    resetSteps();
}
function setStep(stepName, state) {
    const step = document.querySelector(`.step[data-step="${stepName}"]`);
    if (!step) return;
    step.classList.remove("active", "done");
    if (state) step.classList.add(state);
}
function resetSteps() {
    ["upload", "whisper", "translate", "classify"].forEach(s => setStep(s, ""));
}

// ========== RUN HANDLER ==========
async function handleRun() {
    document.getElementById("error-card").hidden = true;
    document.getElementById("text-result").hidden = true;
    document.getElementById("pipeline-result").hidden = true;
    
    if (currentMode === "text") await runTextPrediction();
    else if (currentMode === "audio") await runPipeline("audio");
    else if (currentMode === "csv") await runPipeline("csv");
    else if (currentMode === "youtube") await runYouTube();
}

// ========== TEXT PREDICTION ==========
async function runTextPrediction() {
    const text = document.getElementById("text-input").value.trim();
    if (!text) { showError("Please enter a sentence first."); return; }
    
    const runBtn = document.getElementById("run-btn");
    runBtn.disabled = true;
    setProgress(50, "Analysing");
    
    try {
        const response = await fetch(`${API_URL}/predict/text/full`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: "Unknown error" }));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }
        const result = await response.json();
        hideProgress();
        renderTextResult(result, text);
        showToast(`${result.top_emotion} · ${(result.top_confidence * 100).toFixed(1)}%`, "success", "Prediction");
        
        addRun({
            run_id: `run_${(runs.length + 1).toString().padStart(3, "0")}`,
            timestamp: new Date().toISOString(),
            input_type: "Text",
            text,
            top_emotion: result.top_emotion,
            top_confidence: result.top_confidence,
            all_emotions: result.all_emotions,
            segments: 1,
            t_total: 0, t_whisper: 0, t_translate: 0, t_classify: 0,
            emotions: [result.top_emotion],
            status: "success",
        });
    } catch (err) {
        hideProgress();
        showError(err.message);
    } finally {
        runBtn.disabled = false;
    }
}

// ========== PIPELINE (audio/csv) ==========
async function runPipeline(type) {
    const file = type === "audio" ? selectedAudioFile : selectedCsvFile;
    if (!file) { showError(`Please select a ${type} file first.`); return; }
    
    const runBtn = document.getElementById("run-btn");
    runBtn.disabled = true;
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("is_transcript", type === "csv" ? "true" : "false");
    formData.append("whisper_model", document.getElementById("whisper-model").value);
    formData.append("batch_size", document.getElementById("batch-size").value);
    formData.append("max_length", document.getElementById("max-length").value);
    
    setProgress(15, "Uploading file");
    setStep("upload", "active");
    
    try {
        const response = await fetch(`${API_URL}/predict/pipeline`, { method: "POST", body: formData });
        setStep("upload", "done");
        setStep("whisper", "active");
        setProgress(50, "Processing");
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: "Unknown error" }));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        setStep("whisper", "done"); setStep("translate", "done"); setStep("classify", "done");
        setProgress(100, "Done");
        
        setTimeout(() => {
            hideProgress();
            renderPipelineResult(data);
            showToast(`${data.emotions.length} segments · ${data.t_total}s`, "success", "Pipeline complete");
            addRun({
                run_id: `run_${(runs.length + 1).toString().padStart(3, "0")}`,
                timestamp: new Date().toISOString(),
                input_type: type === "csv" ? "Transcript CSV" : "Audio / Video",
                file_name: file.name,
                segments: data.emotions.length,
                t_whisper: data.t_whisper, t_translate: data.t_translate,
                t_classify: data.t_classify, t_total: data.t_total,
                emotions: data.emotions, data_segments: data.segments,
                status: "success",
            });
        }, 400);
    } catch (err) {
        hideProgress();
        showError(err.message);
        addRun({
            run_id: `run_${(runs.length + 1).toString().padStart(3, "0")}`,
            timestamp: new Date().toISOString(),
            input_type: type === "csv" ? "Transcript CSV" : "Audio / Video",
            file_name: file.name, error: err.message, segments: 0,
            t_whisper: 0, t_translate: 0, t_classify: 0, t_total: 0,
            emotions: [], status: "error",
        });
    } finally {
        runBtn.disabled = false;
    }
}

// ========== YOUTUBE ==========
async function runYouTube() {
    const url = document.getElementById("youtube-input").value.trim();
    if (!url) { showError("Please enter a YouTube URL."); return; }
    
    const runBtn = document.getElementById("run-btn");
    runBtn.disabled = true;
    
    const formData = new FormData();
    formData.append("url", url);
    formData.append("whisper_model", document.getElementById("whisper-model").value);
    formData.append("batch_size", document.getElementById("batch-size").value);
    formData.append("max_length", document.getElementById("max-length").value);
    
    setProgress(15, "Downloading from YouTube");
    setStep("upload", "active");
    
    try {
        const response = await fetch(`${API_URL}/predict/youtube`, { method: "POST", body: formData });
        setStep("upload", "done"); setStep("whisper", "active");
        setProgress(60, "Processing audio");
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: "Unknown error" }));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        setStep("whisper", "done"); setStep("translate", "done"); setStep("classify", "done");
        setProgress(100, "Done");
        
        setTimeout(() => {
            hideProgress();
            renderPipelineResult(data);
            showToast(`${data.emotions.length} segments · ${data.t_total}s`, "success", "YouTube complete");
            addRun({
                run_id: `run_${(runs.length + 1).toString().padStart(3, "0")}`,
                timestamp: new Date().toISOString(),
                input_type: "YouTube URL", url,
                segments: data.emotions.length,
                t_whisper: data.t_whisper, t_translate: data.t_translate,
                t_classify: data.t_classify, t_total: data.t_total,
                emotions: data.emotions, data_segments: data.segments,
                status: "success",
            });
        }, 400);
    } catch (err) {
        hideProgress();
        showError(err.message);
    } finally {
        runBtn.disabled = false;
    }
}

// ========== RENDER: TEXT RESULT ==========
function renderTextResult(result, text) {
    lastResult = result;
    lastResultIsPipeline = false;
    
    const card = document.getElementById("text-result");
    card.hidden = false;
    
    const emotion = result.top_emotion;
    const meta = EMOTION_META[emotion] || EMOTION_META.neutral;
    const confidencePct = (result.top_confidence * 100).toFixed(1);
    
    card.style.setProperty("--emotion-color", meta.color);
    
    document.getElementById("result-emotion-tag").style.background = meta.color;
    document.getElementById("result-emotion").textContent = emotion;
    document.getElementById("result-description").textContent = meta.desc;
    document.getElementById("confidence-value").textContent = `${confidencePct}%`;
    document.getElementById("meta-model").textContent = result.model_version;
    document.getElementById("meta-words").textContent = text.trim().split(/\s+/).length;
    document.getElementById("meta-chars").textContent = text.length;
    
    setTimeout(() => {
        document.getElementById("confidence-fill").style.width = `${confidencePct}%`;
    }, 50);
    
    renderEmotionBars(result.all_emotions, "emotion-bars");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    renderFeedbackUI(result, text);
}

// ========== RENDER: emotion bars (no emoji, color-coded) ==========
function renderEmotionBars(allEmotions, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const sorted = Object.entries(allEmotions).sort((a, b) => b[1] - a[1]);
    container.innerHTML = sorted.map(([emotion, score]) => {
        const meta = EMOTION_META[emotion] || EMOTION_META.neutral;
        const pct = (score * 100).toFixed(1);
        return `
            <div class="emotion-bar">
                <div class="emotion-bar-label">
                    <span class="emotion-bar-dot" style="background: ${meta.color}; color: ${meta.color};"></span>
                    <span>${emotion}</span>
                </div>
                <div class="emotion-bar-track">
                    <div class="emotion-bar-fill" style="width: ${pct}%; background: ${meta.color}; color: ${meta.color};"></div>
                </div>
                <span class="emotion-bar-value">${pct}%</span>
            </div>
        `;
    }).join("");
}

// ========== RENDER: PIPELINE RESULT ==========
function renderPipelineResult(data) {
    lastResult = data;
    lastResultIsPipeline = true;
    
    document.getElementById("pipeline-result").hidden = false;
    
    const emotions = data.emotions || [];
    const counts = {};
    emotions.forEach(e => { counts[e] = (counts[e] || 0) + 1; });
    const dominant = Object.keys(counts).length
        ? Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
        : "-";
    
    document.getElementById("kpi-segments").textContent = emotions.length;
    document.getElementById("kpi-dominant").textContent = dominant;
    if (dominant !== "-") {
        document.getElementById("kpi-dominant").style.color = EMOTION_META[dominant]?.color || "";
    }
    document.getElementById("kpi-runtime").textContent = `${data.t_total}s`;
    document.getElementById("kpi-unique").textContent = new Set(emotions).size;
    
    // Distribution as emotion bars
    const allFromList = {};
    emotions.forEach(e => { allFromList[e] = (allFromList[e] || 0) + 1; });
    const total = emotions.length || 1;
    const normalized = {};
    Object.entries(allFromList).forEach(([e, c]) => { normalized[e] = c / total; });
    renderEmotionBars(normalized, "pipeline-bars");
    
    renderTrace(data);
    renderTranscript(data.segments || []);
    
    document.getElementById("pipeline-result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTrace(data) {
    const list = document.getElementById("trace-list");
    const total = data.t_total || 1;
    const overhead = Math.max(total - (data.t_whisper || 0) - (data.t_translate || 0) - (data.t_classify || 0), 0);
    
    const rows = [
        { label: "Whisper ASR", time: data.t_whisper || 0, color: "#FFC93C" },
        { label: "MarianMT translation", time: data.t_translate || 0, color: "#5BA3F8" },
        { label: "DeBERTa classification", time: data.t_classify || 0, color: "#10A37F" },
        { label: "Overhead / I/O", time: overhead, color: "#94A3B8" },
    ];
    
    list.innerHTML = rows.map(r => `
        <div class="trace-row">
            <span class="trace-label">${r.label}</span>
            <div class="trace-track">
                <div class="trace-fill" style="width: ${(r.time / total * 100).toFixed(1)}%; background: ${r.color}; color: ${r.color};"></div>
            </div>
            <span class="trace-time">${r.time.toFixed(2)}s</span>
        </div>
    `).join("");
}

function renderTranscript(segments) {
    const body = document.getElementById("transcript-body");
    if (!segments || !segments.length) {
        body.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-mute);padding:1.5rem;">No segments</td></tr>';
        return;
    }
    body.innerHTML = segments.map((s, idx) => {
    const meta = EMOTION_META[s.Emotion] || EMOTION_META.neutral;
    const options = Object.keys(EMOTION_META).map(e => 
        `<option value="${e}" ${e === s.Emotion ? 'selected' : ''}>${e}</option>`
    ).join("");
    return `
        <tr>
            <td class="time-cell">${escapeHtml(s.Start || "")}</td>
            <td class="time-cell">${escapeHtml(s.End || "")}</td>
            <td>${escapeHtml(s.Sentence || "")}</td>
            <td style="color: var(--text-mute);">${escapeHtml(s.Translation || "")}</td>
            <td><span class="emotion-pill"><span class="emotion-pill-dot" style="background: ${meta.color};"></span>${s.Emotion}</span></td>
            <td>
                <button onclick="submitSegmentFeedback(${idx}, true)" style="border:none;background:none;cursor:pointer;color:#10A37F;font-size:1rem;">👍</button>
                <button onclick="submitSegmentFeedback(${idx}, false)" style="border:none;background:none;cursor:pointer;color:#FF5454;font-size:1rem;">👎</button>
                <select id="fb-seg-${idx}" style="padding:0.2rem 0.4rem;border-radius:6px;border:1px solid rgba(255,255,255,0.15);background:var(--surface,#1e1e2e);color:var(--text,#fafafa);font-size:0.75rem;display:none;">
                    ${options}
                </select>
            </td>
        </tr>
    `;
}).join("");
}

// ========== ERROR ==========
function showError(message) {
    document.getElementById("text-result").hidden = true;
    document.getElementById("pipeline-result").hidden = true;
    document.getElementById("error-text").textContent = message;
    document.getElementById("error-card").hidden = false;
    showToast(message, "error", "Failed");
}

// ========== RUNS PERSISTENCE ==========
function loadRuns() {
    try {
        runs = JSON.parse(localStorage.getItem(STORAGE_KEY_RUNS) || "[]");
    } catch { runs = []; }
}
function saveRuns() {
    try { localStorage.setItem(STORAGE_KEY_RUNS, JSON.stringify(runs)); }
    catch (e) { console.error(e); }
}
function addRun(run) {
    runs.push(run);
    if (runs.length > 50) runs = runs.slice(-50);
    saveRuns();
}

// ========== DOWNLOADS ==========
function downloadCsv() {
    if (!lastResult || !lastResult.segments) return;
    const headers = ["Start", "End", "Sentence", "Translation", "Emotion"];
    let csv = headers.join(",") + "\n";
    lastResult.segments.forEach(r => {
        csv += headers.map(h => `"${(r[h] || "").toString().replace(/"/g, '""')}"`).join(",") + "\n";
    });
    const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `emotions_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("CSV downloaded", "success");
}
function downloadJson() {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `emotions_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("JSON downloaded", "success");
}

// ========== DASHBOARD ==========
function renderDashboard() {
    const empty = document.getElementById("empty-dashboard");
    const content = document.getElementById("dashboard-content");
    if (!empty || !content) return;
    
    if (runs.length === 0) {
        empty.hidden = false;
        content.hidden = true;
        return;
    }
    
    empty.hidden = true;
    content.hidden = false;
    
    const successful = runs.filter(r => r.status === "success" && r.t_total > 0);
    const successRate = ((runs.filter(r => r.status === "success").length / runs.length) * 100).toFixed(1);
    const avgLatency = successful.length
        ? (successful.reduce((a, r) => a + r.t_total, 0) / successful.length).toFixed(2)
        : 0;
    const avgSegments = successful.length
        ? (successful.reduce((a, r) => a + r.segments, 0) / successful.length).toFixed(1)
        : 0;
    const totalSegments = runs.reduce((a, r) => a + r.segments, 0);
    
    document.getElementById("dash-total-runs").textContent = runs.length;
    document.getElementById("dash-success-rate").textContent = `${successRate}%`;
    document.getElementById("dash-avg-latency").textContent = `${avgLatency}s`;
    document.getElementById("dash-avg-segments").textContent = avgSegments;
    document.getElementById("dash-total-segments").textContent = totalSegments;
    
    // Latency chart
    const colors = getThemeColors();
    const latencyCtx = document.getElementById("latency-chart");
    if (latencyCtx && successful.length > 0) {
        if (latencyChart) latencyChart.destroy();
        const labels = successful.map(r => new Date(r.timestamp).toLocaleTimeString());
        latencyChart = new Chart(latencyCtx, {
            type: "line",
            data: {
                labels,
                datasets: [{
                    label: "Total (s)",
                    data: successful.map(r => r.t_total),
                    borderColor: "#10A37F",
                    backgroundColor: "rgba(16, 163, 127, 0.1)",
                    tension: 0.3,
                    fill: true,
                    pointRadius: 3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: colors.grid }, ticks: { color: colors.textSoft } },
                    y: { grid: { color: colors.grid }, ticks: { color: colors.textSoft } },
                },
            },
        });
    }
    
    // Step duration bar chart
    if (successful.length > 0) {
        const avgW = successful.reduce((a, r) => a + r.t_whisper, 0) / successful.length;
        const avgT = successful.reduce((a, r) => a + r.t_translate, 0) / successful.length;
        const avgC = successful.reduce((a, r) => a + r.t_classify, 0) / successful.length;
        
        const stepCtx = document.getElementById("step-chart");
        if (stepCtx) {
            if (stepChart) stepChart.destroy();
            stepChart = new Chart(stepCtx, {
                type: "bar",
                data: {
                    labels: ["Whisper", "Translation", "Classification"],
                    datasets: [{
                        data: [avgW, avgT, avgC],
                        backgroundColor: ["#FFC93C", "#5BA3F8", "#10A37F"],
                        borderRadius: 6,
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: colors.text } },
                        y: { grid: { color: colors.grid }, ticks: { color: colors.textSoft } },
                    },
                },
            });
        }
        
        // Bottleneck
        const stages = [["Whisper ASR", avgW], ["Translation", avgT], ["Classification", avgC]];
        const total = avgW + avgT + avgC;
        const [stageName, stageTime] = stages.sort((a, b) => b[1] - a[1])[0];
        const pct = total > 0 ? Math.round((stageTime / total) * 100) : 0;
        
        document.getElementById("bottleneck-box").hidden = false;
        document.getElementById("bottleneck-body").innerHTML =
            `<strong>${stageName}</strong> uses ${stageTime.toFixed(2)}s on average (${pct}% of total). Consider GPU acceleration to reduce this.`;
    }
}

// ========== HISTORY ==========
function renderHistory() {
    const container = document.getElementById("history-runs");
    if (!container) return;
    
    if (runs.length === 0) {
        container.innerHTML = '<div class="empty"><p>No runs yet.</p></div>';
        return;
    }
    
    let filtered = [...runs];
    const statusFilter = document.getElementById("history-status-filter")?.value || "all";
    const sortBy = document.getElementById("history-sort")?.value || "newest";
    
    if (statusFilter !== "all") filtered = filtered.filter(r => r.status === statusFilter);
    
    if (sortBy === "newest") filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    else if (sortBy === "oldest") filtered.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    else if (sortBy === "fastest") filtered.sort((a, b) => a.t_total - b.t_total);
    else if (sortBy === "slowest") filtered.sort((a, b) => b.t_total - a.t_total);
    
    if (filtered.length === 0) {
        container.innerHTML = '<div class="empty"><p>No runs match the filter.</p></div>';
        return;
    }
    
    container.innerHTML = filtered.map(r => {
        const time = new Date(r.timestamp).toLocaleString();
        return `
            <div class="run-row">
                <span class="run-id">${r.run_id}</span>
                <span class="run-time">${time}</span>
                <span class="run-stat">${r.segments} seg</span>
                <span class="run-stat">${r.t_total}s</span>
                <span class="run-status ${r.status}">${r.status}</span>
            </div>
        `;
    }).join("");
}

// ========== INIT ==========
document.addEventListener("DOMContentLoaded", () => {
    loadRuns();
    checkApiHealth();
    setInterval(checkApiHealth, 30000);
    
    // Page nav
    document.querySelectorAll(".nav-link[data-page]").forEach(link => {
        link.addEventListener("click", (e) => { e.preventDefault(); switchPage(link.dataset.page); });
    });
    
    // Tabs
    document.querySelectorAll(".tab").forEach(btn => {
        btn.addEventListener("click", () => switchMode(btn.dataset.mode));
    });
    
    // Text input
    const textInput = document.getElementById("text-input");
    const charCount = document.getElementById("char-count");
    if (textInput && charCount) {
        textInput.addEventListener("input", () => { charCount.textContent = textInput.value.length; });
        textInput.addEventListener("keydown", (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                handleRun();
            }
        });
    }
    
    // Chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const text = chip.dataset.text;
            textInput.value = text;
            charCount.textContent = text.length;
            handleRun();
        });
    });
    
    // Buttons
    document.getElementById("run-btn").addEventListener("click", handleRun);
    document.getElementById("random-btn").addEventListener("click", () => {
        if (currentMode !== "text") switchMode("text");
        const random = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
        textInput.value = random;
        charCount.textContent = random.length;
        handleRun();
    });
    
    // Dropzones
    setupDropzone(
        "audio-dropzone", "audio-input", "audio-file-info",
        "audio-file-name", "audio-file-size", "audio-file-remove",
        (file) => { selectedAudioFile = file; }
    );
    setupDropzone(
        "csv-dropzone", "csv-input", "csv-file-info",
        "csv-file-name", "csv-file-size", "csv-file-remove",
        (file) => { selectedCsvFile = file; }
    );
    
    // Downloads
    document.getElementById("download-csv")?.addEventListener("click", downloadCsv);
    document.getElementById("download-json")?.addEventListener("click", downloadJson);
    
    // Clear history
    document.getElementById("clear-all-history")?.addEventListener("click", () => {
        if (confirm("Clear all run history?")) {
            runs = [];
            saveRuns();
            renderHistory();
            renderDashboard();
            showToast("History cleared", "info");
        }
    });
    
    // History filters
    document.getElementById("history-status-filter")?.addEventListener("change", renderHistory);
    document.getElementById("history-sort")?.addEventListener("change", renderHistory);
});

// ========== PARTICLES BACKGROUND ==========
(function initParticles() {
    const canvas = document.getElementById("particle-canvas");
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    let particles = [];
    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;
    
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", (e) => { mouseX = e.clientX; mouseY = e.clientY; });
    
    const colors = ["#5BA3F8", "#C084FC", "#10A37F", "#FF8A4C", "#FFC93C"];
    for (let i = 0; i < 35; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.25,
            vy: (Math.random() - 0.5) * 0.25,
            size: Math.random() * 1.8 + 0.8,
            color: colors[Math.floor(Math.random() * colors.length)],
            opacity: Math.random() * 0.4 + 0.15,
        });
    }
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            // Subtle mouse attraction
            const dx = mouseX - p.x, dy = mouseY - p.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 150) { p.x += dx * 0.0005; p.y += dy * 0.0005; }
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
            
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = p.opacity;
            ctx.shadowBlur = 10;
            ctx.shadowColor = p.color;
            ctx.fill();
        });
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
        
        // Connect close particles
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(255,255,255,${0.08 * (1 - dist / 100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
})();
// ========== FEEDBACK ==========
let lastPredictionForFeedback = null;

function renderFeedbackUI(result, inputText) {
    lastPredictionForFeedback = { result, inputText };
    
    const card = document.getElementById("text-result");
    const existing = document.getElementById("feedback-section");
    if (existing) existing.remove();
    
    const section = document.createElement("div");
    section.id = "feedback-section";
    section.style.cssText = "margin-top:1.5rem; padding:1rem 1.25rem; border:1px solid rgba(255,255,255,0.08); border-radius:12px;";
    section.innerHTML = `
        <p style="margin:0 0 0.75rem; font-size:0.85rem; color:var(--text-mute);">Was this prediction correct?</p>
        <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
            <button id="fb-yes" style="padding:0.4rem 1rem; border-radius:8px; border:none; background:#10A37F; color:#fff; cursor:pointer;">👍 Yes</button>
            <button id="fb-no"  style="padding:0.4rem 1rem; border-radius:8px; border:none; background:#FF5454; color:#fff; cursor:pointer;">👎 No</button>
            <span id="fb-correction" hidden style="display:flex; gap:0.5rem; align-items:center;">
                <select id="fb-label" style="padding:0.35rem 0.5rem; border-radius:8px; border:1px solid rgba(255,255,255,0.15); background:var(--surface, #1e1e2e); color:var(--text, #fafafa);">
                    ${Object.keys(EMOTION_META).map(e => `<option value="${e}">${e}</option>`).join("")}
                </select>
                <button id="fb-submit" style="padding:0.4rem 0.9rem; border-radius:8px; border:none; background:#5BA3F8; color:#fff; cursor:pointer;">Submit</button>
            </span>
        </div>
        <p id="fb-thanks" hidden style="margin:0.5rem 0 0; font-size:0.82rem; color:#10A37F;"></p>
    `;
    card.appendChild(section);
    
    document.getElementById("fb-yes").addEventListener("click", () => submitFeedback(true));
    document.getElementById("fb-no").addEventListener("click", () => {
        document.getElementById("fb-correction").hidden = false;
    });
    document.getElementById("fb-submit").addEventListener("click", () => {
        const label = document.getElementById("fb-label").value;
        submitFeedback(false, label);
    });
}

async function submitFeedback(isCorrect, userLabel = null) {
    if (!lastPredictionForFeedback) return;
    const { result } = lastPredictionForFeedback;
    
    const payload = {
        sentence:        lastPredictionForFeedback.inputText,
        predicted_label: result.top_emotion,
        predicted_conf:  result.top_confidence,
        is_correct:      isCorrect,
        model_version:   result.model_version || "unknown",
        user_label:      userLabel,
        source:          "web-ui",
    };
    
    try {
        const resp = await fetch(`${API_URL}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        
        const thanks = document.getElementById("fb-thanks");
        thanks.hidden = false;
        thanks.textContent = isCorrect ? "Thanks! Marked as correct." : `Thanks! Corrected to "${userLabel}".`;
        document.getElementById("fb-yes").disabled = true;
        document.getElementById("fb-no").disabled = true;
        const submitBtn = document.getElementById("fb-submit");
        if (submitBtn) submitBtn.disabled = true;
        
        showToast("Feedback saved", "success", "Feedback", 2500);
    } catch (err) {
        showToast(`Feedback failed: ${err.message}`, "error");
    }
}

async function submitSegmentFeedback(idx, isCorrect) {
    const seg = lastResult.segments[idx];
    if (!seg) return;

    const select = document.getElementById(`fb-seg-${idx}`);
    
    if (!isCorrect) {
        select.style.display = "inline-block";
        select.onchange = () => doSubmitSegmentFeedback(idx, false, select.value);
        return;
    }
    doSubmitSegmentFeedback(idx, true, null);
}

async function doSubmitSegmentFeedback(idx, isCorrect, userLabel) {
    const seg = lastResult.segments[idx];
    const payload = {
        sentence: seg.Sentence,
        predicted_label: seg.Emotion,
        predicted_conf: seg.Confidence,
        is_correct: isCorrect,
        model_version: "azure-ml-endpoint",
        user_label: userLabel,
        source: "pipeline-ui",
    };
    try {
        const resp = await fetch(`${API_URL}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        showToast(isCorrect ? "Feedback saved" : `Corrected to "${userLabel}"`, "success", "Feedback", 2500);
    } catch (err) {
        showToast(`Feedback failed: ${err.message}`, "error");
    }
}