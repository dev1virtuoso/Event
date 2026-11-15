#!/usr/bin/env python3
"""
Clef-Flash Vision Decision Studio - Web UI
Powered by clef-flash (Decision-Native / Jev / System One API)

Flask backend + Tailwind-styled browser front end. This is the web sibling of
Draft/image_clef.py (pygame); the OpenCV visual-state extraction, the question
schema and the ollama.systemone call are ported over unchanged so both front
ends make the same decision for the same image.

Run:
    python Draft/image_clef_web.py
    # then open http://127.0.0.1:5000
"""

import os

# === Put your remote Ollama host config here (MUST be before `import ollama`) ===
# REMOTE_OLLAMA_HOST = "http://192.168.1.28:11434"
REMOTE_OLLAMA_HOST = "http://127.0.0.1:11434"
os.environ["OLLAMA_HOST"] = REMOTE_OLLAMA_HOST

import json
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import ollama
from flask import (
    Flask,
    abort,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
# Uploads live outside the static tree so they are never publicly servable.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "clef_flash_uploads"

MODEL_NAME = "clef-flash"
KEEP_ALIVE = "30m"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024          # 16 MB
MAX_IMAGE_PIXELS = 40_000_000               # guard against decompression bombs
JOB_TTL_SECONDS = 600                       # forget finished jobs after 10 min

# Folder shipped with the sprint that holds the 10-category sample images.
EXAMPLE_DIR_NAME = "PyConHK2026_Sprint"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".gif", ".tif", ".tiff", ".ppm", ".pgm",
}

# 10 Clef Choice Categories + Score + Noul (identical to the pygame script)
DEFAULT_QUESTIONS = {
    "scene_category": {
        "type": "choice",
        "instructions": (
            "Classify the primary visual category of this image based on its "
            "filename, layout, background brightness, edge complexity, and color profile."
        ),
        "criteria": {
            "ui_screenshot": "Application window, desktop UI, or web browser screenshot",
            "chart_or_plot": "Data visualization, bar chart, line graph, or scientific plot",
            "code_or_terminal": "Dark or light IDE, source code editor, or command-line terminal",
            "slide_or_deck": "Presentation slide, conference deck, or educational visual",
            "diagram_or_flowchart": "System architecture diagram, mind map, or technical schematic",
            "document_or_page": "Scanned paper document, PDF page, article, or text report",
            "spreadsheet_or_table": "Grid data, financial table, matrix, or CSV view",
            "receipt_or_invoice": "Point-of-sale receipt, billing invoice, or financial form",
            "natural_photo": "Real-world camera photograph of people, objects, or scenery",
            "illustration_or_graphic": "Digital artwork, logo, icon, poster, or meme",
        },
    },
    "information_density": {
        "type": "score",
        "instructions": "Rate the visual information and data density of this image.",
        "criteria": [
            "Low density (simple graphic, icon, or sparse layout)",
            "Medium density (standard chart, slide, or UI view)",
            "High density (dense text, complex plot, or detailed code)",
        ],
    },
    "needs_ocr_or_parsing": {
        "type": "noul",
        "instructions": "Does this image contain structured data, charts, or text that should be parsed?",
    },
    "recommended_action": {
        "type": "choice",
        "instructions": "Select the best automated pipeline action for this image.",
        "criteria": {
            "parse_chart_metrics": "Extract data series, labels, and metrics from chart/plot",
            "extract_text_ocr": "Run OCR to extract text, code, or document body",
            "analyze_ui_state": "Inspect interface layout or debug terminal/UI state",
            "archive_media": "Tag and index in visual asset library",
        },
    },
}


# --------------------------------------------------------------------------- #
# Sample-image discovery
# --------------------------------------------------------------------------- #

def resolve_example_dir(override=None) -> Path | None:
    """Locate the bundled sample-image folder (PyConHK2026_Sprint).

    Order: explicit override / CLI arg -> env var -> upward search from this
    script -> current working directory.
    """
    candidates = []

    if override:
        candidates.append(Path(override).expanduser())
    if os.environ.get("CLEF_IMAGE_DIR"):
        candidates.append(Path(os.environ["CLEF_IMAGE_DIR"]).expanduser())

    for parent in [APP_DIR, *APP_DIR.parents]:
        candidates.append(parent / EXAMPLE_DIR_NAME)

    candidates.append(Path.cwd() / EXAMPLE_DIR_NAME)

    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            continue
    return None


def list_sample_groups(root: Path) -> list[dict]:
    """Group the sample images by their sub-folder, sorted for stable display.

    Images sitting directly in the root are also included (as their own group),
    so pointing --samples at a flat folder still yields a usable gallery.
    """
    groups = []

    def collect(entries, prefix):
        images = []
        for f in entries:
            rel = f"{prefix}{f.name}" if prefix else f.name
            images.append({
                "name": f.name,
                # Relative to the sample root; the client posts this back.
                "path": rel,
                "url": f"/samples/{rel}",
            })
        return images

    try:
        root_files = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return groups

    for sub in subdirs:
        if sub.name.startswith("."):
            continue
        try:
            files = sorted(
                p for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        except OSError:
            continue
        if not files:
            continue
        groups.append({"folder": sub.name, "images": collect(files, f"{sub.name}/")})

    if root_files:
        groups.append({"folder": root.name, "images": collect(root_files, "")})

    return groups


# --------------------------------------------------------------------------- #
# Visual state extraction (identical maths to the pygame version)
# --------------------------------------------------------------------------- #

def extract_visual_state(file_path: str | Path) -> dict:
    """Extract structural and visual metrics via OpenCV for ollama.systemone."""
    path_obj = Path(file_path)
    img = cv2.imread(str(file_path))
    if img is None:
        return {"filename": path_obj.name, "error": "unreadable_image"}

    h, w = img.shape[:2]
    if h * w > MAX_IMAGE_PIXELS:
        return {"filename": path_obj.name, "error": "image_too_large"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    mean_bgr = img.mean(axis=(0, 1))
    std_bgr = img.std(axis=(0, 1))
    brightness = float(gray.mean())
    edge_density = float(np.count_nonzero(edges) / (h * w))

    # Ratio of very light pixels (common in charts, documents, slides)
    white_pixel_ratio = float(np.sum(gray > 235) / (h * w))
    # Ratio of very dark pixels (common in terminals, dark IDEs)
    dark_pixel_ratio = float(np.sum(gray < 35) / (h * w))

    return {
        "filename": path_obj.name,
        "extension": path_obj.suffix.lower(),
        "resolution": {"width": int(w), "height": int(h)},
        "aspect_ratio": round(w / max(1, h), 2),
        "brightness_mean_0_255": round(brightness, 1),
        "color_std_dev": round(float(std_bgr.mean()), 1),
        "rgb_mean": {
            "r": round(float(mean_bgr[2]), 1),
            "g": round(float(mean_bgr[1]), 1),
            "b": round(float(mean_bgr[0]), 1),
        },
        "edge_complexity_ratio": round(edge_density, 4),
        "white_background_ratio": round(white_pixel_ratio, 3),
        "dark_background_ratio": round(dark_pixel_ratio, 3),
        "layout_theme": (
            "light_background_chart_or_doc" if white_pixel_ratio > 0.45
            else ("dark_theme_terminal_or_ui" if dark_pixel_ratio > 0.45 else "mixed_color_scene")
        ),
    }


def normalise_response(resp) -> dict:
    """ollama returns a pydantic model, a dict or an object with .json()."""
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if isinstance(resp, dict):
        return resp
    return json.loads(resp.json())


# --------------------------------------------------------------------------- #
# Async job store
# --------------------------------------------------------------------------- #

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _sweep_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        stale = [jid for jid, job in _jobs.items() if job.get("created", 0) < cutoff]
        for jid in stale:
            _jobs.pop(jid, None)


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "created": time.time(),
            "result": None,
            "visual_state": None,
            "latency_ms": None,
            "error": None,
            "filename": None,
        }
    return job_id


def _run_job(job_id: str, image_path: Path, filename: str) -> None:
    started = time.time()
    state_payload = None
    try:
        state_payload = extract_visual_state(image_path)
        if "error" in state_payload:
            raise ValueError(f"Could not read image ({state_payload['error']})")

        # Same call as the pygame version - no unsupported 'images' kwarg.
        resp = ollama.systemone(
            model=MODEL_NAME,
            state=state_payload,
            questions=DEFAULT_QUESTIONS,
            keep_alive=KEEP_ALIVE,
        )
        result = normalise_response(resp)

        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["result"] = result
                job["visual_state"] = state_payload
                job["latency_ms"] = int((time.time() - started) * 1000)
                job["status"] = "done"
    except Exception as exc:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["error"] = str(exc)
                # Keep the metrics so the page can still show them when the
                # model call fails - the OpenCV half succeeded independently.
                if state_payload and "error" not in state_payload:
                    job["visual_state"] = state_payload
                job["latency_ms"] = int((time.time() - started) * 1000)
                job["status"] = "error"
    finally:
        # Best-effort cleanup of uploaded temp files.
        if "_upload_" in image_path.name:
            try:
                image_path.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.errorhandler(413)
def _too_large(_err):
    """The client always parses JSON, so never answer API calls with HTML."""
    return jsonify({
        "error": f"Image too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
    }), 413


@app.errorhandler(404)
def _not_found(_err):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return "Not found", 404


SAMPLE_ROOT = resolve_example_dir()


@app.get("/")
def index():
    return render_template_string(
        PAGE_TEMPLATE,
        sample_root_name=SAMPLE_ROOT.name if SAMPLE_ROOT else None,
        model_name=MODEL_NAME,
        ollama_host=REMOTE_OLLAMA_HOST,
    )


@app.get("/api/samples")
def api_samples():
    if SAMPLE_ROOT is None:
        return jsonify({"root": None, "groups": []})
    return jsonify({"root": SAMPLE_ROOT.name, "groups": list_sample_groups(SAMPLE_ROOT)})


@app.get("/samples/<path:relpath>")
def serve_sample(relpath: str):
    if SAMPLE_ROOT is None:
        abort(404)
    # send_from_directory rejects traversal outside the root.
    return send_from_directory(SAMPLE_ROOT, relpath)


@app.get("/api/health")
def api_health():
    """Report whether Ollama answers and whether clef-flash is pulled."""
    info = {
        "host": REMOTE_OLLAMA_HOST,
        "model": MODEL_NAME,
        "ollama_reachable": False,
        "model_available": False,
        "sample_root": SAMPLE_ROOT.name if SAMPLE_ROOT else None,
        "error": None,
    }
    try:
        listing = ollama.list()
        tags = getattr(listing, "models", None)
        if tags is None and isinstance(listing, dict):
            tags = listing.get("models", [])
        names = []
        for entry in tags or []:
            name = getattr(entry, "model", None) or (
                entry.get("model") if isinstance(entry, dict) else None
            )
            if name:
                names.append(name)
        info["ollama_reachable"] = True
        info["model_available"] = any(
            MODEL_NAME in n for n in names
        )
        info["models"] = names
    except Exception as exc:
        info["error"] = str(exc)
    return jsonify(info)


@app.post("/api/analyze")
def api_analyze():
    """Accept an uploaded file or a sample path; queue the work, return a job id."""
    _sweep_jobs()

    upload = request.files.get("image")
    sample = (request.form.get("sample") or "").strip()
    image_path: Path | None = None
    display_name = "image"

    if upload is not None and upload.filename:
        display_name = Path(upload.filename).name
        suffix = Path(display_name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {suffix or 'unknown'}"}), 400

        buffer = upload.read()
        if not buffer:
            return jsonify({"error": "Empty upload"}), 400

        upload_dir = UPLOAD_DIR
        upload_dir.mkdir(parents=True, exist_ok=True)
        image_path = upload_dir / f"_upload_{uuid.uuid4().hex}{suffix}"
        image_path.write_bytes(buffer)

    elif sample:
        if SAMPLE_ROOT is None:
            return jsonify({"error": "No sample folder configured"}), 400
        candidate = (SAMPLE_ROOT / sample).resolve()
        try:
            candidate.relative_to(SAMPLE_ROOT)
        except ValueError:
            return jsonify({"error": "Invalid sample path"}), 400
        if not candidate.is_file():
            return jsonify({"error": f"Sample not found: {sample}"}), 404
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            return jsonify({"error": "Sample is not an image"}), 400
        image_path = candidate
        display_name = candidate.name
    else:
        return jsonify({"error": "Provide either an 'image' file or a 'sample' path"}), 400

    job_id = _new_job()
    with _jobs_lock:
        _jobs[job_id]["filename"] = display_name

    threading.Thread(
        target=_run_job, args=(job_id, image_path, display_name), daemon=True
    ).start()
    return jsonify({"job_id": job_id, "filename": display_name}), 202


@app.get("/api/job/<job_id>")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job"}), 404
        payload = dict(job)
    payload.pop("created", None)
    return jsonify(payload)


PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Clef-Flash Vision Decision Studio</title>
  <link rel="stylesheet" href="/static/tailwind.css" />
</head>
<body class="min-h-full bg-bg text-text font-sans">

<div class="mx-auto max-w-6xl px-6 py-8">

  <!-- Header -->
  <header class="mb-6">
    <div class="flex flex-wrap items-baseline justify-between gap-3">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-accent">
          Clef-Flash Vision Decision Studio
        </h1>
        <p class="mt-1 text-sm text-muted">
          Decision-Native / Jev &middot; System One API &middot; 10-category classification
        </p>
      </div>
      <div class="flex items-center gap-2 text-xs font-mono">
        <span class="rounded-md bg-panel px-2.5 py-1 text-muted ring-1 ring-edge">
          model: <span class="text-text">{{ model_name }}</span>
        </span>
        <span id="healthBadge" class="rounded-md bg-panel px-2.5 py-1 text-muted ring-1 ring-edge">
          checking ollama...
        </span>
      </div>
    </div>
  </header>

  <!-- Main grid -->
  <div class="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">

    <!-- ============================ Left: input ============================ -->
    <section class="flex flex-col gap-4">

      <div class="rounded-xl bg-panel p-4 ring-1 ring-edge">
        <div class="mb-3 flex items-center justify-between">
          <h2 class="text-sm font-semibold tracking-wide text-muted uppercase">Image Input</h2>
          <span id="fileLabel" class="max-w-[180px] truncate text-xs text-muted"></span>
        </div>

        <!-- Drop zone: drag & drop OR click to browse -->
        <div id="dropzone"
             role="button"
             tabindex="0"
             aria-label="Drop an image here, or press Enter to browse files"
             class="group relative flex h-[300px] cursor-pointer flex-col items-center justify-center
                    overflow-hidden rounded-lg border-2 border-dashed border-edge bg-panelalt
                    transition-colors duration-200 hover:border-hover hover:bg-accent/5
                    focus:outline-none focus-visible:ring-2 focus-visible:ring-accent">

          <!-- Empty state -->
          <div id="dropIdle" class="pointer-events-none flex flex-col items-center px-6 text-center">
            <svg class="mb-4 h-14 w-14 text-muted transition-colors group-hover:text-accent"
                 viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="3" y="5" width="14" height="12" rx="2" />
              <rect x="7" y="3" width="14" height="12" rx="2" />
              <circle cx="12" cy="8" r="1.2" />
              <path d="M8 13l3-3 2.5 2.5L16 10l3 3" />
            </svg>
            <p class="text-sm font-medium text-text">Drag &amp; drop an image here</p>
            <p class="mt-1 text-sm text-accent">or click to browse</p>
            <p class="mt-3 font-mono text-[11px] tracking-wide text-muted">
              PNG / JPG / WebP / BMP
            </p>
            {% if sample_root_name %}
            <p class="mt-3 font-mono text-[11px] text-muted/70">
              Samples: {{ sample_root_name }}/
            </p>
            {% endif %}
          </div>

          <!-- Preview state (also clickable) -->
          <div id="dropPreview" class="pointer-events-none absolute inset-0 hidden items-center justify-center p-3">
            <img id="previewImg" alt="Selected image preview"
                 class="max-h-full max-w-full rounded object-contain" />
            <div class="absolute inset-x-0 bottom-0 bg-gradient-to-t from-panelalt/95 to-transparent px-3 py-2">
              <p class="text-xs font-medium text-text">Click to choose another image</p>
            </div>
          </div>

          <input id="fileInput" type="file" accept="image/*" class="hidden" />
        </div>

        <!-- Status line -->
        <div class="mt-3 flex items-start gap-2">
          <div id="spinner"
               class="mt-1 hidden h-3.5 w-3.5 shrink-0 animate-spin rounded-full
                      border-2 border-edge border-t-accent"></div>
          <p id="statusText" class="font-mono text-xs leading-relaxed text-muted">
            Ready. Drop an image, click the panel, or pick a sample below.
          </p>
        </div>
      </div>

      <!-- Visual state metrics -->
      <div id="metricsCard" class="hidden rounded-xl bg-panel p-4 ring-1 ring-edge">
        <h2 class="mb-3 text-sm font-semibold tracking-wide text-muted uppercase">
          Visual State (OpenCV)
        </h2>
        <dl id="metricsGrid" class="grid grid-cols-2 gap-x-4 gap-y-2 font-mono text-xs"></dl>
      </div>

    </section>

    <!-- =========================== Right: results ========================== -->
    <section class="flex flex-col gap-4">
      <div class="rounded-xl bg-panel p-5 ring-1 ring-edge">
        <div class="mb-4 flex items-center justify-between">
          <h2 class="text-sm font-semibold tracking-wide text-muted uppercase">
            Decision Output
          </h2>
          <span id="latencyBadge" class="hidden rounded-md bg-panelalt px-2 py-1 font-mono text-xs text-muted"></span>
        </div>
        <div id="results" class="space-y-5">
          <p class="text-sm text-muted">
            Waiting for image input to compute probabilities...
          </p>
        </div>
      </div>

      <!-- Raw JSON -->
      <details id="rawCard" class="group hidden rounded-xl bg-panel ring-1 ring-edge">
        <summary class="flex cursor-pointer list-none items-center justify-between px-5 py-3
                        text-sm font-semibold uppercase tracking-wide text-muted
                        hover:text-text">
          <span>Raw Response</span>
          <svg class="h-4 w-4 transition-transform group-open:rotate-180" viewBox="0 0 24 24"
               fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </summary>
        <pre id="rawJson" class="max-h-80 overflow-auto border-t border-edge px-5 py-4
                                font-mono text-[11px] leading-relaxed text-muted"></pre>
      </details>
    </section>
  </div>

  <!-- =========================== Sample gallery ========================== -->
  <section id="gallerySection" class="mt-8 hidden">
    <div class="mb-3 flex items-baseline justify-between">
      <h2 class="text-sm font-semibold uppercase tracking-wide text-muted">
        Sample Images
      </h2>
      <span id="galleryCount" class="font-mono text-xs text-muted"></span>
    </div>
    <div id="gallery" class="space-y-5"></div>
  </section>

  <footer class="mt-10 border-t border-edge pt-4 font-mono text-[11px] text-muted">
    ollama host: {{ ollama_host }}
  </footer>
</div>

<script>
(() => {
  "use strict";

  const IMAGE_EXTS = new Set([
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".gif", ".tif", ".tiff", ".ppm", ".pgm",
  ]);
  const POLL_INTERVAL_MS = 400;
  const POLL_TIMEOUT_MS = 180000;

  const $ = (id) => document.getElementById(id);

  const dropzone   = $("dropzone");
  const fileInput  = $("fileInput");
  const dropIdle   = $("dropIdle");
  const dropPreview= $("dropPreview");
  const previewImg = $("previewImg");
  const statusText = $("statusText");
  const spinner    = $("spinner");
  const fileLabel  = $("fileLabel");
  const results    = $("results");
  const rawCard    = $("rawCard");
  const rawJson    = $("rawJson");
  const metricsCard= $("metricsCard");
  const metricsGrid= $("metricsGrid");
  const latencyBadge = $("latencyBadge");
  const healthBadge  = $("healthBadge");

  let busy = false;

  // ---------------------------------------------------------------- utils ---
  function setStatus(msg, tone) {
    statusText.textContent = msg;
    statusText.className = "font-mono text-xs leading-relaxed " + (
      tone === "busy" ? "text-warn"
      : tone === "error" ? "text-red-400"
      : tone === "ok" ? "text-accent"
      : "text-muted"
    );
  }

  function setBusy(on) {
    busy = on;
    spinner.classList.toggle("hidden", !on);
    dropzone.classList.toggle("pointer-events-none", on);
    dropzone.classList.toggle("opacity-60", on);
  }

  function extOf(name) {
    const i = name.lastIndexOf(".");
    return i < 0 ? "" : name.slice(i).toLowerCase();
  }

  function isImage(name) {
    return IMAGE_EXTS.has(extOf(name));
  }

  function showPreview(src) {
    previewImg.src = src;
    dropIdle.classList.add("hidden");
    dropPreview.classList.remove("hidden");
    dropPreview.classList.add("flex");
  }

  function clearPreview() {
    previewImg.removeAttribute("src");
    dropPreview.classList.add("hidden");
    dropPreview.classList.remove("flex");
    dropIdle.classList.remove("hidden");
  }

  function esc(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function pct(v) {
    return (Math.max(0, Math.min(1, Number(v) || 0)) * 100).toFixed(1) + "%";
  }

  // ---------------------------------------------------------------- render ---
  function barRow(label, value, isWinner) {
    return `
      <div class="flex items-center gap-3">
        <span class="w-[150px] shrink-0 truncate font-mono text-[11px] ${isWinner ? "text-text" : "text-muted"}"
              title="${esc(label)}">${esc(label)}</span>
        <div class="prob-bar flex-1">
          <div class="prob-fill ${isWinner ? "prob-fill-win" : ""}" style="width:${pct(value)}"></div>
        </div>
        <span class="w-[52px] shrink-0 text-right font-mono text-[11px] tabular-nums ${isWinner ? "text-accent" : "text-muted"}">
          ${Number(value || 0).toFixed(3)}
        </span>
      </div>`;
  }

  function renderAnswer(key, ans) {
    const type = (ans && ans.type) || (typeof ans === "number" ? "noul" : "noul");

    if (type === "noul" || typeof ans === "number") {
      const prob = typeof ans === "number" ? ans
        : (ans.noul ?? ans.probability ?? 0);
      const yes = prob >= 0.5;
      return `
        <div class="border-b border-edge pb-4 last:border-0 last:pb-0">
          <div class="mb-2 flex flex-wrap items-center gap-2">
            <span class="rounded bg-panelalt px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-accent ring-1 ring-edge">noul</span>
            <span class="text-sm font-semibold text-text">${esc(key)}</span>
          </div>
          <div class="flex flex-wrap items-center gap-3">
            <span class="rounded px-2.5 py-1 font-mono text-xs font-bold ring-1
                         ${yes ? "bg-accent/15 text-accent ring-accent/40" : "bg-panelalt text-muted ring-edge"}">
              ${yes ? "TRUE" : "FALSE"}
            </span>
            <span class="font-mono text-xs text-muted">P = ${Number(prob || 0).toFixed(4)}</span>
          </div>
        </div>`;
    }

    if (type === "choice") {
      const choice = ans.choice ?? "-";
      const conf = Number(ans.confidence ?? 0);
      const probs = ans.probabilities || {};
      const rows = Object.entries(probs)
        .sort((a, b) => Number(b[1]) - Number(a[1]))
        .map(([k, v]) => barRow(k, v, k === choice))
        .join("");
      return `
        <div class="border-b border-edge pb-4 last:border-0 last:pb-0">
          <div class="mb-2 flex flex-wrap items-center gap-2">
            <span class="rounded bg-panelalt px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-accent ring-1 ring-edge">choice</span>
            <span class="text-sm font-semibold text-text">${esc(key)}</span>
          </div>
          <p class="mb-3 font-mono text-xs">
            <span class="text-muted">selected:</span>
            <span class="font-bold text-accent">${esc(choice)}</span>
            <span class="ml-2 text-muted">conf ${conf.toFixed(3)}</span>
          </p>
          <div class="space-y-1.5">${rows}</div>
        </div>`;
    }

    // score
    const score = Number(ans.score ?? 0);
    const conf = Number(ans.confidence ?? 0);
    const probs = ans.probabilities || {};
    const legend = ans.legend || {};
    const rows = Object.entries(probs)
      .map(([idx, v]) => barRow(legend[String(idx)] || `Level ${idx}`, v, false))
      .join("");
    return `
      <div class="border-b border-edge pb-4 last:border-0 last:pb-0">
        <div class="mb-2 flex flex-wrap items-center gap-2">
          <span class="rounded bg-panelalt px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-accent ring-1 ring-edge">score</span>
          <span class="text-sm font-semibold text-text">${esc(key)}</span>
        </div>
        <p class="mb-3 font-mono text-xs">
          <span class="text-muted">score:</span>
          <span class="font-bold text-text">${score.toFixed(3)}</span>
          <span class="ml-2 text-muted">conf ${conf.toFixed(3)}</span>
        </p>
        <div class="space-y-1.5">${rows}</div>
      </div>`;
  }

  function renderResult(payload) {
    const data = payload.result || {};
    const answers = data.answers || {};
    const keys = Object.keys(answers);

    if (!keys.length) {
      results.innerHTML = `<p class="text-sm text-muted">Model returned no answers.</p>`;
    } else {
      results.innerHTML = keys.map((k) => renderAnswer(k, answers[k])).join("");
    }

    latencyBadge.textContent = payload.latency_ms != null
      ? `${payload.latency_ms} ms` : "";
    latencyBadge.classList.toggle("hidden", payload.latency_ms == null);

    rawJson.textContent = JSON.stringify(data, null, 2);
    rawCard.classList.remove("hidden");
    rawCard.open = false;

    renderMetrics(payload.visual_state);
  }

  function renderMetrics(state) {
    if (!state || state.error) {
      metricsCard.classList.add("hidden");
      return;
    }
    const rows = [
      ["resolution", `${state.resolution.width} x ${state.resolution.height}`],
      ["aspect", state.aspect_ratio],
      ["brightness", state.brightness_mean_0_255],
      ["color std", state.color_std_dev],
      ["edges", state.edge_complexity_ratio],
      ["white bg", state.white_background_ratio],
      ["dark bg", state.dark_background_ratio],
      ["theme", state.layout_theme],
    ];
    metricsGrid.innerHTML = rows.map(([k, v]) => `
      <div class="flex items-baseline justify-between gap-2 border-b border-edge/50 pb-1">
        <dt class="text-muted">${esc(k)}</dt>
        <dd class="truncate text-right text-text" title="${esc(v)}">${esc(v)}</dd>
      </div>`).join("");
    metricsCard.classList.remove("hidden");
  }

  // --------------------------------------------------------------- network ---
  async function pollJob(jobId) {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    while (Date.now() < deadline) {
      const res = await fetch(`/api/job/${jobId}`);
      if (!res.ok) throw new Error(`Job lookup failed (${res.status})`);
      const job = await res.json();

      if (job.status === "done") return job;
      if (job.status === "error") {
        // Attach the job so the caller can still use the partial visual state.
        const e = new Error(job.error || "Analysis failed");
        e.job = job;
        throw e;
      }

      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    }
    throw new Error("Timed out waiting for the model");
  }

  async function analyze(formData, label) {
    if (busy) return;
    setBusy(true);
    setStatus(`Analyzing ${label} with clef-flash...`, "busy");

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      const body = await res.json();

      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);

      const job = await pollJob(body.job_id);
      renderResult(job);
      setStatus(
        `Decision complete (${job.latency_ms} ms) - ${job.filename || label}`,
        "ok"
      );
    } catch (err) {
      setStatus(`Error: ${err.message}`, "error");
      // Keep any OpenCV metrics the server managed to compute.
      if (err.job && err.job.visual_state) renderMetrics(err.job.visual_state);
      else metricsCard.classList.add("hidden");
    } finally {
      setBusy(false);
    }
  }

  function analyzeFile(file) {
    if (!isImage(file.name)) {
      setStatus(`Unsupported file type: ${extOf(file.name) || "unknown"}`, "error");
      return;
    }
    fileLabel.textContent = file.name;
    showPreview(URL.createObjectURL(file));

    const fd = new FormData();
    fd.append("image", file);
    analyze(fd, file.name);
  }

  function analyzeSample(path, name) {
    fileLabel.textContent = name;
    showPreview(`/samples/${path}`);

    const fd = new FormData();
    fd.append("sample", path);
    analyze(fd, name);
  }

  // ----------------------------------------------------------------- input ---
  // Click anywhere on the zone -> native file picker.
  dropzone.addEventListener("click", () => {
    if (!busy) fileInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !busy) {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) analyzeFile(file);
    fileInput.value = "";   // allow re-picking the same file
  });

  // Drag & drop
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("border-accent", "bg-accent/10");
    });
  });

  ["dragleave", "dragend"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      // Only reset when actually leaving the zone, not entering a child.
      if (evt === "dragleave" && dropzone.contains(e.relatedTarget)) return;
      dropzone.classList.remove("border-accent", "bg-accent/10");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("border-accent", "bg-accent/10");

    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) analyzeFile(files[0]);
  });

  // Prevent the browser from navigating when a file is dropped elsewhere.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // --------------------------------------------------------------- gallery ---
  async function loadGallery() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      const groups = data.groups || [];
      if (!groups.length) return;

      const total = groups.reduce((n, g) => n + g.images.length, 0);
      $("galleryCount").textContent = `${total} images / ${groups.length} folders`;

      $("gallery").innerHTML = groups.map((group) => `
        <div>
          <h3 class="mb-2 font-mono text-[11px] uppercase tracking-wider text-muted">${esc(group.folder)}</h3>
          <div class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
            ${group.images.map((img) => `
              <button type="button"
                      data-sample="${esc(img.path)}"
                      data-name="${esc(img.name)}"
                      class="group relative aspect-square overflow-hidden rounded-lg bg-panelalt ring-1 ring-edge
                             transition hover:ring-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                      title="${esc(img.name)}">
                <img src="${esc(img.url)}" alt="${esc(img.name)}" loading="lazy"
                     class="h-full w-full object-cover opacity-80 transition group-hover:scale-105 group-hover:opacity-100" />
                <span class="absolute inset-x-0 bottom-0 truncate bg-bg/85 px-1 py-0.5
                             font-mono text-[9px] text-muted opacity-0 transition group-hover:opacity-100">
                  ${esc(img.name)}
                </span>
              </button>`).join("")}
          </div>
        </div>`).join("");

      $("gallerySection").classList.remove("hidden");
    } catch (err) {
      console.warn("Gallery unavailable:", err);
    }
  }

  $("gallery").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-sample]");
    if (!btn || busy) return;
    analyzeSample(btn.dataset.sample, btn.dataset.name);
  });

  // ---------------------------------------------------------------- health ---
  async function loadHealth() {
    try {
      const res = await fetch("/api/health");
      const h = await res.json();
      if (!h.ollama_reachable) {
        healthBadge.textContent = "ollama offline";
        healthBadge.className =
          "rounded-md bg-warn/15 px-2.5 py-1 text-warn ring-1 ring-warn/40";
      } else if (!h.model_available) {
        healthBadge.textContent = `${h.model} not pulled`;
        healthBadge.className =
          "rounded-md bg-warn/15 px-2.5 py-1 text-warn ring-1 ring-warn/40";
      } else {
        healthBadge.textContent = `${h.model} ready`;
        healthBadge.className =
          "rounded-md bg-accent/15 px-2.5 py-1 text-accent ring-1 ring-accent/40";
      }
    } catch (err) {
      healthBadge.textContent = "ollama offline";
      healthBadge.className =
        "rounded-md bg-warn/15 px-2.5 py-1 text-warn ring-1 ring-warn/40";
    }
  }

  // ------------------------------------------------------------------ boot ---
  loadHealth();
  loadGallery();
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _port_available(host: str, port: int) -> bool:
    """Return True when we can bind host:port (catches the common conflicts)."""
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _explain_port_conflict(host: str, port: int) -> str:
    """Turn a bind failure into a message that says what to do about it."""
    hint = f"Port {port} is already in use."
    if port == 5000 and sys.platform == "darwin":
        hint += (
            "\n  On macOS the AirPlay Receiver often takes port 5000."
            "\n  Either retry with another port, e.g. --port 8000,"
            "\n  or turn off System Settings > General > AirDrop & Handoff >"
            "\n  AirPlay Receiver."
        )
    else:
        hint += f"\n  Pick another port, e.g. --port 8000."
    if host == "0.0.0.0":
        hint += (
            "\n  Note: binding 0.0.0.0 also conflicts when another process holds"
            "\n  the port on a specific interface. Try the default --host 127.0.0.1."
        )
    return hint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clef-Flash Vision Decision Studio (web)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="port (default 5000)")
    parser.add_argument("--debug", action="store_true", help="run Flask in debug mode")
    parser.add_argument(
        "--samples",
        default=None,
        metavar="DIR",
        help=f"override the sample-image folder (default: auto-detect {EXAMPLE_DIR_NAME})",
    )
    args = parser.parse_args()

    if args.samples:
        SAMPLE_ROOT = resolve_example_dir(args.samples)

    css = STATIC_DIR / "tailwind.css"
    print("Clef-Flash Vision Decision Studio")
    print(f"  ollama host : {REMOTE_OLLAMA_HOST}")
    print(f"  model       : {MODEL_NAME}")
    print(f"  samples     : {SAMPLE_ROOT or 'not found'}")
    print(f"  stylesheet  : {css}{'' if css.is_file() else '  (MISSING - run the tailwind build)'}")
    print(f"  serving     : http://{args.host}:{args.port}")

    if not _port_available(args.host, args.port):
        sys.exit(_explain_port_conflict(args.host, args.port))

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
