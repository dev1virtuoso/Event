import os
import sys
import base64
import shutil
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

import requests

app = Flask(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "clef-flash"
FRAME_INTERVAL_SECONDS = 2.5

# --- Upload limits -------------------------------------------------------
MAX_UPLOAD_BYTES = 512 * 1024 * 1024      # 512 MB (videos are big)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Frames pulled from an uploaded video, and how many get a model call.
MAX_FRAMES_EXTRACTED = 120
MAX_FRAMES_ANALYZED = 5
# Longest edge for extracted frames; keeps base64 payloads web-friendly.
FRAME_MAX_EDGE = 1280

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".flv",
    ".mpeg", ".mpg", ".3gp", ".mts", ".m2ts", ".ogv",
}

DEFAULT_QUESTIONS = {
    "scene_category": {
        "type": "choice",
        "instructions": "Classify the primary visual category of this image based on its filename, layout, background brightness, edge complexity, and color profile.",
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
    "contains_people": {
        "type": "noul",
        "instructions": "Does this image contain any human beings (people, faces, human figures)?",
    },
    "contains_nature": {
        "type": "noul",
        "instructions": "Does this image contain natural scenery or elements such as trees, mountains, sky, water, plants, animals, or landscapes (excluding purely indoor/urban scenes)?",
    },
    "contains_game": {
        "type": "noul",
        "instructions": "Does this image show a video game, game UI, gameplay screenshot, or game-related content?",
    },
    # 新增：自訂目標偵測
    "target_object_present": {
        "type": "noul",
        "instructions": "Does this image contain the specific target object requested by the user?",
    },
    "recommended_action": {
        "type": "choice",
        "instructions": "Select the best automated pipeline action for this image.",
        "criteria": {
            "parse_chart_metrics": "Extract data series, labels, and metrics from chart/plot",
            "extract_text_ocr": "Run text extraction to read text, code, or document body",
            "analyze_ui_state": "Inspect interface layout or debug terminal/UI state",
            "archive_media": "Tag and index in visual asset library",
        },
    },
}

def encode_image_base64(image_path):
    """Base64 of an image, used only for returning previews to the browser."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_clef_flash_decision(image_path, state_text):
    """Run one clef-flash decision.

    clef-flash is decision-native: it classifies from the structured `state`
    text, so the image is not sent to the model. Passing `images` still returns
    200 but costs extra work for no change in accuracy, so it is omitted.
    """
    url = f"{OLLAMA_BASE_URL}/v1/systemone"
    payload = {
        "model": MODEL_NAME,
        "state": state_text,
        "questions": DEFAULT_QUESTIONS,
    }
    try:
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Ollama API 錯誤: {str(e)}")


def describe_state(filename, width=None, height=None, duration_s=None,
                   frame_index=None, timestamp=None):
    """Build the `state` text that the decision model classifies on."""
    if frame_index is not None and timestamp is not None:
        parts = [f"Video frame at {timestamp:.1f}s (frame {frame_index})."]
        if width and height:
            parts.append(f"Resolution {width}x{height}.")
        if duration_s:
            parts.append(f"Source video duration {duration_s:.1f}s.")
        parts.append(f"Source file: {filename}.")
        return " ".join(parts)
    bits = [f"Image file: {filename}."]
    if width and height:
        bits.append(f"Resolution {width}x{height}.")
    return " ".join(bits)


def _ffmpeg_reason(stderr):
    """Pull the one useful line out of ffmpeg's stderr noise."""
    noise = ("libavformat", "libavcodec", "libavutil", "libavdevice",
             "libavfilter", "libswscale", "libswresample", "built with",
             "configuration:", "Press [q]")
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    useful = [ln for ln in lines
              if not any(ln.startswith(n) for n in noise)
              and not ln.startswith("lib")]
    return " | ".join(useful[-2:])[:300] or "ffmpeg failed"


def probe_image(path):
    """Return (width, height) for an image, or (None, None) if unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        txt = (out.stdout or "").strip().splitlines()
        if txt:
            w, h = txt[0].split("x")[:2]
            return int(w), int(h)
    except Exception:
        pass
    return None, None


def sniff_kind(path):
    """Classify a file by content, for uploads with a missing/unknown extension.

    ffprobe decodes a still image as a one-frame video stream, so the demuxer
    name is the reliable signal: images come from `*_pipe` demuxers (png_pipe,
    jpeg_pipe, ...) while clips come from real containers with a duration.
    """
    meta = probe_video(path)
    fmt = (meta["format"] or "").lower()
    still_formats = {"png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe",
                     "tiff_pipe", "gif", "image2"}
    if fmt in still_formats or fmt.endswith("_pipe"):
        return "image"
    if meta["duration"] and meta["duration"] > 0:
        return "video"
    if meta["codec"] and meta["codec"] not in {
            "png", "mjpeg", "bmp", "gif", "webp", "tiff"}:
        return "video"
    if meta["width"] and meta["height"]:
        return "image"
    return "unknown"


def probe_video(path):
    """Return metadata about a video file."""
    info = {"duration": None, "width": None, "height": None,
            "fps": None, "codec": None, "format": None}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=width,height,avg_frame_rate,codec_name:format=duration,format_name",
             "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return info

    for line in (out.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key in ("width", "height") and val.isdigit():
            info[key] = int(val)
        elif key == "codec_name":
            info["codec"] = val
        elif key == "format_name":
            info["format"] = val
        elif key == "duration":
            try:
                info["duration"] = float(val)
            except ValueError:
                pass
        elif key == "avg_frame_rate" and "/" in val:
            num, _, den = val.partition("/")
            try:
                if float(den):
                    info["fps"] = round(float(num) / float(den), 2)
            except ValueError:
                pass
    return info


def extract_frames(video_path, output_dir,
                   interval=FRAME_INTERVAL_SECONDS,
                   max_frames=MAX_FRAMES_EXTRACTED):
    """Pull one JPEG frame every `interval` seconds from a video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / max(0.1, interval)
    output_pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-nostdin", "-i", str(video_path),
        "-vf", f"fps={fps},scale='min({FRAME_MAX_EDGE},iw)':-2",
        "-frames:v", str(max_frames),
        "-q:v", "2", "-y", output_pattern,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg 抽幀超時（超過 600 秒）")
    except FileNotFoundError:
        raise RuntimeError("找不到 ffmpeg，請先安裝（macOS: brew install ffmpeg）")
    if result.returncode != 0:
        raise MediaDecodeError(
            f"無法解碼這個影片（可能已損壞或格式不支援）：{_ffmpeg_reason(result.stderr)}")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise MediaDecodeError("沒有抽取到任何幀（影片可能太短或格式不支援）")
    return frames


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.errorhandler(413)
def too_large(_err):
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({
        "error": f"檔案太大，超過上限 {limit_mb} MB。請改用較短的影片片段。"
    }), 413


def _cleanup_upload(upload):
    """Remove the temp dir backing an upload, if it is one of ours."""
    if not upload:
        return
    tmp = upload.get("temp_dir")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


class UnsupportedFile(Exception):
    """Client sent something we cannot decode as image or video."""


class MediaDecodeError(Exception):
    """Client sent media we cannot read (empty, corrupt, or zero frames)."""


def _analyze_image_upload(upload, filename):
    """Decide on a single still image."""
    path = upload["path"]
    width, height = probe_image(path)
    state = describe_state(filename, width, height)
    decision = call_clef_flash_decision(str(path), state)
    return {
        "status": "success",
        "kind": "image",
        "filename": filename,
        "visual_state": {
            "width": width,
            "height": height,
            "state_text": state,
        },
        "image_b64": encode_image_base64(path),
        "decision": decision,
    }


def _analyze_video_upload(upload, filename):
    """Sample frames from a video and decide on the first few."""
    path = upload["path"]
    meta = probe_video(path)
    frames_dir = Path(upload["temp_dir"]) / "frames"
    frames = extract_frames(path, frames_dir)

    chosen = frames[:MAX_FRAMES_ANALYZED]
    results = []
    errors = []
    for idx, frame_file in enumerate(chosen):
        ts = idx * FRAME_INTERVAL_SECONDS
        state = describe_state(
            filename,
            width=meta["width"], height=meta["height"],
            duration_s=meta["duration"],
            frame_index=idx + 1, timestamp=ts,
        )
        try:
            decision = call_clef_flash_decision(str(frame_file), state)
            frame_error = None
        except Exception as exc:      # keep the frames we already got
            decision, frame_error = None, str(exc)
            errors.append(f"frame {idx + 1}: {exc}")

        results.append({
            "frame_index": idx + 1,
            "timestamp": ts,
            "name": filename,
            "image_b64": encode_image_base64(frame_file),
            "decision": decision,
            "error": frame_error,
        })

    payload = {
        "status": "success" if any(r["decision"] for r in results) else "error",
        "kind": "video",
        "filename": filename,
        "video": meta,
        "frames_extracted": len(frames),
        "frames_analyzed": len(results),
        "results": results,
    }
    if errors and not payload["status"] == "success":
        payload["error"] = "; ".join(errors[:3])
    elif errors:
        payload["warnings"] = errors[:3]
    return payload


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Single endpoint for still images and video clips.

    Accepts a multipart upload in the `file` field (or `image`, for backward
    compatibility). The kind is detected from the extension, falling back to
    ffprobe when the extension is unknown or missing.
    """
    file = request.files.get("file") or request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "No file uploaded. Use the 'file' field."}), 400

    filename = os.path.basename(file.filename)
    ext = Path(filename).suffix.lower()

    temp_dir = Path(tempfile.mkdtemp(prefix="web_clef_"))
    path = temp_dir / (filename or "upload.bin")
    try:
        file.save(path)
        if path.stat().st_size == 0:
            raise MediaDecodeError("上傳的檔案是空的")

        upload = {"path": path, "temp_dir": temp_dir}

        if ext in IMAGE_EXTENSIONS:
            return jsonify(_analyze_image_upload(upload, filename))
        if ext in VIDEO_EXTENSIONS:
            return jsonify(_analyze_video_upload(upload, filename))

        # Unknown or missing extension: classify by content.
        kind = sniff_kind(path)
        if kind == "image":
            return jsonify(_analyze_image_upload(upload, filename))
        if kind == "video":
            return jsonify(_analyze_video_upload(upload, filename))

        supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
        raise UnsupportedFile(
            f"無法辨識的檔案類型（{ext or '無副檔名'}）。支援: {supported}")

    except (UnsupportedFile, MediaDecodeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("analyze failed")
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup_upload({"temp_dir": temp_dir})


@app.route("/api/analyze_image", methods=["POST"])
def api_analyze_image():
    """Backward-compatible alias for still images."""
    return api_analyze()


@app.route("/api/analyze_video", methods=["POST"])
def api_analyze_video():
    """Analyze a local path or an http(s) URL.

    Accepts JSON `{"url": ...}`, or a direct multipart video upload.
    """
    upload = request.files.get("file") or request.files.get("video")
    if upload is not None and upload.filename:
        filename = os.path.basename(upload.filename)
        temp_dir = Path(tempfile.mkdtemp(prefix="web_clef_"))
        path = temp_dir / filename
        try:
            upload.save(path)
            if path.stat().st_size == 0:
                raise MediaDecodeError("上傳的檔案是空的")
            return jsonify(_analyze_video_upload(
                {"path": path, "temp_dir": temp_dir}, filename))
        except (UnsupportedFile, MediaDecodeError) as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.exception("video upload failed")
            return jsonify({"error": str(e)}), 500
        finally:
            _cleanup_upload({"temp_dir": temp_dir})

    data = request.get_json(silent=True) or {}
    url_or_path = (data.get("url") or "").strip()
    if not url_or_path:
        return jsonify({
            "error": "Empty URL/Path. Send JSON {'url': ...} or upload a file."
        }), 400

    temp_dir = Path(tempfile.mkdtemp(prefix="web_clef_"))
    frames_dir = temp_dir / "frames"
    try:
        video_path = url_or_path
        if url_or_path.startswith(("http://", "https://")):
            out_tpl = str(temp_dir / "video.%(ext)s")
            try:
                res = subprocess.run(
                    [
                        "yt-dlp",
                        "--no-playlist",
                        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
                        "--merge-output-format", "mp4",
                        "-o", out_tpl,
                        url_or_path,
                    ],
                    capture_output=True, text=True, timeout=120,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("yt-dlp 下載超時（超過 120 秒）")
            except FileNotFoundError:
                raise RuntimeError("找不到 yt-dlp，請先安裝或改用檔案上傳")

            if res.returncode != 0:
                raise RuntimeError(f"yt-dlp 下載失敗：{res.stderr[-300:]}")

            videos = list(temp_dir.glob("video.*"))
            if not videos:
                raise RuntimeError("yt-dlp 下載完成但找不到檔案")
            video_path = str(videos[0])
        elif not Path(video_path).is_file():
            return jsonify({"error": f"找不到本地檔案: {video_path}"}), 400

        return jsonify(_analyze_video_upload(
            {"path": Path(video_path), "temp_dir": temp_dir},
            Path(video_path).name))
    except MediaDecodeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("analyze video failed")
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup_upload({"temp_dir": temp_dir})


@app.route("/api/health")
def api_health():
    """Report which optional tools this machine has available."""
    def has(binary):
        return shutil.which(binary) is not None

    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "ollama": OLLAMA_BASE_URL,
        "ffmpeg": has("ffmpeg"),
        "ffprobe": has("ffprobe"),
        "yt_dlp": has("yt-dlp"),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "video_extensions": sorted(VIDEO_EXTENSIONS),
    })


# ============================================================
# HTML/CSS/JS 前端模板
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clef-Flash Vision Decision Studio</title>
    <style>
        :root {
            --bg-color: #0f0f10;
            --panel-bg: #18181b;
            --panel-border: #27272a;
            --text-main: #e4e4e7;
            --text-muted: #a1a1aa;
            --accent: #4ade80;
            --bar-bg: #27272a;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .header {
            width: 100%;
            max-width: 1200px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        .title h1 {
            color: var(--accent);
            margin: 0 0 5px 0;
            font-size: 24px;
            font-weight: 600;
        }
        .title p {
            color: var(--text-muted);
            margin: 0;
            font-size: 14px;
        }
        .status-badge {
            background-color: #1f2937;
            color: var(--accent);
            padding: 4px 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 12px;
            border: 1px solid var(--accent);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .main-container {
            display: flex;
            gap: 20px;
            width: 100%;
            max-width: 1200px;
            align-items: flex-start;
        }
        .left-col {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
            min-width: 350px;
        }
        .right-col {
            flex: 1.5;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .panel {
            background-color: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 20px;
            position: relative;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 15px;
        }
        .upload-area {
            border: 1px dashed var(--panel-border);
            border-radius: 6px;
            height: 300px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            overflow: hidden;
            margin-bottom: 15px;
            background-color: #121214;
            position: relative;
        }
        .upload-area img,
        .upload-area video {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        .upload-text {
            color: var(--text-muted);
            font-size: 14px;
        }
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .input-group input {
            flex: 1;
            background-color: #121214;
            border: 1px solid var(--panel-border);
            color: white;
            padding: 10px;
            border-radius: 6px;
            outline: none;
            font-family: monospace;
        }
        .input-group input:focus {
            border-color: var(--accent);
        }
        button {
            background-color: #27272a;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
            white-space: nowrap;
        }
        button:hover { background-color: #3f3f46; }
        button.primary { background-color: var(--accent); color: #000; }
        button.primary:hover { background-color: #22c55e; }
        button.camera { background-color: #3b82f6; color: #fff; }
        button.camera:hover { background-color: #2563eb; }
        
        .decision-block { margin-bottom: 25px; }
        .decision-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            font-family: monospace;
        }
        .tag {
            background-color: #3f3f46;
            color: #d4d4d8;
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        .tag.target { background-color: #ea580c; color: white; }
        .decision-title {
            font-size: 14px;
            font-weight: 600;
        }
        .metric-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 13px;
            font-family: monospace;
        }
        .metric-label {
            width: 250px;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-bar-bg {
            flex: 1;
            height: 6px;
            background-color: var(--bar-bg);
            border-radius: 3px;
            margin: 0 15px;
            position: relative;
        }
        .metric-bar-fill {
            height: 100%;
            background-color: var(--accent);
            border-radius: 3px;
        }
        .metric-value {
            width: 60px;
            text-align: right;
            color: var(--accent);
        }
        .noul-row {
            font-family: monospace;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .noul-yes { color: var(--accent); }
        .noul-no { color: #ef4444; }
        .raw-response {
            font-family: monospace;
            font-size: 12px;
            color: var(--text-muted);
            white-space: pre-wrap;
            background: #121214;
            padding: 15px;
            border-radius: 6px;
            max-height: 300px;
            overflow-y: auto;
        }
        .hidden { display: none !important; }
        
        .playback-controls {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 15px;
            font-family: monospace;
            font-size: 13px;
            color: var(--accent);
        }
        .timeline-bar {
            flex: 1;
            height: 4px;
            background: #27272a;
            border-radius: 2px;
            overflow: hidden;
            position: relative;
        }
        .timeline-fill {
            height: 100%;
            background: var(--accent);
            width: 0%;
            transition: width 0.3s linear;
        }
    </style>
</head>
<body>

    <div class="header">
        <div class="title">
            <h1>Clef-Flash Vision Decision Studio</h1>
            <p>Decision-Native / Jev · System One API · 10-category classification</p>
        </div>
        <div class="status-badge">
            <span>model: clef-flash</span>
            <span>clef-flash ready</span>
        </div>
    </div>

    <div class="main-container">
        <div class="left-col">
            <div class="panel">
                <div class="panel-header">
                    <span>MEDIA INPUT</span>
                    <span id="input-filename">No file selected</span>
                </div>
                <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
                    <span class="upload-text" id="upload-text">Click to choose an image or video clip</span>
                    <img id="preview-img" class="hidden" src="" alt="Preview">
                    <video id="preview-video" class="hidden" src="" muted playsinline controls></video>
                </div>
                <input type="file" id="file-input" class="hidden" accept="image/*,video/*,.mp4,.mov,.m4v,.webm,.avi,.mkv,.mpeg,.mpg,.3gp,.mts,.m2ts" onchange="handleFileUpload(event)">
                <p style="color: var(--text-muted); font-size: 11px; margin: 0 0 12px 0;">
                    Images: png · jpg · webp · bmp · gif &nbsp;|&nbsp; Video: mp4 · mov · m4v · webm · avi · mkv · mpeg (up to <span id="max-upload">512</span> MB)
                </p>
                
                <!-- 鏡頭控制 -->
                <div class="input-group" style="margin-top: 10px;">
                    <button onclick="startCamera()" id="btn-camera" class="camera">Start Camera</button>
                    <button onclick="captureCameraFrame()" id="btn-capture" class="primary" style="display: none;">Capture & Analyze</button>
                </div>

                <!-- 影片輸入與目標設定 -->
                <div class="input-group" style="margin-top: 10px;">
                    <input type="text" id="target-object-input" placeholder="Target object (e.g. cat, person, chart)...">
                </div>
                <div class="input-group">
                    <input type="text" id="url-input" placeholder="Enter YouTube URL or local video path...">
                    <button onclick="pasteClipboard()">Paste</button>
                    <button class="primary" onclick="analyzeVideo()">Analyze Video</button>
                </div>

                <div id="playback-controls" class="playback-controls hidden">
                    <span id="playback-status">Playing Frame 1/1</span>
                    <div class="timeline-bar">
                        <div id="timeline-fill" class="timeline-fill"></div>
                    </div>
                    <button id="toggle-view" onclick="toggleView()" style="padding: 4px 10px; font-size: 11px;">Show video</button>
                </div>
            </div>
            
            <div class="panel" id="visual-state-panel">
                <div class="panel-header">
                    <span>MEDIA STATE</span>
                </div>
                <div class="metric-row">
                    <div class="metric-label">resolution</div>
                    <div class="metric-value" id="vs-res">-</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">duration</div>
                    <div class="metric-value" id="vs-dur">-</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">codec / fps</div>
                    <div class="metric-value" id="vs-codec">-</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">frames</div>
                    <div class="metric-value" id="vs-frames">-</div>
                </div>
            </div>
        </div>

        <div class="right-col">
            <div class="panel" id="decision-output">
                <div class="panel-header">
                    <span>DECISION OUTPUT</span>
                    <span id="exec-time">- ms</span>
                </div>
                <div id="decision-content">
                    <p style="color: var(--text-muted); text-align: center; margin-top: 50px;">Waiting for input...</p>
                </div>
            </div>

            <div class="panel" id="raw-response-panel">
                <div class="panel-header">
                    <span>RAW RESPONSE</span>
                </div>
                <div class="raw-response" id="raw-json">Waiting...</div>
            </div>
        </div>
    </div>

    <script>
        let videoFramesData = [];
        let playbackInterval = null;
        let currentFrameIdx = 0;
        let currentObjectUrl = null;
        let lastReqMs = 0;
        let sourceName = "";
        let showingVideo = false;

        const IMAGE_EXT = ['png','jpg','jpeg','webp','bmp','gif','tif','tiff'];
        const VIDEO_EXT = ['mp4','mov','m4v','avi','mkv','webm','wmv','flv','mpeg','mpg','3gp','mts','m2ts','ogv'];

        function fileKind(file) {
            const ext = (file.name.split('.').pop() || '').toLowerCase();
            if (VIDEO_EXT.includes(ext)) return 'video';
            if (IMAGE_EXT.includes(ext)) return 'image';
            if (file.type.startsWith('video/')) return 'video';
            if (file.type.startsWith('image/')) return 'image';
            return 'unknown';
        }

        function resetPreview() {
            document.getElementById('preview-img').classList.add('hidden');
            const v = document.getElementById('preview-video');
            v.pause();
            v.removeAttribute('src');
            v.load();
            v.classList.add('hidden');
            document.getElementById('upload-text').classList.remove('hidden');
            if (currentObjectUrl) {
                URL.revokeObjectURL(currentObjectUrl);
                currentObjectUrl = null;
            }
            showingVideo = false;
            const btn = document.getElementById('toggle-view');
            if (btn) btn.innerText = 'Show video';
        }

        function toggleView() {
            if (!currentObjectUrl) return;
            showingVideo = !showingVideo;
            const img = document.getElementById('preview-img');
            const vid = document.getElementById('preview-video');
            if (showingVideo) {
                if (playbackInterval) clearInterval(playbackInterval);
                img.classList.add('hidden');
                vid.classList.remove('hidden');
                const f = videoFramesData[currentFrameIdx];
                if (f && vid.duration) vid.currentTime = Math.min(f.timestamp, vid.duration - 0.05);
                document.getElementById('toggle-view').innerText = 'Show frames';
            } else {
                vid.pause();
                vid.classList.add('hidden');
                img.classList.remove('hidden');
                document.getElementById('toggle-view').innerText = 'Show video';
                showFrame(currentFrameIdx);
            }
        }

        function handleFileUpload(event) {
            stopPlayback();
            stopCamera();
            const file = event.target.files[0];
            if (!file) return;

            resetPreview();
            document.getElementById('input-filename').innerText = file.name;
            setMediaState({});          // clear stale numbers from the last file

            const kind = fileKind(file);
            sourceName = file.name;
            currentObjectUrl = URL.createObjectURL(file);

            if (kind === 'video') {
                const v = document.getElementById('preview-video');
                v.src = currentObjectUrl;
                v.classList.remove('hidden');
                document.getElementById('upload-text').classList.add('hidden');
            } else {
                const img = document.getElementById('preview-img');
                img.src = currentObjectUrl;
                img.classList.remove('hidden');
                document.getElementById('upload-text').classList.add('hidden');
            }

            const formData = new FormData();
            formData.append('file', file);

            setLoading(true, kind);
            const t0 = performance.now();
            fetch('/api/analyze', { method: 'POST', body: formData })
            .then(res => res.json().then(body => ({ ok: res.ok, body })))
            .then(({ ok, body }) => {
                if (body.error) throw new Error(body.error);
                if (!ok) throw new Error('HTTP error');
                renderPayload(body, performance.now() - t0);
            })
            .catch(err => {
                document.getElementById('decision-content').innerHTML =
                    `<p style="color:#ef4444;text-align:center;margin-top:50px;">${err.message}</p>`;
            })
            .finally(() => setLoading(false));
        }

        function analyzeVideo() {
            stopPlayback();
            stopCamera();
            const url = document.getElementById('url-input').value.trim();
            const targetObj = document.getElementById('target-object-input').value.trim();
            if (!url) return alert("Please enter a URL or path");

            resetPreview();
            setMediaState({});
            setLoading(true, 'video');

            fetch('/api/analyze_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url, target_object: targetObj })
            })
            .then(res => res.json().then(body => ({ ok: res.ok, body })))
            .then(({ body }) => {
                if (body.error) throw new Error(body.error);
                renderPayload(body);
            })
            .catch(err => {
                document.getElementById('decision-content').innerHTML =
                    `<p style="color:#ef4444;text-align:center;margin-top:50px;">${err.message}</p>`;
            })
            .finally(() => setLoading(false));
        }

        // One shape for both kinds: {kind, decision, image_b64, ...}
        function renderPayload(payload, reqMs) {
            if (reqMs) lastReqMs = reqMs;
            if (payload.kind === 'video') {
                if (!payload.results || payload.results.length === 0) {
                    document.getElementById('decision-content').innerHTML =
                        '<p style="color:var(--text-muted);text-align:center;margin-top:50px;">No frames analyzed.</p>';
                    return;
                }
                videoFramesData = payload.results;
                currentFrameIdx = 0;

                const v = payload.video || {};
                setMediaState({
                    res: (v.width && v.height) ? `${v.width}x${v.height}` : '-',
                    dur: v.duration ? `${v.duration.toFixed(1)}s` : '-',
                    codec: [v.codec, v.fps ? v.fps + ' fps' : null].filter(Boolean).join(' / ') || '-',
                    frames: `${payload.frames_analyzed} of ${payload.frames_extracted}`,
                });

                document.getElementById('playback-controls').classList.remove('hidden');
                showFrame(0, reqMs);
                startPlayback();
            } else {
                const vs = payload.visual_state || {};
                setMediaState({
                    res: (vs.width && vs.height) ? `${vs.width}x${vs.height}` : '-',
                    dur: '-',
                    codec: '-',
                    frames: '-',
                });
                // Image mode: no frame playback, so no video/frames toggle.
                document.getElementById('playback-controls').classList.add('hidden');
                if (payload.image_b64) {
                    const img = document.getElementById('preview-img');
                    img.src = "data:image/jpeg;base64," + payload.image_b64;
                    img.classList.remove('hidden');
                    document.getElementById('upload-text').classList.add('hidden');
                }
                renderResults(payload.decision, reqMs);
            }
        }

        function setMediaState(v) {
            document.getElementById('vs-res').innerText = v.res || '-';
            document.getElementById('vs-dur').innerText = v.dur || '-';
            document.getElementById('vs-codec').innerText = v.codec || '-';
            document.getElementById('vs-frames').innerText = v.frames || '-';
        }


        function showFrame(idx, reqMs) {
            if (idx >= videoFramesData.length) idx = 0;
            const frame = videoFramesData[idx];
            
            const img = document.getElementById('preview-img');
            const vid = document.getElementById('preview-video');
            vid.classList.add('hidden');
            vid.pause();
            img.src = "data:image/jpeg;base64," + frame.image_b64;
            img.classList.remove('hidden');
            document.getElementById('upload-text').classList.add('hidden');
            videoElement.style.display = 'none';
            
            document.getElementById('input-filename').innerText = 
                `${frame.name || sourceName || 'video'} · frame ${frame.frame_index} @ ${frame.timestamp.toFixed(1)}s`;
            
            document.getElementById('playback-status').innerText = 
                `Frame ${frame.frame_index}/${videoFramesData.length}`;
            const pct = ((idx + 1) / videoFramesData.length) * 100;
            document.getElementById('timeline-fill').style.width = pct + "%";

            // The "Show video" toggle only makes sense for a locally uploaded clip.
            const btn = document.getElementById('toggle-view');
            btn.style.display = currentObjectUrl ? '' : 'none';
            
            if (frame.error) {
                document.getElementById('decision-content').innerHTML =
                    `<p style="color:#ef4444;text-align:center;margin-top:50px;">${frame.error}</p>`;
                document.getElementById('raw-json').innerText = JSON.stringify(frame, null, 2);
            } else {
                renderResults(frame.decision, reqMs);
            }
        }

        function startPlayback() {
            if (playbackInterval) clearInterval(playbackInterval);
            playbackInterval = setInterval(() => {
                currentFrameIdx++;
                if (currentFrameIdx >= videoFramesData.length) {
                    currentFrameIdx = 0;
                }
                showFrame(currentFrameIdx);
            }, 1500);
        }

        function stopPlayback() {
            if (playbackInterval) {
                clearInterval(playbackInterval);
                playbackInterval = null;
            }
            document.getElementById('playback-controls').classList.add('hidden');
            videoFramesData = [];
            currentFrameIdx = 0;
        }

        function pasteClipboard() {
            navigator.clipboard.readText().then(text => {
                document.getElementById('url-input').value = text;
            });
        }

        function setLoading(isLoading, kind) {
            const content = document.getElementById('decision-content');
            if (isLoading) {
                const what = kind === 'video'
                    ? 'Extracting frames and analyzing with clef-flash...'
                    : 'Analyzing with clef-flash...';
                content.innerHTML = `<p style="color: var(--accent); text-align: center; margin-top: 50px;">${what}</p>`;
                document.getElementById('raw-json').innerText = 'Waiting...';
            }
        }

        function renderResults(decision, reqMs) {
            const container = document.getElementById('decision-content');
            const rawJson = document.getElementById('raw-json');
            const execTime = document.getElementById('exec-time');
            
            rawJson.innerText = JSON.stringify(decision, null, 2);
            execTime.innerText = reqMs ? Math.round(reqMs) + " ms" : (lastReqMs ? Math.round(lastReqMs) + " ms" : "-");
            
            const answers = decision.answers || decision;
            const targetInput = document.getElementById('target-object-input').value.trim();
            let html = '';

            // 優先顯示目標物件（如果有）
            if (targetInput && answers.target_object_present) {
                const ans = answers.target_object_present;
                const prob = ans.noul !== undefined ? ans.noul : 0;
                const isYes = prob >= 0.5;
                const cls = isYes ? 'noul-yes' : 'noul-no';
                const label = isYes ? 'FOUND' : 'NOT FOUND';
                html += `
                <div class="decision-block" style="border-left: 3px solid #ea580c; padding-left: 15px; margin-bottom: 25px;">
                    <div class="decision-header">
                        <span class="tag target">TARGET DETECTED</span>
                        <span class="decision-title" style="color: #ea580c;">"${targetInput}"</span>
                    </div>
                    <div class="noul-row">
                        <span class="${cls}" style="font-size: 18px; font-weight: bold;">${label}</span> &nbsp; P = ${prob.toFixed(4)}
                    </div>
                </div>`;
            }

            if (answers.information_density) {
                const idn = answers.information_density;
                html += `
                <div class="decision-block">
                    <div class="decision-header">
                        <span class="tag">SCORE</span>
                        <span class="decision-title">information_density</span>
                    </div>
                    <div class="metric-row">
                        <div class="metric-label">score: ${idn.score.toFixed(3)}</div>
                        <div class="metric-value">conf ${idn.confidence ? idn.confidence.toFixed(3) : 'N/A'}</div>
                    </div>
                `;
                if (idn.probabilities) {
                    for (const [key, val] of Object.entries(idn.probabilities)) {
                        const pct = (val * 100).toFixed(1);
                        const label = idn.legend && idn.legend[key] ? idn.legend[key] : `Category ${key}`;
                        html += `
                        <div class="metric-row">
                            <div class="metric-label">${label}</div>
                            <div class="metric-bar-bg"><div class="metric-bar-fill" style="width: ${pct}%"></div></div>
                            <div class="metric-value">${pct}%</div>
                        </div>`;
                    }
                }
                html += `</div>`;
            }

            const nouls = ['needs_ocr_or_parsing', 'contains_people', 'contains_nature', 'contains_game'];
            nouls.forEach(q => {
                if (answers[q]) {
                    const ans = answers[q];
                    const prob = ans.noul !== undefined ? ans.noul : 0;
                    const isYes = prob >= 0.5;
                    const cls = isYes ? 'noul-yes' : 'noul-no';
                    const label = isYes ? 'TRUE' : 'FALSE';
                    html += `
                    <div class="decision-block">
                        <div class="decision-header">
                            <span class="tag">NOUL</span>
                            <span class="decision-title">${q}</span>
                        </div>
                        <div class="noul-row">
                            <span class="${cls}">${label}</span> &nbsp; P = ${prob.toFixed(4)}
                        </div>
                    </div>`;
                }
            });

            const choices = ['scene_category', 'recommended_action'];
            choices.forEach(q => {
                if (answers[q]) {
                    const ans = answers[q];
                    html += `
                    <div class="decision-block">
                        <div class="decision-header">
                            <span class="tag">CHOICE</span>
                            <span class="decision-title">${q}</span>
                        </div>
                        <div class="metric-row">
                            <div class="metric-label">selected: ${ans.choice}</div>
                            <div class="metric-value">conf ${ans.confidence ? ans.confidence.toFixed(3) : 'N/A'}</div>
                        </div>
                    `;
                    if (ans.probabilities) {
                        const sorted = Object.entries(ans.probabilities).sort((a, b) => b[1] - a[1]);
                        for (const [key, val] of sorted) {
                            const pct = (val * 100).toFixed(1);
                            const isSelected = key === ans.choice;
                            const barColor = isSelected ? 'var(--accent)' : '#3f3f46';
                            html += `
                            <div class="metric-row">
                                <div class="metric-label" style="color: ${isSelected ? 'var(--text-main)' : 'var(--text-muted)'}">${key}</div>
                                <div class="metric-bar-bg"><div class="metric-bar-fill" style="width: ${pct}%; background-color: ${barColor}"></div></div>
                                <div class="metric-value" style="color: ${isSelected ? 'var(--accent)' : 'var(--text-muted)'}">${pct}%</div>
                            </div>`;
                        }
                    }
                    html += `</div>`;
                }
            });

            container.innerHTML = html;
        }

        // Show the real server-side upload limit and flag missing ffmpeg early.
        fetch('/api/health').then(r => r.json()).then(h => {
            if (h.max_upload_mb) {
                document.getElementById('max-upload').innerText = h.max_upload_mb;
            }
            if (!h.ffmpeg || !h.ffprobe) {
                const badge = document.querySelector('.status-badge');
                badge.innerHTML += '<span style="color:#ef4444;">ffmpeg missing</span>';
            }
        }).catch(() => {});
    </script>
</body>
</html>
"""

def _port_is_free(host, port):
    """True when nothing else is bound to host:port.

    SO_REUSEADDR is deliberately NOT set: on macOS it allows binding a specific
    address even while a wildcard listener (e.g. AirPlay's ControlCenter on
    *:5000) already holds the port, which would silently shadow this server.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host, preferred, span=20):
    """Return `preferred` if free, else the next free port in the span."""
    for port in range(preferred, preferred + span):
        if _port_is_free(host, port):
            return port
    return preferred


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clef-Flash Vision Decision Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)),
                        help="Preferred port; a free one is chosen if busy")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug reloader")
    args = parser.parse_args()

    host = args.host
    port = args.port
    if not _port_is_free(host, port):
        fallback = _pick_port(host, port + 1)
        print(f"⚠️  連接埠 {port} 已被占用。")
        print("   macOS 上常見原因是 AirPlay Receiver（ControlCenter）。")
        print(f"   改用 {fallback}，或關閉 AirPlay：系統設定 → 一般 → AirDrop 與 Handoff。")
        port = fallback

    print(f"啟動 Web UI 於 http://{host}:{port}")
    app.run(host=host, port=port, debug=args.debug)
