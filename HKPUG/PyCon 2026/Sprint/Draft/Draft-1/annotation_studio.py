"""Clef Vision Annotation Studio.

A video-first sibling of `app.py`. It samples frames from a clip, sends each
frame to the decision-native `clef-flash` model, and produces structured
annotation tags: who and what is visible, whether the frame is a screen
recording, whether it looks like a livestream layout, and so on.

Two properties drive the design:

1. Questions live in `questions.yaml`, not in code. Edit that file to change
   what the model is asked, then hit "Reload questions" in the UI.
2. Frames are actually sent to the model as images. `clef-flash` is
   decision-native, but its visual questions only work with image input: with
   no image the same "are there people?" question returns an identical prior
   probability for every frame regardless of content. An annotation tool that
   never looks at the pixels would be inventing its answers.

Run:
    python annotation_studio.py
    python annotation_studio.py --port 8000 --questions ./questions.yaml
"""

import argparse
import base64
import copy
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import requests
import yaml
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "clef-flash"

# --- Paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_PATH = BASE_DIR / "questions.yaml"

# --- Media handling ------------------------------------------------------
MAX_UPLOAD_BYTES = 512 * 1024 * 1024          # 512 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

FRAME_INTERVAL_SECONDS = 2.5
MAX_FRAMES_EXTRACTED = 120
MAX_FRAMES_ANALYZED = 5
FRAME_MAX_EDGE = 1024                         # keeps base64 payloads sane

NOUL_THRESHOLD = 0.5

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".flv",
    ".mpeg", ".mpg", ".3gp", ".mts", ".m2ts", ".ogv",
}

# Tag buckets used for UI colouring; unknown tags fall back to a neutral colour.
TAG_BUCKETS = ("scene", "people", "vehicles", "screen", "motion",
               "text", "environment", "pipeline")

_questions_lock = threading.Lock()
_questions_path = DEFAULT_QUESTIONS_PATH
_questions_spec = None


# ===========================================================================
# Question loading (YAML)
# ===========================================================================

class QuestionSpecError(Exception):
    """The YAML file is malformed or uses an unsupported schema."""


def load_questions(path):
    """Parse and validate the question YAML."""
    path = Path(path)
    if not path.is_file():
        raise QuestionSpecError(f"找不到題目檔: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise QuestionSpecError(f"YAML 解析失敗: {exc}") from exc

    if not isinstance(data, dict):
        raise QuestionSpecError("YAML 最外層必須是 mapping")
    questions = data.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise QuestionSpecError("YAML 必須包含非空的 `questions:` mapping")

    valid_types = {"noul", "choice", "score"}
    cleaned = {}
    for key, spec in questions.items():
        if not isinstance(spec, dict):
            raise QuestionSpecError(f"題目 `{key}` 必須是 mapping")
        qtype = spec.get("type")
        if qtype not in valid_types:
            raise QuestionSpecError(
                f"題目 `{key}` 的 type 必須是 {sorted(valid_types)}，收到 {qtype!r}")
        entry = {
            "type": qtype,
            "label": spec.get("label", key),
            "instructions": spec.get("instructions", ""),
            "tag": spec.get("tag", "general"),
        }
        if qtype == "choice":
            criteria = spec.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise QuestionSpecError(
                    f"choice 題目 `{key}` 需要非空的 `criteria` mapping")
            entry["criteria"] = {str(k): str(v) for k, v in criteria.items()}
        else:
            criteria = spec.get("criteria")
            if isinstance(criteria, list):
                entry["criteria"] = [str(c) for c in criteria]
            elif isinstance(criteria, dict):
                entry["criteria"] = [str(v) for v in criteria.values()]
        if isinstance(spec.get("legend"), dict):
            entry["legend"] = spec["legend"]
        if isinstance(spec.get("when"), dict):
            rule = spec["when"]
            if "question" not in rule:
                raise QuestionSpecError(f"題目 `{key}` 的 when 需要 `question`")
            entry["when"] = rule
        cleaned[key] = entry

    # Validate that `when` rules only reference questions defined here.
    for key, spec in cleaned.items():
        rule = spec.get("when")
        if rule and rule["question"] not in cleaned:
            raise QuestionSpecError(
                f"題目 `{key}` 的 when 指向未定義的題目 `{rule['question']}`")

    return {
        "version": data.get("version"),
        "name": data.get("name", path.stem),
        "description": (data.get("description") or "").strip(),
        "context": (data.get("context") or "").strip(),
        "questions": cleaned,
        "source_path": str(path),
        "raw": raw_text,
    }


def get_questions():
    """Return the loaded spec, loading it on first use."""
    global _questions_spec
    with _questions_lock:
        if _questions_spec is None:
            _questions_spec = load_questions(_questions_path)
        return _questions_spec


def reload_questions():
    global _questions_spec
    with _questions_lock:
        _questions_spec = load_questions(_questions_path)
        return _questions_spec


def _model_question(spec):
    """Convert one YAML question into the System One API shape."""
    out = {"type": spec["type"], "instructions": spec["instructions"]}
    if spec["type"] == "choice":
        out["criteria"] = spec["criteria"]
    elif spec["type"] == "score" and spec.get("criteria"):
        out["criteria"] = spec["criteria"]
    return out


# ===========================================================================
# Conditional question resolution
# ===========================================================================

def _rule_matches(rule, answers):
    """True when a `when` rule is satisfied by the answers so far."""
    if not rule:
        return True
    parent = rule["question"]
    ans = answers.get(parent)
    if not ans:
        return False

    unless = rule.get("unless_choice") or {}
    if parent in unless and ans.get("choice") in unless[parent]:
        return False

    if "is" in rule:
        return ans.get("choice") == rule["is"]
    if "is_not" in rule:
        return ans.get("choice") != rule["is"]

    for op, cmp in (("at_least", lambda a, b: a >= b),
                    ("at_most", lambda a, b: a <= b)):
        if op in rule:
            val = ans.get("noul")
            if val is None:
                val = ans.get("score")
            if val is None:
                return False
            return cmp(float(val), float(rule[op]))
    return True


def question_waves(spec):
    """Group questions by dependency depth.

    Wave 0 holds questions with no `when` rule. Wave N holds questions whose
    rule depends on something answered in an earlier wave, so each wave can be
    sent as one model call once the previous wave's answers are known.
    """
    questions = spec["questions"]
    depth = {}

    def resolve(key, stack=()):
        if key in depth:
            return depth[key]
        if key in stack:                      # cycle guard
            depth[key] = 0
            return 0
        rule = questions[key].get("when")
        if not rule:
            depth[key] = 0
            return 0
        parent = rule["question"]
        if parent not in questions:
            depth[key] = 0
            return 0
        value = resolve(parent, stack + (key,)) + 1
        depth[key] = value
        return value

    for key in questions:
        resolve(key)

    waves = {}
    for key, level in depth.items():
        waves.setdefault(level, []).append(key)
    return [waves[level] for level in sorted(waves)]


def annotate_frame(image_path, state_text, spec, timeout=300):
    """Annotate one frame, asking conditional questions only when they apply.

    Returns (answers, skipped_keys, usage_totals).
    """
    answers = {}
    skipped = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    waves = question_waves(spec)

    for wave_index, keys in enumerate(waves):
        asked = []
        for key in keys:
            rule = spec["questions"][key].get("when")
            if _rule_matches(rule, answers):
                asked.append(key)
            else:
                skipped.append(key)
        if not asked:
            continue

        payload_questions = {
            key: _model_question(spec["questions"][key]) for key in asked
        }
        payload = {
            "model": MODEL_NAME,
            "state": state_text,
            "images": [encode_image_base64(image_path)],
            "questions": payload_questions,
        }
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/v1/systemone",
                json=payload, timeout=timeout)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Ollama API 錯誤: {exc}") from exc

        body = response.json()
        for key, value in (body.get("answers") or {}).items():
            if key in payload_questions:
                answers[key] = value
        for field in usage:
            usage[field] += int((body.get("usage") or {}).get(field) or 0)

        if wave_index == 0 and not body.get("answers"):
            raise RuntimeError("模型沒有回傳任何答案")

    return answers, skipped, usage


# ===========================================================================
# Answer -> tags
# ===========================================================================

def score_level(score, criteria, legend=None, probabilities=None):
    """Resolve a score answer to (level_name, confidence).

    `score` is a continuous value whose scale is not fixed: the model returns
    values across 0..N where N is the number of criteria, so a raw
    `int(score)` is not a safe index. The model also returns a discrete
    `probabilities` map keyed to `legend` entries, which is unambiguous, so
    that is preferred and the score is only a fallback.
    """
    if probabilities and legend:
        try:
            best_key = max(probabilities, key=lambda key: probabilities[key])
            level = legend.get(str(best_key))
            if level is not None:
                return level, float(probabilities[best_key])
        except (TypeError, ValueError):
            pass

    if not criteria:
        return None, None
    # Fallback: criteria are levels 0..len-1; clamp the continuous score.
    index = int(round(score))
    index = max(0, min(index, len(criteria) - 1))
    return criteria[index], None


def answer_to_tags(answers, spec, skipped=()):
    """Flatten model answers into displayable annotation tags."""
    tags = []
    for key, qspec in spec["questions"].items():
        ans = answers.get(key)
        if not ans:
            continue
        qtype = qspec["type"]
        label = qspec["label"]
        bucket = qspec.get("tag", "general")

        if qtype == "noul":
            prob = float(ans.get("noul") or 0.0)
            tags.append({
                "key": key, "label": label, "bucket": bucket,
                "kind": "noul", "text": label,
                "value": prob >= NOUL_THRESHOLD, "prob": round(prob, 4),
            })
        elif qtype == "choice":
            choice = ans.get("choice")
            if not choice:
                continue
            tags.append({
                "key": key, "label": label, "bucket": bucket,
                "kind": "choice", "text": str(choice),
                "value": choice,
                "prob": round(float(ans.get("confidence") or 0.0), 4),
            })
        elif qtype == "score":
            score = float(ans.get("score") or 0.0)
            level, level_conf = score_level(
                score, qspec.get("criteria"),
                legend=ans.get("legend"), probabilities=ans.get("probabilities"))
            tags.append({
                "key": key, "label": label, "bucket": bucket,
                "kind": "score", "text": level or f"{score:.2f}",
                "value": round(score, 4),
                "prob": round(level_conf, 4) if level_conf is not None
                        else round(score, 4),
                "levels": ans.get("probabilities"),
                "legend": ans.get("legend"),
            })
    return tags


def positive_tags(tags):
    """Tags worth surfacing as annotations (a noul only when it fired)."""
    out = []
    for tag in tags:
        if tag["kind"] == "noul" and not tag["value"]:
            continue
        out.append(tag)
    return out


def summarize_frames(frames, spec):
    """Roll per-frame annotations up into a clip-level summary."""
    tag_counts = {}
    key_values = {}
    for frame in frames:
        for tag in frame.get("tags", []):
            if tag["kind"] == "noul" and not tag["value"]:
                continue
            tag_counts[tag["text"]] = tag_counts.get(tag["text"], 0) + 1
            slot = key_values.setdefault(
                tag["key"], {"label": tag["label"], "values": {}})
            if tag["kind"] == "noul":
                slot["values"]["yes"] = slot["values"].get("yes", 0) + 1
            else:
                slot["values"][tag["text"]] = \
                    slot["values"].get(tag["text"], 0) + 1

    analyzed = len(frames) or 1
    return {
        "frames_analyzed": len(frames),
        "tag_counts": dict(sorted(
            tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_question": key_values,
        "flags": {
            "has_people": any(
                t["key"] == "contains_people" and t["value"]
                for f in frames for t in f.get("tags", [])),
            "has_vehicles": any(
                t["key"] == "contains_vehicles" and t["value"]
                for f in frames for t in f.get("tags", [])),
            "has_screen": any(
                t["key"] == "is_screen_content" and t["value"]
                for f in frames for t in f.get("tags", [])),
            "has_livestream": any(
                t["key"] == "is_livestream_frame" and t["value"]
                for f in frames for t in f.get("tags", [])),
            "has_game": any(
                t["key"] == "contains_game" and t["value"]
                for f in frames for t in f.get("tags", [])),
            "has_text": any(
                t["key"] == "has_text" and t["value"]
                for f in frames for t in f.get("tags", [])),
        },
        "coverage": {
            tag: round(count / analyzed, 3) for tag, count in tag_counts.items()
        },
    }


# ===========================================================================
# Media helpers
# ===========================================================================

def encode_image_base64(image_path):
    with open(image_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _ffmpeg_reason(stderr):
    noise = ("libavformat", "libavcodec", "libavutil", "libavdevice",
             "libavfilter", "libswscale", "libswresample", "built with",
             "configuration:", "Press [q]")
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    useful = [ln for ln in lines
              if not any(ln.startswith(prefix) for prefix in noise)
              and not ln.startswith("lib")]
    return " | ".join(useful[-2:])[:300] or "ffmpeg failed"


def probe_media(path):
    """Read codec, duration, resolution and fps in one ffprobe call."""
    info = {"duration": None, "width": None, "height": None,
            "fps": None, "codec": None, "format": None, "frames": None}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=width,height,avg_frame_rate,codec_name,nb_frames"
             ":format=duration,format_name",
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
        elif key == "nb_frames" and val.isdigit():
            info["frames"] = int(val)
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


def sniff_kind(path):
    """Classify by content when the extension is missing or unknown."""
    meta = probe_media(path)
    fmt = (meta["format"] or "").lower()
    if fmt.endswith("_pipe") or fmt in {"gif", "image2"}:
        return "image"
    if meta["duration"] and meta["duration"] > 0:
        return "video"
    if meta["codec"] and meta["codec"] not in {
            "png", "mjpeg", "bmp", "gif", "webp", "tiff"}:
        return "video"
    if meta["width"] and meta["height"]:
        return "image"
    return "unknown"


def extract_frames(video_path, output_dir,
                   interval=FRAME_INTERVAL_SECONDS,
                   max_frames=MAX_FRAMES_EXTRACTED):
    """Sample one JPEG every `interval` seconds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / max(0.1, interval)
    cmd = [
        "ffmpeg", "-nostdin", "-i", str(video_path),
        "-vf", f"fps={fps},scale='min({FRAME_MAX_EDGE},iw)':-2",
        "-frames:v", str(max_frames),
        "-q:v", "3", "-y", str(output_dir / "frame_%06d.jpg"),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg 抽幀超時")
    except FileNotFoundError:
        raise RuntimeError("找不到 ffmpeg，請先安裝（macOS: brew install ffmpeg）")
    if result.returncode != 0:
        raise RuntimeError(
            f"無法解碼這個影片（可能已損壞）：{_ffmpeg_reason(result.stderr)}")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("沒有抽取到任何幀（影片可能太短）")
    return frames


def local_metrics(frame_paths):
    """Cheap local visual features that enrich the model's state text.

    `motion` is mean absolute pixel difference between consecutive sampled
    frames. It measures how much the *content* changed, which conflates camera
    movement with scene cuts and on-screen animation, so it is reported to the
    model as raw change rather than as camera motion.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return [{} for _ in frame_paths]

    metrics = []
    previous = None
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            metrics.append({})
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean()) / 255.0
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float((edges > 0).mean())

        motion = None
        if previous is not None and previous.shape == gray.shape:
            motion = float(np.abs(gray.astype("int16")
                                  - previous.astype("int16")).mean()) / 255.0
        previous = gray

        metrics.append({
            "brightness": round(brightness, 3),
            "edges": round(edge_ratio, 4),
            "motion": None if motion is None else round(motion, 4),
        })
    return metrics


def build_state_text(spec, filename, meta, frame_index, total_frames,
                     timestamp, metrics):
    """Compose the `state` string the model classifies against."""
    parts = []
    if spec.get("context"):
        parts.append(spec["context"])

    facts = [f"Frame {frame_index} of {total_frames} at {timestamp:.1f}s."]
    if meta.get("duration"):
        facts.append(f"Clip length {meta['duration']:.1f}s.")
    if meta.get("width") and meta.get("height"):
        facts.append(f"Resolution {meta['width']}x{meta['height']}.")
    if meta.get("fps"):
        facts.append(f"{meta['fps']} fps.")
    if meta.get("codec"):
        facts.append(f"Codec {meta['codec']}.")
    facts.append(f"Source file: {filename}.")
    parts.append(" ".join(facts))

    if metrics:
        visual = []
        if metrics.get("brightness") is not None:
            visual.append(f"mean brightness {metrics['brightness']:.2f}")
        if metrics.get("edges") is not None:
            visual.append(f"edge density {metrics['edges']:.3f}")
        if metrics.get("motion") is not None:
            visual.append(
                f"frame-to-frame pixel change {metrics['motion']:.3f} "
                "(content change, not camera movement — scene cuts also raise this)")
        if visual:
            parts.append("Visual measurements: " + ", ".join(visual) + ".")
    return "\n\n".join(parts)


# ===========================================================================
# Jobs — annotation is slow, so work runs off-request with progress polling
# ===========================================================================

_jobs = {}
_jobs_lock = threading.Lock()


def _job_new(kind):
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "created": time.time(),
            "progress": {"done": 0, "total": 0},
            "frames": [],
            "summary": None,
            "meta": {},
            "error": None,
            "warnings": [],
        }
    return job_id


def _job_update(job_id, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job.update(fields)


def _job_progress(job_id, done, total):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job["progress"] = {"done": done, "total": total}


def _job_push_frame(job_id, frame):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job["frames"].append(frame)


def run_annotation_job(job_id, media_path, filename, kind, max_frames):
    """Worker: sample frames, annotate each, then summarise."""
    temp_root = Path(tempfile.mkdtemp(prefix="clef_annot_"))
    try:
        spec = get_questions()
        meta = probe_media(media_path)

        if kind == "video":
            frame_dir = temp_root / "frames"
            frames = extract_frames(media_path, frame_dir)
            # Keep the sampled frames that get analyzed, but report the total.
            selected = frames[:max_frames]
            timestamps = [i * FRAME_INTERVAL_SECONDS for i in range(len(selected))]
        else:
            selected = [Path(media_path)]
            timestamps = [0.0]

        metrics = local_metrics(selected)
        total = len(selected)
        _job_update(job_id, meta={
            "kind": kind, "filename": filename,
            "video": meta if kind == "video" else {"width": meta.get("width"),
                                                   "height": meta.get("height")},
            "sampled_frames": (len(frames) if kind == "video" else 1),
        })
        _job_progress(job_id, 0, total)

        annotated = []
        for index, frame_path in enumerate(selected):
            state_text = build_state_text(
                spec, filename, meta,
                frame_index=index + 1, total_frames=total,
                timestamp=timestamps[index],
                metrics=metrics[index] if index < len(metrics) else {},
            )
            started = time.perf_counter()
            try:
                answers, skipped, usage = annotate_frame(
                    frame_path, state_text, spec)
                tags = answer_to_tags(answers, spec, skipped)
                frame_error = None
            except Exception as exc:
                answers, skipped, tags, usage = {}, [], [], {}
                frame_error = str(exc)
                with _jobs_lock:
                    _jobs[job_id]["warnings"].append(
                        f"frame {index + 1}: {exc}")

            elapsed_ms = round((time.perf_counter() - started) * 1000)

            record = {
                "frame_index": index + 1,
                "timestamp": round(timestamps[index], 2),
                "image_b64": encode_image_base64(frame_path),
                "metrics": metrics[index] if index < len(metrics) else {},
                "tags": tags,
                "highlights": [t["text"] for t in positive_tags(tags)],
                "answers": answers,
                "skipped": skipped,
                "usage": usage,
                "latency_ms": elapsed_ms,
                "error": frame_error,
            }
            annotated.append(record)
            _job_push_frame(job_id, record)
            _job_progress(job_id, index + 1, total)

        summary = summarize_frames(annotated, spec)
        summary["question_set"] = spec.get("name")
        summary["question_version"] = spec.get("version")
        summary["media"] = {
            "filename": filename, "kind": kind,
            "duration": meta.get("duration"),
            "width": meta.get("width"), "height": meta.get("height"),
        }
        _job_update(job_id, status="done", summary=summary)

    except Exception as exc:
        _job_update(job_id, status="error", error=str(exc))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# ===========================================================================
# Routes
# ===========================================================================

@app.errorhandler(413)
def too_large(_err):
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({"error": f"檔案太大，上限 {limit_mb} MB。請改用較短的片段。"}), 413


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/questions")
def api_questions():
    """Expose the loaded question set (and raw YAML) to the UI."""
    try:
        spec = get_questions()
    except QuestionSpecError as exc:
        return jsonify({"error": str(exc)}), 500

    want_raw = request.args.get("raw") in ("1", "true", "yes")
    payload = {
        "status": "ok",
        "name": spec["name"],
        "version": spec["version"],
        "description": spec["description"],
        "source_path": spec["source_path"],
        "count": len(spec["questions"]),
        "waves": question_waves(spec),
        "questions": [
            {
                "key": key,
                "type": q["type"],
                "label": q["label"],
                "tag": q["tag"],
                "conditional": bool(q.get("when")),
                "when": q.get("when"),
            }
            for key, q in spec["questions"].items()
        ],
    }
    if want_raw:
        payload["raw"] = spec["raw"]
    return jsonify(payload)


@app.route("/api/questions/reload", methods=["POST"])
def api_questions_reload():
    """Re-read the YAML from disk so questions can change without a restart."""
    try:
        spec = reload_questions()
    except QuestionSpecError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "status": "reloaded",
        "name": spec["name"],
        "version": spec["version"],
        "count": len(spec["questions"]),
    })


@app.route("/api/health")
def api_health():
    def has(binary):
        return shutil.which(binary) is not None

    try:
        spec = get_questions()
        question_info = {
            "ok": True, "name": spec["name"],
            "version": spec["version"], "count": len(spec["questions"]),
        }
    except QuestionSpecError as exc:
        question_info = {"ok": False, "error": str(exc)}

    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "ollama": OLLAMA_BASE_URL,
        "ffmpeg": has("ffmpeg"),
        "ffprobe": has("ffprobe"),
        "yt_dlp": has("yt-dlp"),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "max_frames": MAX_FRAMES_ANALYZED,
        "questions": question_info,
    })


@app.route("/api/annotate", methods=["POST"])
def api_annotate():
    """Accept an image or video upload and start an annotation job."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "沒有收到檔案（欄位名稱需為 'file'）"}), 400

    filename = os.path.basename(upload.filename)
    ext = Path(filename).suffix.lower()
    try:
        max_frames = int(request.form.get("max_frames") or MAX_FRAMES_ANALYZED)
    except ValueError:
        max_frames = MAX_FRAMES_ANALYZED
    max_frames = max(1, min(max_frames, MAX_FRAMES_EXTRACTED))

    staging = Path(tempfile.mkdtemp(prefix="clef_upload_"))
    stored = staging / (filename or "upload.bin")
    upload.save(stored)
    if stored.stat().st_size == 0:
        shutil.rmtree(staging, ignore_errors=True)
        return jsonify({"error": "上傳的檔案是空的"}), 400

    if ext in IMAGE_EXTENSIONS:
        kind = "image"
    elif ext in VIDEO_EXTENSIONS:
        kind = "video"
    else:
        kind = sniff_kind(stored)
        if kind == "unknown":
            supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
            shutil.rmtree(staging, ignore_errors=True)
            return jsonify({
                "error": f"無法辨識的檔案類型（{ext or '無副檔名'}）。"
                         f"支援: {supported}"}), 400

    job_id = _job_new(kind)
    thread = threading.Thread(
        target=_run_and_cleanup,
        args=(job_id, stored, filename, kind, max_frames, staging),
        daemon=True)
    thread.start()
    return jsonify({"status": "started", "job_id": job_id, "kind": kind}), 202


def _run_and_cleanup(job_id, stored, filename, kind, max_frames, staging):
    try:
        run_annotation_job(job_id, stored, filename, kind, max_frames)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@app.route("/api/job/<job_id>")
def api_job(job_id):
    """Poll an annotation job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "找不到這個 job"}), 404
        snapshot = copy.deepcopy(job)
    return jsonify(snapshot)


@app.route("/api/annotate_sync", methods=["POST"])
def api_annotate_sync():
    """Synchronous variant, handy for scripts and curl-based checks."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "沒有收到檔案"}), 400

    filename = os.path.basename(upload.filename)
    staging = Path(tempfile.mkdtemp(prefix="clef_upload_"))
    stored = staging / filename
    upload.save(stored)
    ext = Path(filename).suffix.lower()
    kind = ("image" if ext in IMAGE_EXTENSIONS else
            "video" if ext in VIDEO_EXTENSIONS else sniff_kind(stored))

    job_id = _job_new(kind)
    try:
        run_annotation_job(job_id, stored, filename, kind, MAX_FRAMES_ANALYZED)
        with _jobs_lock:
            job = copy.deepcopy(_jobs[job_id])
        if job["status"] == "error":
            return jsonify({"error": job["error"]}), 500
        return jsonify(job)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ===========================================================================
# Frontend
# ===========================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clef Vision Annotation Studio</title>
<style>
  :root {
    --bg: #0f0f10; --panel: #18181b; --border: #27272a;
    --text: #e4e4e7; --muted: #a1a1aa; --accent: #4ade80;
    --warn: #e6a03c; --err: #ef4444;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex; flex-direction: column; align-items: center;
  }
  .wrap { width: 100%; max-width: 1240px; }
  header { display: flex; justify-content: space-between; align-items: flex-start;
           gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
  h1 { color: var(--accent); margin: 0 0 4px 0; font-size: 23px; font-weight: 600; }
  header p { color: var(--muted); margin: 0; font-size: 13px; max-width: 620px; }
  .badges { display: flex; gap: 8px; flex-wrap: wrap; }
  .badge { background: #1f2937; border: 1px solid var(--border); color: var(--accent);
           padding: 4px 10px; border-radius: 6px; font: 11px ui-monospace, monospace; }
  .badge.bad { border-color: var(--err); color: var(--err); }
  .badge.dim { color: var(--muted); }
  .cols { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
  .col { display: flex; flex-direction: column; gap: 18px; min-width: 340px; flex: 1; }
  .col.right { flex: 1.35; }
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 9px; padding: 16px; }
  .panel-head { display: flex; justify-content: space-between; align-items: center;
                gap: 10px; font-size: 11px; font-weight: 600; letter-spacing: .5px;
                text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
  .drop { border: 1px dashed var(--border); border-radius: 7px; height: 260px;
          display: flex; align-items: center; justify-content: center; cursor: pointer;
          background: #121214; overflow: hidden; position: relative; }
  .drop:hover { border-color: var(--accent); }
  .drop img, .drop video { max-width: 100%; max-height: 100%; object-fit: contain; }
  .hint { color: var(--muted); font-size: 13px; text-align: center; padding: 0 20px; }
  .tiny { color: var(--muted); font-size: 11px; margin: 10px 0 0 0; line-height: 1.6; }
  label.row { display: flex; align-items: center; gap: 8px; font-size: 12px;
              color: var(--muted); margin-top: 10px; }
  input[type=number] { width: 62px; background: #121214; border: 1px solid var(--border);
                       color: var(--text); border-radius: 5px; padding: 4px 6px; font: 12px ui-monospace, monospace; }
  button { background: #27272a; color: var(--text); border: none; padding: 8px 12px;
           border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; }
  button:hover { background: #3f3f46; }
  button.primary { background: var(--accent); color: #06270f; }
  button.primary:hover { background: #22c55e; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .kv { display: flex; justify-content: space-between; gap: 10px;
        font: 12px ui-monospace, monospace; padding: 3px 0; }
  .kv span:first-child { color: var(--muted); }
  .kv span:last-child { color: var(--accent); text-align: right; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
  .chip { font: 11px ui-monospace, monospace; padding: 3px 8px; border-radius: 999px;
          border: 1px solid #3f3f46; background: #232326; color: #d4d4d8; white-space: nowrap; }
  .chip.people { border-color: #7c9cf5; color: #a9c0ff; }
  .chip.vehicles { border-color: #e6a03c; color: #f0c07a; }
  .chip.screen { border-color: #4ade80; color: #86efac; }
  .chip.scene { border-color: #a78bfa; color: #c4b5fd; }
  .chip.motion { border-color: #f472b6; color: #f9a8d4; }
  .chip.text { border-color: #38bdf8; color: #7dd3fc; }
  .chip.environment { border-color: #34d399; color: #6ee7b7; }
  .chip.pipeline { border-color: #64748b; color: #cbd5e1; }
  .chip.neg { opacity: .42; text-decoration: line-through; }
  .timeline { margin-top: 12px; }
  .bar { height: 5px; background: #27272a; border-radius: 3px; overflow: hidden; }
  .bar > div { height: 100%; width: 0%; background: var(--accent); transition: width .25s ease; }
  .frames { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
  .fthumb { width: 58px; height: 40px; border-radius: 4px; overflow: hidden;
            border: 2px solid transparent; cursor: pointer; background: #000; flex: 0 0 auto; }
  .fthumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .fthumb.active { border-color: var(--accent); }
  .fthumb.pending { opacity: .35; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
  .stat { background: #121214; border: 1px solid var(--border); border-radius: 6px; padding: 9px 11px; }
  .stat b { display: block; font: 16px ui-monospace, monospace; color: var(--accent); }
  .stat span { font-size: 11px; color: var(--muted); }
  pre { font: 11px/1.5 ui-monospace, monospace; color: var(--muted); background: #121214;
        border-radius: 6px; padding: 12px; max-height: 280px; overflow: auto; margin: 0;
        white-space: pre-wrap; word-break: break-word; }
  .muted { color: var(--muted); }
  .center { text-align: center; padding: 36px 0; }
  .spin { color: var(--accent); }
  .err { color: var(--err); }
  .qlist { max-height: 210px; overflow: auto; display: flex; flex-direction: column; gap: 3px; }
  .qrow { display: flex; gap: 8px; align-items: baseline; font: 11px ui-monospace, monospace;
          padding: 2px 0; border-bottom: 1px solid #1f1f23; }
  .qrow .type { color: var(--muted); flex: 0 0 42px; }
  .qrow .name { color: var(--text); }
  .qrow .cond { color: var(--warn); margin-left: auto; font-size: 10px; }
  .hidden { display: none !important; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Clef Vision Annotation Studio</h1>
      <p>Frame-level tagging with <code>clef-flash</code>. Questions are loaded from
         <code>questions.yaml</code> &mdash; edit the YAML and reload without restarting.</p>
    </div>
    <div class="badges">
      <div class="badge" id="b-model">model: clef-flash</div>
      <div class="badge dim" id="b-questions">questions: ...</div>
      <div class="badge dim" id="b-ffmpeg">ffmpeg: ...</div>
      <button onclick="reloadQuestions()" id="b-reload">Reload questions</button>
    </div>
  </header>

  <div class="cols">
    <div class="col">
      <div class="panel">
        <div class="panel-head"><span>Media input</span><span id="fname" class="muted">no file</span></div>
        <div class="drop" id="drop" onclick="document.getElementById('file').click()">
          <div class="hint" id="hint">Click to choose an image or video clip<br>
            <span style="font-size:11px">mp4 · mov · m4v · webm · avi · mkv &nbsp;|&nbsp; png · jpg · webp</span>
          </div>
          <img id="pimg" class="hidden" alt="">
          <video id="pvid" class="hidden" muted playsinline controls></video>
        </div>
        <input type="file" id="file" class="hidden"
               accept="image/*,video/*,.mp4,.mov,.m4v,.webm,.avi,.mkv,.mpeg,.mpg,.3gp,.mts,.m2ts"
               onchange="onFile(event)">
        <label class="row">
          Frames to annotate
          <input type="number" id="maxframes" value="5" min="1" max="20">
          <span class="muted" style="font-size:11px">each frame = 1&ndash;2 model calls</span>
        </label>
        <div class="timeline" id="timeline" style="display:none">
          <div class="bar"><div id="progress"></div></div>
          <div class="frames" id="thumbs"></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head"><span>Media state</span><span id="jobstatus" class="muted">idle</span></div>
        <div class="kv"><span>resolution</span><span id="m-res">-</span></div>
        <div class="kv"><span>duration</span><span id="m-dur">-</span></div>
        <div class="kv"><span>codec / fps</span><span id="m-codec">-</span></div>
        <div class="kv"><span>frames analyzed</span><span id="m-frames">-</span></div>
        <div class="kv"><span>total latency</span><span id="m-latency">-</span></div>
      </div>

      <div class="panel">
        <div class="panel-head"><span>Question set</span><span id="qcount" class="muted">-</span></div>
        <div class="qlist" id="qlist"><div class="muted" style="font-size:12px">loading...</div></div>
      </div>
    </div>

    <div class="col right">
      <div class="panel">
        <div class="panel-head"><span>Clip summary</span><span id="sum-tag" class="muted">-</span></div>
        <div id="summary"><div class="center muted">Upload a clip to begin.</div></div>
      </div>

      <div class="panel">
        <div class="panel-head"><span>Frame annotations</span><span id="frametag" class="muted">-</span></div>
        <div id="frameview"><div class="center muted">No frame selected.</div></div>
      </div>

      <div class="panel">
        <div class="panel-head"><span>Raw response</span><span class="muted" id="rawtag"></span></div>
        <pre id="raw">Waiting...</pre>
      </div>
    </div>
  </div>
</div>

<script>
let jobId = null, poll = null;
let frames = [];
let selected = 0;
let objectUrl = null;
let startedAt = 0;

const IMG_EXT = ['png','jpg','jpeg','webp','bmp','gif','tif','tiff'];
const VID_EXT = ['mp4','mov','m4v','avi','mkv','webm','wmv','flv','mpeg','mpg','3gp','mts','m2ts','ogv'];

function kindOf(file) {
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  if (VID_EXT.includes(ext)) return 'video';
  if (IMG_EXT.includes(ext)) return 'image';
  if ((file.type || '').startsWith('video/')) return 'video';
  if ((file.type || '').startsWith('image/')) return 'image';
  return 'unknown';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function setBadge(id, text, cls) {
  const el = document.getElementById(id);
  el.innerText = text;
  el.className = 'badge' + (cls ? ' ' + cls : '');
}

function onFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  stopPolling();

  document.getElementById('fname').innerText = file.name;
  document.getElementById('hint').classList.add('hidden');
  document.getElementById('pimg').classList.add('hidden');
  const pvid = document.getElementById('pvid');
  pvid.classList.add('hidden');
  pvid.pause(); pvid.removeAttribute('src'); pvid.load();

  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  objectUrl = URL.createObjectURL(file);

  if (kindOf(file) === 'video') {
    pvid.src = objectUrl;
    pvid.classList.remove('hidden');
  } else {
    const img = document.getElementById('pimg');
    img.src = objectUrl;
    img.classList.remove('hidden');
  }

  frames = []; selected = 0;
  document.getElementById('summary').innerHTML =
    '<div class="center spin">Annotating frames...</div>';
  document.getElementById('frameview').innerHTML =
    '<div class="center muted">Waiting for the first frame.</div>';
  document.getElementById('raw').innerText = 'Waiting...';
  document.getElementById('timeline').style.display = 'block';
  document.getElementById('progress').style.width = '0%';
  document.getElementById('thumbs').innerHTML = '';
  document.getElementById('jobstatus').innerText = 'running';
  startedAt = performance.now();

  const body = new FormData();
  body.append('file', file);
  body.append('max_frames', document.getElementById('maxframes').value || '5');

  fetch('/api/annotate', { method: 'POST', body })
    .then(r => r.json().then(j => ({ ok: r.ok, j })))
    .then(({ ok, j }) => {
      if (j.error) throw new Error(j.error);
      if (!ok) throw new Error('HTTP ' + r.status);
      jobId = j.job_id;
      poll = setInterval(pollJob, 700);
    })
    .catch(err => fail(err.message));
}

function pollJob() {
  if (!jobId) return;
  fetch('/api/job/' + jobId)
    .then(r => r.json())
    .then(job => {
      if (job.error && job.status !== 'running') throw new Error(job.error);

      if (job.meta && job.meta.video && !document.getElementById('m-res').dataset.set) {
        const v = job.meta.video || {};
        document.getElementById('m-res').innerText =
          (v.width && v.height) ? v.width + 'x' + v.height : '-';
        document.getElementById('m-dur').innerText =
          v.duration ? v.duration.toFixed(1) + 's' : '-';
        document.getElementById('m-codec').innerText =
          [v.codec, v.fps ? v.fps + ' fps' : null].filter(Boolean).join(' / ') || '-';
        document.getElementById('m-res').dataset.set = '1';
      }

      frames = job.frames || [];
      const p = job.progress || { done: 0, total: 0 };
      if (p.total) {
        document.getElementById('progress').style.width =
          Math.round((p.done / p.total) * 100) + '%';
      }
      document.getElementById('m-frames').innerText =
        p.total ? p.done + ' / ' + p.total : '-';
      renderThumbs(frames, p.total);

      if (frames.length && !document.getElementById('frameview').dataset.picked) {
        selectFrame(0);
      }
      if (job.summary) renderSummary(job.summary);

      if (job.status === 'done') {
        stopPolling();
        document.getElementById('jobstatus').innerText = 'done';
        document.getElementById('m-latency').innerText =
          Math.round(performance.now() - startedAt) + ' ms';
        const totalLatency = frames.reduce((a, f) => a + (f.latency_ms || 0), 0);
        document.getElementById('rawtag').innerText = totalLatency + ' ms model time';
        if (job.warnings && job.warnings.length) {
          document.getElementById('jobstatus').innerText =
            'done (' + job.warnings.length + ' warning)';
        }
      } else if (job.status === 'error') {
        throw new Error(job.error || 'annotation failed');
      }
    })
    .catch(err => fail(err.message));
}

function stopPolling() { if (poll) { clearInterval(poll); poll = null; } }

function fail(msg) {
  stopPolling();
  document.getElementById('jobstatus').innerText = 'error';
  document.getElementById('jobstatus').className = 'err';
  document.getElementById('summary').innerHTML =
    '<div class="center err">' + escapeHtml(msg) + '</div>';
}

function renderThumbs(list, total) {
  const box = document.getElementById('thumbs');
  const want = Math.max(total || 0, list.length);
  if (box.dataset.count === String(want) && box.dataset.loaded === String(list.length)) return;
  box.dataset.count = String(want);
  box.dataset.loaded = String(list.length);

  let html = '';
  for (let i = 0; i < want; i++) {
    const f = list[i];
    if (f) {
      html += '<div class="fthumb' + (i === selected ? ' active' : '') +
              '" onclick="selectFrame(' + i + ')"><img src="data:image/jpeg;base64,' +
              f.image_b64 + '" alt="' + (i + 1) + '"></div>';
    } else {
      html += '<div class="fthumb pending" title="pending"></div>';
    }
  }
  box.innerHTML = html;
}

function selectFrame(index) {
  if (!frames[index]) return;
  selected = index;
  document.getElementById('frameview').dataset.picked = '1';
  renderThumbs(frames, frames.length);
  renderFrame(frames[index]);
}

function chipClass(bucket) {
  return 'chip ' + (['people','vehicles','screen','scene','motion','text','environment','pipeline']
    .includes(bucket) ? bucket : '');
}

function renderFrame(frame) {
  document.getElementById('frametag').innerText =
    'frame ' + frame.frame_index + ' @ ' + frame.timestamp.toFixed(1) + 's' +
    (frame.latency_ms ? ' · ' + frame.latency_ms + ' ms' : '');

  document.getElementById('pimg').classList.remove('hidden');
  document.getElementById('pimg').src = 'data:image/jpeg;base64,' + frame.image_b64;
  document.getElementById('pvid').classList.add('hidden');

  if (frame.error) {
    document.getElementById('frameview').innerHTML =
      '<div class="center err">' + escapeHtml(frame.error) + '</div>';
    document.getElementById('raw').innerText = JSON.stringify(frame, null, 2);
    return;
  }

  const tags = frame.tags || [];
  const on = tags.filter(t => t.kind !== 'noul' || t.value);
  const off = tags.filter(t => t.kind === 'noul' && !t.value);

  let html = '<div class="chips">';
  on.forEach(t => {
    html += '<span class="' + chipClass(t.bucket) + '" title="' + escapeHtml(t.key) + '">' +
            escapeHtml(t.kind === 'noul' ? t.label : t.label + ': ' + t.text) + '</span>';
  });
  if (!on.length) html += '<span class="muted" style="font-size:12px">no positive tags</span>';
  html += '</div>';

  if (off.length) {
    html += '<div class="chips" style="margin-top:10px">';
    off.forEach(t => {
      html += '<span class="' + chipClass(t.bucket) + ' neg">' + escapeHtml(t.label) + '</span>';
    });
    html += '</div>';
  }

  if (frame.metrics && Object.keys(frame.metrics).length) {
    const m = frame.metrics;
    html += '<div style="margin-top:12px">';
    html += '<div class="kv"><span>brightness</span><span>' + (m.brightness ?? '-') + '</span></div>';
    html += '<div class="kv"><span>edge density</span><span>' + (m.edges ?? '-') + '</span></div>';
    html += '<div class="kv"><span>pixel change (vs prev)</span><span>' + (m.motion ?? 'n/a') + '</span></div>';
    html += '</div>';
  }

  // Score questions carry a discrete level distribution; show it as bars.
  const scores = on.filter(t => t.kind === 'score');
  if (scores.length) {
    html += '<div style="margin-top:14px">';
    scores.forEach(t => {
      html += '<div class="panel-head" style="margin:10px 0 6px"><span>' +
              escapeHtml(t.label) + '</span><span>' + escapeHtml(t.text) + '</span></div>';
      if (t.levels && t.legend) {
        Object.keys(t.legend).forEach(k => {
          const pct = Math.round((t.levels[k] || 0) * 100);
          html += '<div class="kv"><span>' + escapeHtml(t.legend[k]) + '</span>' +
                  '<span>' + pct + '%</span></div>';
          html += '<div class="bar" style="margin-bottom:5px"><div style="width:' + pct + '%"></div></div>';
        });
      }
    });
    html += '</div>';
  }

  if (frame.skipped && frame.skipped.length) {
    html += '<p class="tiny">Skipped by rules: ' +
            escapeHtml(frame.skipped.join(', ')) + '</p>';
  }

  document.getElementById('frameview').innerHTML = html;
  document.getElementById('raw').innerText = JSON.stringify(
    { frame_index: frame.frame_index, timestamp: frame.timestamp,
      answers: frame.answers }, null, 2);
}

function renderSummary(sum) {
  document.getElementById('sum-tag').innerText =
    (sum.frames_analyzed || 0) + ' frame(s)' +
    (sum.question_version ? ' · q v' + sum.question_version : '');

  const flags = sum.flags || {};
  const flagLabels = {
    has_people: 'people', has_vehicles: 'vehicles', has_screen: 'screen content',
    has_livestream: 'livestream', has_game: 'game', has_text: 'text'
  };

  let html = '<div class="summary-grid">';
  Object.entries(flagLabels).forEach(([key, label]) => {
    const on = !!flags[key];
    html += '<div class="stat"><b style="color:' + (on ? 'var(--accent)' : 'var(--muted)') + '">' +
            (on ? 'YES' : 'no') + '</b><span>' + label + '</span></div>';
  });
  html += '</div>';

  const counts = sum.tag_counts || {};
  const entries = Object.entries(counts);
  if (entries.length) {
    html += '<div style="margin-top:14px"><div class="panel-head" style="margin-bottom:8px">' +
            '<span>Tag frequency</span></div><div class="chips">';
    entries.forEach(([tag, n]) => {
      html += '<span class="chip">' + escapeHtml(tag) + ' ×' + n + '</span>';
    });
    html += '</div></div>';
  }
  document.getElementById('summary').innerHTML = html;
}

function loadQuestions() {
  fetch('/api/questions')
    .then(r => r.json())
    .then(q => {
      if (q.error) {
        setBadge('b-questions', 'yaml error', 'bad');
        document.getElementById('qlist').innerHTML =
          '<div class="err" style="font-size:12px">' + escapeHtml(q.error) + '</div>';
        return;
      }
      setBadge('b-questions', 'questions: ' + q.count + ' (v' + (q.version ?? '?') + ')');
      document.getElementById('qcount').innerText =
        q.count + ' · ' + (q.name || '');
      let html = '';
      q.questions.forEach(item => {
        html += '<div class="qrow"><span class="type">' + escapeHtml(item.type) + '</span>' +
                '<span class="name">' + escapeHtml(item.key) + '</span>' +
                (item.conditional ? '<span class="cond">conditional</span>' : '') +
                '</div>';
      });
      document.getElementById('qlist').innerHTML = html;
    })
    .catch(() => setBadge('b-questions', 'questions: offline', 'bad'));
}

function reloadQuestions() {
  const btn = document.getElementById('b-reload');
  btn.disabled = true;
  fetch('/api/questions/reload', { method: 'POST' })
    .then(r => r.json())
    .then(j => {
      if (j.error) { setBadge('b-questions', 'yaml error', 'bad');
        document.getElementById('qlist').innerHTML =
          '<div class="err" style="font-size:12px">' + escapeHtml(j.error) + '</div>';
        return; }
      loadQuestions();
    })
    .catch(e => setBadge('b-questions', 'reload failed', 'bad'))
    .finally(() => { btn.disabled = false; });
}

fetch('/api/health').then(r => r.json()).then(h => {
  if (!h.ffmpeg || !h.ffprobe) setBadge('b-ffmpeg', 'ffmpeg missing', 'bad');
  else setBadge('b-ffmpeg', 'ffmpeg ok', 'dim');
  if (h.questions && !h.questions.ok) {
    setBadge('b-questions', 'yaml error', 'bad');
  }
  if (h.max_frames) document.getElementById('maxframes').value = h.max_frames;
}).catch(() => {});

loadQuestions();
</script>
</body>
</html>
"""


# ===========================================================================
# Entry point
# ===========================================================================

def _port_is_free(host, port):
    """True when nothing else is bound to host:port.

    SO_REUSEADDR is deliberately not set: on macOS it allows binding a specific
    address while a wildcard listener (AirPlay's ControlCenter on *:5000)
    already holds the port, silently shadowing this server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host, preferred, span=20):
    for port in range(preferred, preferred + span):
        if _port_is_free(host, port):
            return port
    return preferred


def main():
    global _questions_path, _questions_spec

    parser = argparse.ArgumentParser(description="Clef Vision Annotation Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH),
                        help="Path to the questions YAML")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="Validate the question YAML and exit")
    args = parser.parse_args()

    _questions_path = Path(args.questions).resolve()
    _questions_spec = None

    try:
        spec = reload_questions()
    except QuestionSpecError as exc:
        print(f"題目檔錯誤: {exc}")
        return 2

    print(f"題目: {spec['name']} (v{spec['version']}) · "
          f"{len(spec['questions'])} 題 · {_questions_path}")
    if args.check:
        for index, wave in enumerate(question_waves(spec)):
            print(f"  wave {index}: {', '.join(wave)}")
        return 0

    port = args.port
    if not _port_is_free(args.host, port):
        fallback = _pick_port(args.host, port + 1)
        print(f"⚠️  連接埠 {port} 已被占用（macOS 常見為 AirPlay Receiver）。改用 {fallback}。")
        port = fallback

    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            print(f"⚠️  找不到 {binary}，影片功能無法使用。")

    print(f"啟動 Annotation Studio 於 http://{args.host}:{port}")
    app.run(host=args.host, port=port, debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
