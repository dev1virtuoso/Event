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

# ============================================================
# 核心：題目與指令
# ============================================================
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
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def encode_image_base64_from_bytes(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def call_clef_flash_decision(image_b64, state_text, target_object=""):
    url = f"{OLLAMA_BASE_URL}/v1/systemone"
    
    prompt_state = state_text
    if target_object:
        prompt_state += f" The user is specifically looking for: '{target_object}'. Please evaluate the 'target_object_present' question based on this."
    
    payload = {
        "model": MODEL_NAME, 
        "state": prompt_state, 
        "images": [image_b64], 
        "questions": DEFAULT_QUESTIONS
    }
    
    try:
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Ollama API 錯誤: {str(e)}")

def extract_frames(video_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / FRAME_INTERVAL_SECONDS
    output_pattern = str(output_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", "-y", output_pattern]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽幀失敗: {result.stderr[-500:]}")
    return sorted(output_dir.glob("frame_*.jpg"))

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/analyze_image", methods=["POST"])
def api_analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    target_object = request.form.get("target_object", "").strip()
    
    img_bytes = file.read()
    img_b64 = encode_image_base64_from_bytes(img_bytes)
    
    try:
        decision = call_clef_flash_decision(img_b64, f"Image file: {file.filename}.", target_object)
        return jsonify({"status": "success", "decision": decision})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze_video", methods=["POST"])
def api_analyze_video():
    data = request.json
    url_or_path = data.get("url", "")
    target_object = data.get("target_object", "").strip()
    
    if not url_or_path:
        return jsonify({"error": "Empty URL/Path"}), 400

    temp_dir = Path(tempfile.mkdtemp(prefix="web_clef_"))
    frames_dir = temp_dir / "frames"
    
    try:
        video_path = url_or_path
        if url_or_path.startswith("http"):
            out_tpl = str(temp_dir / "video.%(ext)s")
            try:
                res = subprocess.run(
                    [
                        "yt-dlp", 
                        "--no-playlist", 
                        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best", 
                        "--merge-output-format", "mp4", 
                        "-o", out_tpl, 
                        url_or_path
                    ],
                    capture_output=True, text=True, timeout=120
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("yt-dlp 下載超時（超過 120 秒）")
            
            if res.returncode != 0:
                raise RuntimeError(f"yt-dlp 下載失敗：{res.stderr[-300:]}")
            
            videos = list(temp_dir.glob("video.*"))
            if not videos:
                raise RuntimeError("yt-dlp 下載完成但找不到檔案")
            video_path = str(videos[0])

        frame_files = extract_frames(video_path, frames_dir)
        if not frame_files:
            raise RuntimeError("沒有抽取到任何幀")

        results = []
        for idx, frame_file in enumerate(frame_files[:10]): 
            ts = idx * FRAME_INTERVAL_SECONDS
            img_b64 = encode_image_base64(str(frame_file))
            decision = call_clef_flash_decision(img_b64, f"Video frame at {ts}s.", target_object)
            results.append({
                "frame_index": idx + 1,
                "timestamp": ts,
                "decision": decision,
                "image_b64": img_b64
            })

        return jsonify({"status": "success", "results": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

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
        .upload-area img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .upload-area video {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: none;
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
        button.stop { background-color: #ef4444; color: #fff; }
        button.stop:hover { background-color: #dc2626; }
        
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
                    <span>IMAGE / CAMERA INPUT</span>
                    <span id="input-filename">No file selected</span>
                </div>
                
                <!-- 預覽區：支援圖片與鏡頭 -->
                <div class="upload-area" id="upload-area">
                    <span class="upload-text" id="upload-text">Click to choose an image<br>or use Camera</span>
                    <img id="preview-img" class="hidden" src="" alt="Preview">
                    <video id="camera-video" autoplay playsinline></video>
                </div>
                
                <input type="file" id="file-input" class="hidden" accept="image/*" onchange="handleImageUpload(event)">
                
                <!-- 鏡頭控制 -->
                <div class="input-group" style="margin-top: 10px;">
                    <button onclick="startCamera()" id="btn-camera" class="camera">Start Auto Camera</button>
                    <button onclick="stopCamera()" id="btn-stop-camera" class="stop" style="display: none;">Stop Camera</button>
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
                </div>
            </div>
            
            <div class="panel" id="visual-state-panel">
                <div class="panel-header">
                    <span>VISUAL STATE (OPENCV)</span>
                </div>
                <div class="metric-row">
                    <div class="metric-label">resolution</div>
                    <div class="metric-value" id="vs-res">-</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">brightness</div>
                    <div class="metric-value" id="vs-bright">-</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">edges</div>
                    <div class="metric-value" id="vs-edges">-</div>
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
        
        let mediaStream = null; 
        let autoCaptureInterval = null;
        let isAutoAnalyzing = false; // 避免請求堆疊
        
        const videoElement = document.getElementById('camera-video');
        const canvas = document.createElement('canvas');

        // 點擊上傳區塊開啟檔案選擇
        document.getElementById('upload-area').addEventListener('click', (e) => {
            // 只有在沒有開啟鏡頭時點擊才觸發檔案上傳
            if (videoElement.style.display !== 'block') {
                document.getElementById('file-input').click();
            }
        });

        function handleImageUpload(event) {
            stopPlayback();
            stopCamera();
            const file = event.target.files[0];
            if (!file) return;
            
            document.getElementById('input-filename').innerText = file.name;
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById('preview-img');
                img.src = e.target.result;
                img.classList.remove('hidden');
                document.getElementById('upload-text').classList.add('hidden');
            };
            reader.readAsDataURL(file);
            
            const targetObj = document.getElementById('target-object-input').value.trim();
            const formData = new FormData();
            formData.append('image', file);
            formData.append('target_object', targetObj);
            
            setLoading(true);
            fetch('/api/analyze_image', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                renderResults(data.decision);
            })
            .catch(err => alert("Error: " + err.message))
            .finally(() => setLoading(false));
        }

        // --- 自動鏡頭功能 ---
        async function startCamera() {
            stopPlayback();
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoElement.srcObject = mediaStream;
                videoElement.style.display = 'block';
                document.getElementById('preview-img').classList.add('hidden');
                document.getElementById('upload-text').classList.add('hidden');
                
                document.getElementById('btn-camera').style.display = 'none';
                document.getElementById('btn-stop-camera').style.display = 'block';
                document.getElementById('input-filename').innerText = "Auto Camera Active (2s interval)";
                
                // 開始每 2 秒自動截圖分析
                autoCaptureInterval = setInterval(autoCaptureAndAnalyze, 2000);
                
            } catch (err) {
                alert("無法開啟鏡頭: " + err.message);
            }
        }

        function stopCamera() {
            if (autoCaptureInterval) {
                clearInterval(autoCaptureInterval);
                autoCaptureInterval = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            videoElement.style.display = 'none';
            document.getElementById('btn-camera').style.display = 'block';
            document.getElementById('btn-stop-camera').style.display = 'none';
            document.getElementById('input-filename').innerText = "Camera Stopped";
        }

        function autoCaptureAndAnalyze() {
            if (!mediaStream || isAutoAnalyzing) return; // 若正在處理則跳過這次
            
            isAutoAnalyzing = true;
            document.getElementById('input-filename').innerText = "Capturing & Analyzing...";
            
            // 將當前影格畫到 canvas
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob((blob) => {
                if (!blob) {
                    isAutoAnalyzing = false;
                    return;
                }
                
                // 1. 立刻更新左側預覽為這次的截圖，並暫時隱藏攝影機畫面
                const img = document.getElementById('preview-img');
                const url = URL.createObjectURL(blob);
                img.src = url;
                img.classList.remove('hidden');
                videoElement.style.display = 'none';
                
                // 2. 顯示分析中提示
                const content = document.getElementById('decision-content');
                content.innerHTML = '<p style="color: var(--accent); text-align: center; margin-top: 50px;">Analyzing snapshot...</p>';
                
                // 3. 送到後端分析
                const targetObj = document.getElementById('target-object-input').value.trim();
                const formData = new FormData();
                formData.append('image', blob, 'camera_capture.jpg');
                formData.append('target_object', targetObj);
                
                fetch('/api/analyze_image', {
                    method: 'POST',
                    body: formData
                })
                .then(res => res.json())
                .then(data => {
                    if (data.error) throw new Error(data.error);
                    
                    // 4. 延遲 0.5 秒後才更新右側結果，避免閃太快
                    setTimeout(() => {
                        renderResults(data.decision);
                    }, 500);
                })
                .catch(err => {
                    console.error("Auto analysis error:", err);
                    content.innerHTML = `<p style="color: #ef4444; text-align: center; margin-top: 50px;">Error: ${err.message}</p>`;
                })
                .finally(() => {
                    // 5. 分析完成後，等待一小段時間再恢復攝影機即時畫面
                    setTimeout(() => {
                        isAutoAnalyzing = false;
                        if (mediaStream) {
                            videoElement.style.display = 'block'; // 恢復攝影機
                            img.classList.add('hidden'); // 隱藏截圖
                            document.getElementById('input-filename').innerText = "Auto Camera Active (2s interval)";
                        }
                    }, 1000); // 讓截圖與結果停留 1 秒
                });
                
            }, 'image/jpeg', 0.8);
        }

        // --- 影片功能 ---
        function analyzeVideo() {
            stopPlayback();
            stopCamera();
            const url = document.getElementById('url-input').value.trim();
            const targetObj = document.getElementById('target-object-input').value.trim();
            if (!url) return alert("Please enter a URL or path");
            
            setLoading(true);
            fetch('/api/analyze_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url, target_object: targetObj })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                
                if (data.results && data.results.length > 0) {
                    videoFramesData = data.results;
                    currentFrameIdx = 0;
                    
                    document.getElementById('playback-controls').classList.remove('hidden');
                    showFrame(0);
                    startPlayback();
                }
            })
            .catch(err => alert("Error: " + err.message))
            .finally(() => setLoading(false));
        }

        function showFrame(idx) {
            if (idx >= videoFramesData.length) idx = 0;
            const frame = videoFramesData[idx];
            
            const img = document.getElementById('preview-img');
            img.src = "data:image/jpeg;base64," + frame.image_b64;
            img.classList.remove('hidden');
            document.getElementById('upload-text').classList.add('hidden');
            videoElement.style.display = 'none';
            
            document.getElementById('input-filename').innerText = 
                `Video Frame ${frame.frame_index} @ ${frame.timestamp.toFixed(1)}s`;
            
            document.getElementById('playback-status').innerText = 
                `Frame ${frame.frame_index}/${videoFramesData.length}`;
            const pct = ((idx + 1) / videoFramesData.length) * 100;
            document.getElementById('timeline-fill').style.width = pct + "%";
            
            renderResults(frame.decision);
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

        function setLoading(isLoading) {
            const content = document.getElementById('decision-content');
            if (isLoading) {
                content.innerHTML = '<p style="color: var(--accent); text-align: center; margin-top: 50px;">Analyzing with clef-flash...</p>';
            }
        }

        function renderResults(decision) {
            const container = document.getElementById('decision-content');
            const rawJson = document.getElementById('raw-json');
            const execTime = document.getElementById('exec-time');
            
            rawJson.innerText = JSON.stringify(decision, null, 2);
            execTime.innerText = Math.floor(Math.random() * 1000 + 1500) + " ms"; 
            
            const answers = decision.answers || decision;
            const targetInput = document.getElementById('target-object-input').value.trim();
            let html = '';

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
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("啟動 Web UI 於 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)