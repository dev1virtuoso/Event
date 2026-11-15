#!/usr/bin/env python3
"""
clef_flash_pygame.py (Pygame + pygame_gui 版本)

功能：
  1. 深色 GUI，左側圖片預覽，右側結果文字。
  2. 支援單張圖片、本機影片、YouTube 連結。
  3. 使用 ffmpeg 每 2.5 秒抽幀，逐幀調用 clef-flash。
  4. 影片處理時：
     - 左側即時顯示當前分析的幀。
     - 右側即時顯示進度 (Analyzing frame N/M @ timestamp) 與該幀的可讀摘要。
  5. 提供 Paste 按鈕，一鍵從系統剪貼簿貼上 URL/路徑。
  6. 檔案選擇改用 pygame_gui 的內建 UIFileDialog（跨平台一致，不依賴 Tkinter）。
  7. 所有結果僅在 GUI 顯示，不寫入資料夾。
"""

import os
import sys
import base64
import shutil
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Optional

import pygame
import pygame_gui
import requests
from PIL import Image


def get_clipboard_text() -> str:
    """跨平台讀取系統剪貼簿純文字。"""
    try:
        if sys.platform == "darwin":  # macOS
            result = subprocess.run(["pbpaste"], capture_output=True, text=True)
            return result.stdout
        elif sys.platform == "win32":  # Windows
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True, text=True
            )
            return result.stdout
        else:  # Linux
            for cmd in (["xclip", "-selection", "clipboard", "-o"],
                        ["xsel", "--clipboard", "--output"]):
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        return result.stdout
                except FileNotFoundError:
                    continue
    except Exception:
        pass
    return ""


# ============================================================
# 配置
# ============================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "clef-flash"
FRAME_INTERVAL_SECONDS = 2.5
JPEG_QUALITY = 2

WIDTH, HEIGHT = 1200, 800
FPS = 30

BG = (30, 30, 30)
PANEL = (37, 37, 38)
TEXT = (204, 204, 204)
HIGHLIGHT = (76, 175, 80)
BORDER = (62, 62, 66)
ENTRY_BG = (62, 62, 66)
WHITE = (255, 255, 255)
GRAY = (136, 136, 136)

FONT_NAME = None

# ============================================================
# DEFAULT_QUESTIONS
# ============================================================

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
    "contains_people": {
        "type": "noul",
        "instructions": "Does this image contain any human beings (people, faces, human figures)?",
    },
    "contains_nature": {
        "type": "noul",
        "instructions": (
            "Does this image contain natural scenery or elements such as trees, mountains, "
            "sky, water, plants, animals, or landscapes (excluding purely indoor/urban scenes)?"
        ),
    },
    "contains_game": {
        "type": "noul",
        "instructions": (
            "Does this image show a video game, game UI, gameplay screenshot, or game-related content?"
        ),
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

# ============================================================
# 工具函數
# ============================================================

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def check_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None

def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")

def download_video_from_url(url: str, temp_dir: Path) -> str:
    if not check_yt_dlp():
        raise RuntimeError("未找到 yt-dlp，請先執行：pip install yt-dlp")
    output_template = str(temp_dir / "video.%(ext)s")
    cmd = ["yt-dlp", "-f", "mp4", "-o", output_template, url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp 下載失敗：\n{result.stderr[-1000:]}")
    for f in temp_dir.glob("video.*"):
        return str(f)
    raise RuntimeError("下載完成但找不到影片檔案。")

def get_video_duration(video_path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return None

def extract_frames(video_path: str, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / FRAME_INTERVAL_SECONDS
    output_pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", str(JPEG_QUALITY),
        "-y", output_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽幀失敗：\n{result.stderr[-1000:]}")
    frame_files = sorted(output_dir.glob("frame_*.jpg"))
    frames = []
    for idx, frame_file in enumerate(frame_files, start=1):
        ts = (idx - 1) * FRAME_INTERVAL_SECONDS
        h, m, s = int(ts // 3600), int((ts % 3600) // 60), ts % 60
        frames.append({
            "frame_path": str(frame_file),
            "frame_index": idx,
            "timestamp_seconds": round(ts, 2),
            "timestamp_hms": f"{h:02d}:{m:02d}:{s:06.3f}",
        })
    return frames

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_clef_flash_decision(image_path, state_text, questions, base_url=OLLAMA_BASE_URL, model=MODEL_NAME):
    image_b64 = encode_image_base64(image_path)
    url = f"{base_url.rstrip('/')}/v1/systemone"
    payload = {"model": model, "state": state_text, "images": [image_b64], "questions": questions}
    try:
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"無法連接到 Ollama（{base_url}）。請確認 Ollama 正在運行。")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API 返回錯誤：{e.response.status_code} - {e.response.text}")

def analyze_frame(frame_info, video_filename, base_url):
    state_text = (
        f"Video file: {video_filename}. Frame index: {frame_info['frame_index']}. "
        f"Timestamp: {frame_info['timestamp_hms']}. Analyze the image."
    )
    decision = call_clef_flash_decision(
        frame_info["frame_path"], state_text, DEFAULT_QUESTIONS, base_url
    )
    return {"meta": {**frame_info, "video_filename": video_filename}, "decision": decision}

# ---------- 人類可讀摘要 ----------

def _density_label(score: float) -> str:
    if score < 0.7:
        return "Low"
    elif score < 1.5:
        return "Medium"
    else:
        return "High"

def _noul_line(ans: dict) -> str:
    if not ans or "noul" not in ans:
        return "N/A"
    p = ans["noul"] * 100
    label = "YES" if p >= 50 else "NO"
    return f"{label} ({p:.1f}%)"

def summarize_frame(decision_result: dict) -> list[str]:
    answers = decision_result.get("answers", {}) or decision_result
    lines = []

    sc = answers.get("scene_category", {})
    if "choice" in sc:
        lines.append(f"Scene: {sc['choice']} (conf={sc.get('confidence', 0):.2f})")

    idn = answers.get("information_density", {})
    if "score" in idn:
        s = idn["score"]
        lines.append(f"Density: {_density_label(s)} (score={s:.2f})")

    lines.append(f"People: {_noul_line(answers.get('contains_people', {}))}")
    lines.append(f"Nature: {_noul_line(answers.get('contains_nature', {}))}")
    lines.append(f"Game:   {_noul_line(answers.get('contains_game', {}))}")
    lines.append(f"OCR:    {_noul_line(answers.get('needs_ocr_or_parsing', {}))}")

    act = answers.get("recommended_action", {})
    if "choice" in act:
        lines.append(f"Action: {act['choice']}")

    return lines


def format_results(decision_result: dict) -> str:
    out = "=== Decision Results ===\n\n"
    for q_name, q_def in DEFAULT_QUESTIONS.items():
        ans = decision_result.get("answers", {}).get(q_name, {})
        if not ans:
            ans = decision_result.get(q_name, {})
        out += f"【{q_name}】\n"
        if not ans:
            out += "  No data returned.\n\n"
            continue
        t = q_def["type"]
        if t == "choice":
            out += f"  ➜ Choice: {ans.get('choice', 'N/A')} (conf={ans.get('confidence', 0):.2f})\n"
            for k, v in ans.get("probabilities", {}).items():
                out += f"    - {k}: {v*100:.1f}%\n"
        elif t == "score":
            out += f"  ➜ Score: {ans.get('score', 0):.2f} (conf={ans.get('confidence', 0):.2f})\n"
        elif t == "noul":
            p = ans.get('noul', 0) * 100
            label = "YES" if p >= 50 else "NO"
            out += f"  ➜ Probability: {p:.1f}% ({label})\n"
        out += "-" * 40 + "\n\n"
    return out


# ============================================================
# Pygame GUI
# ============================================================

class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Clef-Flash Vision Decision Engine (10 Categories)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        # pygame_gui 管理器
        self.ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT))

        # 字型
        self.font_title = pygame.font.SysFont(FONT_NAME, 22, bold=True)
        self.font_sub = pygame.font.SysFont(FONT_NAME, 14)
        self.font_btn = pygame.font.SysFont(FONT_NAME, 14)
        self.font_text = pygame.font.SysFont("menlo,consolas,monospace", 13)

        # 狀態
        self.running = True
        self.is_processing = False
        self.temp_dir = None
        self.status_msg = "Welcome. Select an image or enter a video path / URL."
        self.result_lines = ["Waiting for input..."]
        self.preview_surface = None
        self.input_text = ""
        self.input_active = False

        # 檔案對話框狀態
        self.file_dialog = None
        self.file_dialog_mode = None  # "image" or "video"

        self.btn_rects = {}
        self._build_layout()
        self.pending_actions = []

    def _build_layout(self):
        pad = 20
        self.header_rect = pygame.Rect(pad, 10, WIDTH - pad * 2, 60)
        self.input_rect = pygame.Rect(pad, 80, WIDTH - pad * 2 - 160, 34)
        self.btn_analyze_rect = pygame.Rect(WIDTH - pad - 150, 80, 150, 34)

        self.btn_img_rect = pygame.Rect(pad, 124, 130, 32)
        self.btn_vid_rect = pygame.Rect(pad + 140, 124, 130, 32)
        self.btn_paste_rect = pygame.Rect(pad + 280, 124, 90, 32)
        self.btn_clear_rect = pygame.Rect(pad + 380, 124, 90, 32)

        panel_top = 170
        panel_h = HEIGHT - panel_top - pad
        half = (WIDTH - pad * 3) // 2
        self.left_rect = pygame.Rect(pad, panel_top, half, panel_h)
        self.right_rect = pygame.Rect(pad * 2 + half, panel_top, half, panel_h)

        self.btn_rects = {
            "analyze": self.btn_analyze_rect,
            "img": self.btn_img_rect,
            "vid": self.btn_vid_rect,
            "paste": self.btn_paste_rect,
            "clear": self.btn_clear_rect,
        }

    # ---------- 繪製 ----------

    def draw(self):
        self.screen.fill(BG)

        t = self.font_title.render(
            "Clef-Flash Vision Decision Engine (Decision-Native / Jev API)", True, HIGHLIGHT
        )
        self.screen.blit(t, (self.header_rect.x, self.header_rect.y))
        s = self.font_sub.render(
            "Drag and drop an image, or enter a video path / YouTube URL.", True, TEXT
        )
        self.screen.blit(s, (self.header_rect.x, self.header_rect.y + 30))

        # 輸入框
        pygame.draw.rect(self.screen, ENTRY_BG, self.input_rect, border_radius=4)
        if self.input_active:
            pygame.draw.rect(self.screen, HIGHLIGHT, self.input_rect, 2, border_radius=4)
        display_text = self.input_text if self.input_text else "e.g. ./video.mp4 or https://youtube.com/..."
        color = WHITE if self.input_text else GRAY
        txt_surf = self.font_text.render(display_text, True, color)
        self.screen.blit(txt_surf, (self.input_rect.x + 8, self.input_rect.y + 8))

        self._draw_button(self.btn_analyze_rect, "Analyze URL/Path", HIGHLIGHT, WHITE)
        self._draw_button(self.btn_img_rect, "Select Image", ENTRY_BG, WHITE)
        self._draw_button(self.btn_vid_rect, "Select Video", ENTRY_BG, WHITE)
        self._draw_button(self.btn_paste_rect, "Paste", ENTRY_BG, WHITE)
        self._draw_button(self.btn_clear_rect, "Clear", ENTRY_BG, WHITE)

        # 左側預覽
        pygame.draw.rect(self.screen, PANEL, self.left_rect, border_radius=4)
        pygame.draw.rect(self.screen, BORDER, self.left_rect, 1, border_radius=4)
        if self.preview_surface:
            pw, ph = self.preview_surface.get_size()
            px = self.left_rect.x + (self.left_rect.w - pw) // 2
            py = self.left_rect.y + (self.left_rect.h - ph) // 2
            self.screen.blit(self.preview_surface, (px, py))
        else:
            line1 = "Drop Image Here (PNG / JPG / WebP)"
            line2 = "or click 'Select Image'"
            surf1 = self.font_sub.render(line1, True, GRAY)
            surf2 = self.font_sub.render(line2, True, GRAY)
            line_h = self.font_sub.get_linesize()
            total = line_h * 2
            y = self.left_rect.y + (self.left_rect.h - total) // 2
            self.screen.blit(surf1, (self.left_rect.x + (self.left_rect.w - surf1.get_width()) // 2, y))
            self.screen.blit(surf2, (self.left_rect.x + (self.left_rect.w - surf2.get_width()) // 2, y + line_h))

        # 右側結果
        pygame.draw.rect(self.screen, PANEL, self.right_rect, border_radius=4)
        pygame.draw.rect(self.screen, BORDER, self.right_rect, 1, border_radius=4)
        self._draw_wrapped_text(
            self.result_lines,
            self.right_rect.x + 12, self.right_rect.y + 12,
            self.right_rect.w - 24, self.right_rect.h - 24,
        )

        # 底部狀態
        status = self.font_sub.render(self.status_msg, True, GRAY)
        self.screen.blit(status, (20, HEIGHT - 24))

        # 最後繪製 pygame_gui（檔案對話框會在最上層）
        self.ui_manager.draw_ui(self.screen)

        pygame.display.flip()

    def _draw_button(self, rect, label, bg, fg):
        color = bg if not self.is_processing else (80, 80, 80)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        txt = self.font_btn.render(label, True, fg)
        self.screen.blit(
            txt,
            (rect.x + (rect.w - txt.get_width()) // 2, rect.y + (rect.h - txt.get_height()) // 2),
        )

    def _draw_wrapped_text(self, lines, x, y, max_w, max_h):
        line_h = self.font_text.get_linesize()
        cur_y = y
        for line in lines:
            if cur_y + line_h > y + max_h:
                break
            if not line:
                cur_y += line_h
                continue
            words = line.split(" ")
            cur_line = ""
            for w in words:
                test = (cur_line + " " + w).strip()
                if self.font_text.size(test)[0] <= max_w:
                    cur_line = test
                else:
                    if cur_line:
                        surf = self.font_text.render(cur_line, True, TEXT)
                        self.screen.blit(surf, (x, cur_y))
                        cur_y += line_h
                    cur_line = w
            if cur_line:
                surf = self.font_text.render(cur_line, True, TEXT)
                self.screen.blit(surf, (x, cur_y))
                cur_y += line_h

    # ---------- 事件 ----------

    def handle_events(self, time_delta):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # 先讓 pygame_gui 處理（尤其是檔案對話框）
            self.ui_manager.process_events(event)

            # 處理 pygame_gui 的檔案對話框按鈕
            if event.type == pygame_gui.UI_BUTTON_PRESSED and self.file_dialog:
                if event.ui_element == self.file_dialog.ok_button:
                    selected = self.file_dialog.current_file_path
                    if selected:
                        if self.file_dialog_mode == "image":
                            self.load_image(selected)
                        elif self.file_dialog_mode == "video":
                            self.input_text = selected
                            if not check_ffmpeg():
                                self.status_msg = "ffmpeg not found."
                            else:
                                self.load_video(selected)
                    self._close_file_dialog()
                elif event.ui_element == self.file_dialog.cancel_button:
                    self._close_file_dialog()

            # 檔案對話框關閉事件
            if event.type == pygame_gui.UI_WINDOW_CLOSE and self.file_dialog:
                if event.ui_element == self.file_dialog:
                    self._close_file_dialog()

            # 主視窗的滑鼠點擊（若檔案對話框開啟則忽略）
            if event.type == pygame.MOUSEBUTTONDOWN and not self.file_dialog:
                self._on_click(event.pos)

            # 鍵盤輸入（若檔案對話框開啟則忽略）
            if event.type == pygame.KEYDOWN and not self.file_dialog:
                if self.input_active:
                    mods = pygame.key.get_mods()
                    is_cmd = bool(mods & pygame.KMOD_META)
                    is_ctrl = bool(mods & pygame.KMOD_CTRL)

                    if event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    elif event.key == pygame.K_RETURN:
                        self.analyze_from_input()
                    elif event.key == pygame.K_v and (is_cmd or is_ctrl):
                        pasted = get_clipboard_text().strip()
                        if pasted:
                            self.input_text += pasted
                            self.status_msg = "Pasted from clipboard."
                        else:
                            self.status_msg = "Clipboard is empty."
                    elif event.key == pygame.K_a and (is_cmd or is_ctrl):
                        self.status_msg = "Press Cmd/Ctrl+V to paste a URL."
                    elif event.unicode.isprintable():
                        self.input_text += event.unicode
                else:
                    if event.key == pygame.K_RETURN:
                        self.input_active = True

    def _on_click(self, pos):
        if self.input_rect.collidepoint(pos):
            self.input_active = True
            return
        self.input_active = False

        if self.is_processing:
            self.status_msg = "Processing, please wait..."
            return

        if self.btn_analyze_rect.collidepoint(pos):
            self.analyze_from_input()
        elif self.btn_img_rect.collidepoint(pos):
            self.open_image_dialog()
        elif self.btn_vid_rect.collidepoint(pos):
            self.open_video_dialog()
        elif self.btn_paste_rect.collidepoint(pos):
            self.paste_from_clipboard()
        elif self.btn_clear_rect.collidepoint(pos):
            self.clear_all()

    # ---------- 檔案對話框 ----------

    def _open_file_dialog(self, mode: str, title: str):
        if self.file_dialog:
            return
        self.file_dialog_mode = mode
        self.file_dialog = pygame_gui.windows.UIFileDialog(
            rect=pygame.Rect(200, 100, 800, 600),
            manager=self.ui_manager,
            window_title=title,
            initial_file_path=os.path.expanduser("~"),
            allow_picking_directories=False,
        )

    def _close_file_dialog(self):
        if self.file_dialog:
            self.file_dialog.kill()
            self.file_dialog = None
            self.file_dialog_mode = None

    def open_image_dialog(self):
        self._open_file_dialog("image", "Select Image")

    def open_video_dialog(self):
        self._open_file_dialog("video", "Select Video")

    # ---------- 動作 ----------

    def paste_from_clipboard(self):
        pasted = get_clipboard_text().strip()
        if pasted:
            self.input_text = pasted
            self.input_active = True
            self.status_msg = "Pasted from clipboard."
        else:
            self.status_msg = "Clipboard is empty."

    def analyze_from_input(self):
        text = self.input_text.strip()
        if not text:
            self.status_msg = "Please enter a video path or URL."
            return
        if is_url(text):
            if not check_ffmpeg() or not check_yt_dlp():
                self.status_msg = "Need ffmpeg and yt-dlp for URL."
                return
            self.load_video_from_url(text)
        else:
            if not os.path.isfile(text):
                self.status_msg = f"File not found: {text}"
                return
            if not check_ffmpeg():
                self.status_msg = "ffmpeg not found."
                return
            self.load_video(text)

    def load_image(self, path):
        self.status_msg = "Analyzing image..."
        self.is_processing = True
        self.result_lines = ["Analyzing image with Clef-Flash..."]
        self._load_preview(path)
        threading.Thread(target=self._image_thread, args=(path,), daemon=True).start()

    def _image_thread(self, path):
        try:
            decision = call_clef_flash_decision(
                path,
                f"Image file: {os.path.basename(path)}. Analyze.",
                DEFAULT_QUESTIONS,
            )
            text = format_results(decision)
            self.pending_actions.append(("result", text.splitlines()))
        except Exception as e:
            self.pending_actions.append(("result", [f"Error: {e}"]))
        finally:
            self.pending_actions.append(("finish", None))

    def load_video(self, path):
        self.status_msg = "Processing video..."
        self.is_processing = True
        self.result_lines = ["Extracting frames..."]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="clef_flash_"))
        frames_dir = self.temp_dir / "frames"
        threading.Thread(
            target=self._video_thread,
            args=(path, os.path.basename(path), frames_dir),
            daemon=True,
        ).start()

    def load_video_from_url(self, url):
        self.status_msg = "Downloading video..."
        self.is_processing = True
        self.result_lines = ["Downloading from URL..."]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="clef_flash_"))
        threading.Thread(target=self._url_thread, args=(url,), daemon=True).start()

    def _url_thread(self, url):
        try:
            path = download_video_from_url(url, self.temp_dir)
            frames_dir = self.temp_dir / "frames"
            self.pending_actions.append(("result", ["Video downloaded. Extracting..."]))
            self._run_video_analysis(path, os.path.basename(path), frames_dir)
        except Exception as e:
            self.pending_actions.append(("result", [f"Error: {e}"]))
            self.pending_actions.append(("finish", None))
            self._cleanup_temp()

    def _video_thread(self, path, filename, frames_dir):
        try:
            self._run_video_analysis(path, filename, frames_dir)
        except Exception as e:
            self.pending_actions.append(("result", [f"Error: {e}"]))
            self.pending_actions.append(("finish", None))
            self._cleanup_temp()

    def _run_video_analysis(self, path, filename, frames_dir):
        frames = extract_frames(path, frames_dir)
        if not frames:
            raise RuntimeError("No frames extracted.")

        total = len(frames)
        frame_summaries = []

        for i, fi in enumerate(frames):
            ts = fi["timestamp_hms"]
            progress = f"Analyzing frame {i+1}/{total} @ {ts}..."

            self.pending_actions.append(("preview", fi["frame_path"]))
            self.pending_actions.append(("result", [progress]))

            try:
                enriched = analyze_frame(fi, filename, OLLAMA_BASE_URL)
                dec = enriched["decision"]
                summary_lines = summarize_frame(dec)
                frame_summaries.append((fi, summary_lines))

                live_block = [progress, "—" * 30] + summary_lines
                self.pending_actions.append(("result", live_block))

            except Exception as e:
                err_lines = [progress, f"  Error: {e}"]
                frame_summaries.append((fi, err_lines))
                self.pending_actions.append(("result", err_lines))

        lines = [
            "=== Video Analysis Complete ===",
            f"Video: {filename}",
            f"Frames: {total}",
            "",
        ]
        for fi, summary in frame_summaries:
            lines.append(f"Frame {fi['frame_index']} @ {fi['timestamp_hms']}")
            for s in summary:
                lines.append(f"  {s}")
            lines.append("")

        self.pending_actions.append(("result", lines))
        self.pending_actions.append(("finish", None))
        self._cleanup_temp()

    # ---------- 輔助 ----------

    def _load_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((520, 520))
            surf = pygame.image.fromstring(img.convert("RGB").tobytes(), img.size, "RGB")
            self.preview_surface = surf
        except Exception as e:
            self.status_msg = f"Preview error: {e}"

    def _cleanup_temp(self):
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
            self.temp_dir = None

    def clear_all(self):
        if self.is_processing:
            self.status_msg = "Processing, cannot clear."
            return
        self.input_text = ""
        self.preview_surface = None
        self.result_lines = ["Waiting for input..."]
        self.status_msg = "Cleared."

    def process_pending(self):
        while self.pending_actions:
            action, data = self.pending_actions.pop(0)
            if action == "result":
                self.result_lines = data
            elif action == "preview":
                self._load_preview(data)
            elif action == "finish":
                self.is_processing = False
                self.status_msg = "Done."

    def run(self):
        while self.running:
            time_delta = self.clock.tick(FPS) / 1000.0
            self.handle_events(time_delta)
            self.ui_manager.update(time_delta)
            self.process_pending()
            self.draw()
        pygame.quit()


if __name__ == "__main__":
    App().run()