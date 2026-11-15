#!/usr/bin/env python3
"""
Drag-and-Drop Vision Decision Tool
Powered by clef-flash (Decision-Native / Jev / System One API)
All-English UI & 10-Category Classification
"""

import os

# === Put your remote Ollama host config here (MUST be before `import ollama`) ===
# REMOTE_OLLAMA_HOST = "http://192.168.1.28:11434"  # <-- 改成你的遠端 IP
REMOTE_OLLAMA_HOST = "http://127.0.0.1:11434"  # <-- 改成你的遠端 IP
os.environ["OLLAMA_HOST"] = REMOTE_OLLAMA_HOST

import pygame
import ollama
import cv2
import numpy as np
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SCREEN_WIDTH = 1020
SCREEN_HEIGHT = 680

BG_COLOR = (18, 18, 24)
PANEL_COLOR = (28, 29, 38)
ACCENT_COLOR = (80, 200, 120)
TEXT_COLOR = (235, 235, 240)
MUTED_COLOR = (140, 145, 160)
WARN_COLOR = (230, 160, 60)
BORDER_COLOR = (60, 65, 85)
HOVER_COLOR = (110, 220, 150)

# Folder shipped with the sprint that holds the 10-category sample images.
EXAMPLE_DIR_NAME = "PyConHK2026_Sprint"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".gif", ".tif", ".tiff", ".ppm", ".pgm",
}

# Runs in a separate interpreter so the native file dialog never blocks the
# pygame loop (and never shares an event loop / thread with SDL).
FILE_DIALOG_SCRIPT = r"""
import sys
import tkinter as tk
from tkinter import filedialog

initialdir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None

root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
root.update_idletasks()

options = dict(
    parent=root,
    title="Select an image for Clef-Flash",
    filetypes=[
        ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff"),
        ("All files", "*.*"),
    ],
)
if initialdir:
    options["initialdir"] = initialdir

path = filedialog.askopenfilename(**options)
root.destroy()
if path:
    sys.stdout.write(path)
"""


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

    script_dir = Path(__file__).resolve().parent
    for parent in [script_dir, *script_dir.parents]:
        candidates.append(parent / EXAMPLE_DIR_NAME)

    candidates.append(Path.cwd() / EXAMPLE_DIR_NAME)

    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            continue
    return None

# 10 Clef Choice Categories + Score + Noul
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
            "illustration_or_graphic": "Digital artwork, logo, icon, poster, or meme"
        }
    },
    "information_density": {
        "type": "score",
        "instructions": "Rate the visual information and data density of this image.",
        "criteria": [
            "Low density (simple graphic, icon, or sparse layout)",
            "Medium density (standard chart, slide, or UI view)",
            "High density (dense text, complex plot, or detailed code)"
        ]
    },
    "needs_ocr_or_parsing": {
        "type": "noul",
        "instructions": "Does this image contain structured data, charts, or text that should be parsed?"
    },
    "recommended_action": {
        "type": "choice",
        "instructions": "Select the best automated pipeline action for this image.",
        "criteria": {
            "parse_chart_metrics": "Extract data series, labels, and metrics from chart/plot",
            "extract_text_ocr": "Run OCR to extract text, code, or document body",
            "analyze_ui_state": "Inspect interface layout or debug terminal/UI state",
            "archive_media": "Tag and index in visual asset library"
        }
    }
}


class VisionDecisionApp:
    def __init__(self, example_dir=None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Clef-Flash Vision Decision Engine (10 Categories)")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("arial,helvetica,sans-serif", 22, bold=True)
        self.font_body = pygame.font.SysFont("arial,helvetica,sans-serif", 15, bold=True)
        self.font_small = pygame.font.SysFont("menlo,consolas,monospace", 12)

        # Where the file browser starts: the bundled sample-image folder.
        self.example_dir = resolve_example_dir(example_dir)

        # Left panel doubles as the drop zone and the click-to-upload target.
        self.drop_rect = pygame.Rect(24, 78, 430, 575)
        self.drop_hover = False

        self.image_surface = None
        self.image_path = None
        self.busy = False
        self.status_msg = "Drag and drop an image here, or click the panel to browse..."
        self.result_data = None
        self.latency_ms = 0

        # Async file-picker state (see open_file_dialog / _poll_dialog).
        self.dialog_open = False
        self.dialog_done = False
        self.dialog_result = None

    # ---------------------------------------------------------------- input ---
    def _is_image_path(self, path_str: str) -> bool:
        try:
            path = Path(path_str)
        except (TypeError, ValueError):
            return False
        if not path.is_file():
            return False
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return True
        # No/unknown suffix: still try it, pygame can sniff the header.
        return path.suffix == ""

    def _pick_initial_dir(self) -> str:
        """Prefer the folder of the last image, else the bundled examples."""
        if self.image_path:
            parent = Path(self.image_path).parent
            if parent.is_dir():
                return str(parent)
        if self.example_dir:
            return str(self.example_dir)
        return str(Path.home())

    def open_file_dialog(self):
        """Open a native 'choose a file' dialog without freezing the window.

        The picker runs in a worker thread (Tk in a throwaway subprocess), and
        the main loop polls `_poll_dialog()` each frame for the result.
        """
        if self.busy:
            self.status_msg = "Analysis in progress - pick another image once it finishes."
            return
        if self.dialog_open:
            return

        self.dialog_open = True
        self.dialog_result = None
        self.dialog_done = False
        self.status_msg = "File browser open - choose an image (PNG / JPG / WebP)..."

        initial = self._pick_initial_dir()

        def worker():
            chosen = self._run_picker(initial)
            self.dialog_result = chosen
            self.dialog_done = True

        threading.Thread(target=worker, daemon=True).start()

    def _poll_dialog(self):
        """Called from the main loop to consume a finished file pick."""
        if not (self.dialog_open and self.dialog_done):
            return
        self.dialog_open = False
        chosen = self.dialog_result
        self.dialog_result = None
        self.dialog_done = False

        if chosen and self._is_image_path(chosen):
            self.analyze_image(chosen)
        elif chosen:
            self.status_msg = f"Not a supported image file: {Path(chosen).name}"
        else:
            self.status_msg = "No file selected - drop an image or click the panel to browse."

    def _run_picker(self, initial: str) -> str | None:
        """Blocking picker call; only ever runs on the dialog worker thread."""
        chosen = None
        tmp_path = None

        # Preferred path: Tk in a throwaway process -> truly modal native dialog.
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="clef_picker_", suffix=".py")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(FILE_DIALOG_SCRIPT)
            result = subprocess.run(
                [sys.executable, tmp_path, initial],
                capture_output=True,
                text=True,
                timeout=180,
            )
            chosen = (result.stdout or "").strip() or None
        except Exception:
            chosen = None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Fallback: OS-native picker if Tk is unavailable.
        if chosen is None:
            chosen = self._fallback_native_pick(initial)

        return chosen

    def _fallback_native_pick(self, initial_dir: str) -> str | None:
        """macOS / Linux native pickers via the shell, used if Tk is missing."""
        system = platform.system()
        try:
            if system == "Darwin":
                result = subprocess.run(
                    [
                        "osascript",
                        "-e", "on run argv",
                        "-e", "set p to POSIX path of (choose file with prompt "
                              '"Select an image for Clef-Flash" default location '
                              "(POSIX file (item 1 of argv)) of type "
                              '{"public.image"})',
                        "-e", "return p",
                        "-e", "end run",
                        str(initial_dir) + "/",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                return (result.stdout or "").strip() or None
            if system == "Linux" and shutil.which("zenity"):
                result = subprocess.run(
                    ["zenity", "--file-selection", "--filename", str(initial_dir) + "/"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                return (result.stdout or "").strip() or None
        except Exception:
            return None
        return None

    def _update_hover(self, pos):
        """Single source of truth for hover state + cursor shape."""
        hovered = self.drop_rect.collidepoint(pos)
        if hovered == self.drop_hover:
            return
        self.drop_hover = hovered
        pygame.mouse.set_cursor(
            pygame.SYSTEM_CURSOR_HAND if hovered else pygame.SYSTEM_CURSOR_ARROW
        )

    def _handle_click(self, pos):
        if self.drop_rect.collidepoint(pos):
            self.open_file_dialog()

    # ------------------------------------------------------------- analysis ---
    def _extract_visual_state(self, file_path: str) -> dict:
        """Extract rich structural and visual metrics via OpenCV for ollama.systemone."""
        path_obj = Path(file_path)
        img = cv2.imread(file_path)
        if img is None:
            return {"filename": path_obj.name, "error": "unreadable_image"}

        h, w, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        mean_bgr = img.mean(axis=(0, 1))
        std_bgr = img.std(axis=(0, 1))
        brightness = float(gray.mean())
        edge_density = float(np.count_nonzero(edges) / (h * w))

        # Check ratio of very light pixels (common in charts, documents, slides)
        white_pixel_ratio = float(np.sum(gray > 235) / (h * w))
        # Check ratio of very dark pixels (common in terminals, dark IDEs)
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
                "b": round(float(mean_bgr[0]), 1)
            },
            "edge_complexity_ratio": round(edge_density, 4),
            "white_background_ratio": round(white_pixel_ratio, 3),
            "dark_background_ratio": round(dark_pixel_ratio, 3),
            "layout_theme": (
                "light_background_chart_or_doc" if white_pixel_ratio > 0.45
                else ("dark_theme_terminal_or_ui" if dark_pixel_ratio > 0.45 else "mixed_color_scene")
            )
        }

    def analyze_image(self, file_path: str):
        if self.busy:
            return
        self.busy = True
        self.image_path = file_path
        self.status_msg = f"Analyzing with clef-flash: {Path(file_path).name} ..."
        # Clear the previous verdict so a new pick never shows stale results.
        self.result_data = None
        self.image_surface = None

        try:
            loaded = pygame.image.load(file_path)
            iw, ih = loaded.get_size()
            scale = min(410 / iw, 480 / ih)
            new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
            self.image_surface = pygame.transform.smoothscale(loaded, new_size)
        except Exception as e:
            self.status_msg = f"Failed to load image: {e}"
            self.busy = False
            return

        def worker():
            t0 = time.time()
            try:
                state_payload = self._extract_visual_state(file_path)

                # Call ollama.systemone WITHOUT the unsupported 'images' keyword argument
                resp = ollama.systemone(
                    model="clef-flash",
                    state=state_payload,
                    questions=DEFAULT_QUESTIONS,
                    keep_alive="30m"
                )

                if hasattr(resp, "model_dump"):
                    self.result_data = resp.model_dump()
                elif isinstance(resp, dict):
                    self.result_data = resp
                else:
                    self.result_data = json.loads(resp.json())

                self.latency_ms = int((time.time() - t0) * 1000)
                self.status_msg = (
                    f"Decision complete ({self.latency_ms} ms) - "
                    "drop or click for the next image"
                )
            except Exception as e:
                self.status_msg = f"AI Decision Error: {e}"
            finally:
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    def _draw_drop_hint(self, rect, border_color):
        """Empty-state art: dashed border + drop/click instructions."""
        # Dashed inner border to read as a drop zone.
        dash_color = border_color
        step = 12
        for x in range(rect.left + 16, rect.right - 16, step * 2):
            pygame.draw.line(self.screen, dash_color, (x, rect.top + 16), (x + step, rect.top + 16), 2)
            pygame.draw.line(self.screen, dash_color, (x, rect.bottom - 16), (x + step, rect.bottom - 16), 2)
        for y in range(rect.top + 16, rect.bottom - 16, step * 2):
            pygame.draw.line(self.screen, dash_color, (rect.left + 16, y), (rect.left + 16, y + step), 2)
            pygame.draw.line(self.screen, dash_color, (rect.right - 16, y), (rect.right - 16, y + step), 2)

        cx, cy = rect.centerx, rect.centery

        # Stacked "photos" glyph above the text.
        pygame.draw.rect(self.screen, MUTED_COLOR, pygame.Rect(cx - 26, cy - 78, 52, 40), width=2, border_radius=4)
        pygame.draw.rect(self.screen, border_color, pygame.Rect(cx - 34, cy - 88, 52, 40), width=2, border_radius=4)
        pygame.draw.circle(self.screen, MUTED_COLOR, (cx - 18, cy - 74), 4)
        pygame.draw.polygon(
            self.screen, MUTED_COLOR,
            [(cx - 32, cy - 52), (cx - 14, cy - 68), (cx + 2, cy - 52)],
        )

        line1 = self.font_body.render("Drag & drop an image here", True, TEXT_COLOR)
        line2 = self.font_body.render("or click anywhere in this panel", True, ACCENT_COLOR if self.drop_hover else MUTED_COLOR)
        line3 = self.font_small.render("PNG  /  JPG  /  WebP  /  BMP", True, MUTED_COLOR)

        self.screen.blit(line1, (cx - line1.get_width() // 2, cy - 18))
        self.screen.blit(line2, (cx - line2.get_width() // 2, cy + 6))
        self.screen.blit(line3, (cx - line3.get_width() // 2, cy + 32))

        # Tell the user where the picker will open.
        if self.example_dir:
            folder = self.example_dir.name
            line4 = self.font_small.render(f"Browser opens in: {folder}/", True, MUTED_COLOR)
            self.screen.blit(line4, (cx - line4.get_width() // 2, rect.bottom - 42))

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Header & status bar
        title = self.font_title.render("Clef-Flash Vision Decision Engine (Decision-Native / Jev API)", True, ACCENT_COLOR)
        self.screen.blit(title, (24, 16))
        status_color = WARN_COLOR if self.busy else MUTED_COLOR
        status = self.font_body.render(self.status_msg, True, status_color)
        self.screen.blit(status, (24, 46))

        # Left Panel: Image Drop Zone & Preview (also click-to-browse)
        left_rect = self.drop_rect
        border_color = HOVER_COLOR if self.drop_hover else BORDER_COLOR
        pygame.draw.rect(self.screen, PANEL_COLOR, left_rect, border_radius=10)
        pygame.draw.rect(self.screen, border_color, left_rect, width=2, border_radius=10)

        if self.image_surface:
            ix = left_rect.centerx - self.image_surface.get_width() // 2
            iy = left_rect.centery - self.image_surface.get_height() // 2
            self.screen.blit(self.image_surface, (ix, iy))

            if self.drop_hover:
                overlay = pygame.Surface(left_rect.size, pygame.SRCALPHA)
                overlay.fill((80, 200, 120, 38))
                self.screen.blit(overlay, (left_rect.x, left_rect.y))
                swap = self.font_body.render("Click to choose another image", True, TEXT_COLOR)
                self.screen.blit(
                    swap,
                    (left_rect.x + 12, left_rect.bottom - swap.get_height() - 12),
                )
        else:
            self._draw_drop_hint(left_rect, border_color)

        # Right Panel: Jev / SystemOne Decision Output
        right_rect = pygame.Rect(474, 78, 522, 575)
        pygame.draw.rect(self.screen, PANEL_COLOR, right_rect, border_radius=10)

        y_offset = 92
        if self.result_data and "answers" in self.result_data:
            answers = self.result_data["answers"]
            for q_key, ans in answers.items():
                q_type = ans.get("type") if isinstance(ans, dict) else "noul"
                header_str = f"[{str(q_type).upper()}] {q_key}"
                self.screen.blit(self.font_body.render(header_str, True, ACCENT_COLOR), (492, y_offset))
                y_offset += 20

                if isinstance(ans, (int, float)) or q_type == "noul":
                    prob = ans if isinstance(ans, (int, float)) else ans.get("noul", ans.get("probability", 0.0))
                    verdict = "TRUE (Yes)" if prob >= 0.5 else "FALSE (No)"
                    line = f"  -> Verdict: {verdict}  |  Prob (noul): {prob:.4f}"
                    self.screen.blit(self.font_small.render(line, True, TEXT_COLOR), (492, y_offset))
                    y_offset += 22

                elif isinstance(ans, dict) and ("choice" in ans or q_type == "choice"):
                    choice = ans.get("choice")
                    conf = ans.get("confidence", 0.0)
                    line = f"  -> Choice: {choice}  (Confidence: {conf:.3f})"
                    self.screen.blit(self.font_small.render(line, True, TEXT_COLOR), (492, y_offset))
                    y_offset += 18
                    probs = ans.get("probabilities", {})
                    # Sort probabilities highest-first so all 10 choices are easy to read
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    for k, p in sorted_probs:
                        bar = "#" * int(p * 18)
                        p_line = f"     {k:<24}: {p:.3f} {bar}"
                        color = TEXT_COLOR if k == choice else MUTED_COLOR
                        self.screen.blit(self.font_small.render(p_line, True, color), (492, y_offset))
                        y_offset += 15
                    y_offset += 8

                elif isinstance(ans, dict) and ("score" in ans or q_type == "score"):
                    score = ans.get("score", 0.0)
                    conf = ans.get("confidence", 0.0)
                    line = f"  -> Score: {score:.3f}  (Confidence: {conf:.3f})"
                    self.screen.blit(self.font_small.render(line, True, TEXT_COLOR), (492, y_offset))
                    y_offset += 18
                    probs = ans.get("probabilities", {})
                    legend = ans.get("legend", {})
                    for idx_str, p in probs.items():
                        lbl = legend.get(str(idx_str), f"Level {idx_str}")
                        p_line = f"     [{idx_str}] {lbl[:28]:<28}: {p:.3f}"
                        self.screen.blit(self.font_small.render(p_line, True, MUTED_COLOR), (492, y_offset))
                        y_offset += 15
                    y_offset += 8
        else:
            empty = self.font_body.render("Waiting for image input to compute probabilities...", True, MUTED_COLOR)
            self.screen.blit(empty, (492, 105))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.DROPFILE:
                    # A drop supersedes any picker that is still open.
                    self.dialog_open = False
                    self.dialog_done = False
                    self.dialog_result = None
                    if self._is_image_path(event.file):
                        self.analyze_image(event.file)
                    else:
                        self.status_msg = (
                            f"Unsupported file: {Path(event.file).name} "
                            "(PNG / JPG / WebP only)"
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self._update_hover(event.pos)

            self._poll_dialog()
            self.draw()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    cli_example_dir = sys.argv[1] if len(sys.argv) > 1 else None
    app = VisionDecisionApp(example_dir=cli_example_dir)
    app.run()
