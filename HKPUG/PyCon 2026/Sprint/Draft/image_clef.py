#!/usr/bin/env python3
"""
Drag-and-Drop Vision Decision Tool
Powered by clef-flash (Decision-Native / Jev / System One API)
All-English UI & 10-Category Classification
"""

import pygame
import ollama
import cv2
import numpy as np
import json
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
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Clef-Flash Vision Decision Engine (10 Categories)")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("arial,helvetica,sans-serif", 22, bold=True)
        self.font_body = pygame.font.SysFont("arial,helvetica,sans-serif", 15, bold=True)
        self.font_small = pygame.font.SysFont("menlo,consolas,monospace", 12)

        self.image_surface = None
        self.image_path = None
        self.busy = False
        self.status_msg = "Drag and drop an image (PNG/JPG/WebP) into this window..."
        self.result_data = None
        self.latency_ms = 0

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
                self.status_msg = f"Decision complete ({self.latency_ms} ms) - Drop another image anytime"
            except Exception as e:
                self.status_msg = f"AI Decision Error: {e}"
            finally:
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Header & status bar
        title = self.font_title.render("Clef-Flash Vision Decision Engine (Decision-Native / Jev API)", True, ACCENT_COLOR)
        self.screen.blit(title, (24, 16))
        status_color = WARN_COLOR if self.busy else MUTED_COLOR
        status = self.font_body.render(self.status_msg, True, status_color)
        self.screen.blit(status, (24, 46))

        # Left Panel: Image Drop Zone & Preview
        left_rect = pygame.Rect(24, 78, 430, 575)
        pygame.draw.rect(self.screen, PANEL_COLOR, left_rect, border_radius=10)
        pygame.draw.rect(self.screen, (60, 65, 85), left_rect, width=2, border_radius=10)

        if self.image_surface:
            ix = left_rect.centerx - self.image_surface.get_width() // 2
            iy = left_rect.centery - self.image_surface.get_height() // 2
            self.screen.blit(self.image_surface, (ix, iy))
        else:
            hint = self.font_body.render("Drop Image Here (PNG / JPG / WebP)", True, MUTED_COLOR)
            self.screen.blit(hint, (left_rect.centerx - hint.get_width() // 2, left_rect.centery - 10))

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
                    self.analyze_image(event.file)

            self.draw()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    app = VisionDecisionApp()
    app.run()
