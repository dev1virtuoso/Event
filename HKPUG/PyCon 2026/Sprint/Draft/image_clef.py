#!/usr/bin/env python3
"""
Drag-and-Drop Vision Decision Tool
Powered by clef-flash (Decision-Native / Jev / System One API)
"""

import pygame
import ollama
import base64
import json
import threading
import time
import urllib.request
from pathlib import Path

SCREEN_WIDTH = 960
SCREEN_HEIGHT = 640

BG_COLOR = (18, 18, 24)
PANEL_COLOR = (28, 29, 38)
ACCENT_COLOR = (80, 200, 120)
TEXT_COLOR = (235, 235, 240)
MUTED_COLOR = (140, 145, 160)
WARN_COLOR = (230, 160, 60)

# Default Jev / System One decision schema
DEFAULT_QUESTIONS = {
    "scene_category": {
        "type": "choice",
        "instructions": "Classify the primary visual content of this image.",
        "criteria": {
            "ui_screenshot": "Software interface, terminal, code, or webpage screenshot",
            "document": "Slide deck, receipt, form, diagram, or text document",
            "photo": "Real-world photography of people, objects, or nature",
            "graphic": "Illustration, meme, logo, or generated artwork"
        }
    },
    "information_density": {
        "type": "score",
        "instructions": "Rate how much readable text or structured data is present in the image.",
        "criteria": [
            "Minimal or no text",
            "Moderate labels or short snippets",
            "Dense text, code blocks, or detailed tables"
        ]
    },
    "has_error_or_warning": {
        "type": "noul",
        "instructions": "Does the image show an error message, warning, crash, or bug?"
    },
    "actionable_next_step": {
        "type": "choice",
        "instructions": "What is the most appropriate automated action for this image?",
        "criteria": {
            "extract_text_ocr": "Extract text or code snippets",
            "debug_issue": "Analyze and troubleshoot an error",
            "archive_media": "Tag and store in media library",
            "human_review": "Escalate for manual inspection"
        }
    }
}


class VisionDecisionApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Clef-Flash Vision Decision Analyzer (Drag & Drop Image)")
        self.clock = pygame.time.Clock()

        # Standard system fonts for clean English rendering
        self.font_title = pygame.font.SysFont("arial,helvetica,sans-serif", 22, bold=True)
        self.font_body = pygame.font.SysFont("arial,helvetica,sans-serif", 15)
        self.font_small = pygame.font.SysFont("menlo,consolas,monospace", 13)

        self.image_surface = None
        self.image_path = None
        self.busy = False
        self.status_msg = "Drag and drop an image (PNG/JPG/WebP) into this window..."
        self.result_data = None
        self.latency_ms = 0

    def _encode_image_b64(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_image(self, file_path: str):
        if self.busy:
            return
        self.busy = True
        self.image_path = file_path
        self.status_msg = f"Analyzing with clef-flash: {Path(file_path).name} ..."

        # Load and scale image preview
        try:
            loaded = pygame.image.load(file_path)
            iw, ih = loaded.get_size()
            scale = min(400 / iw, 420 / ih)
            new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
            self.image_surface = pygame.transform.smoothscale(loaded, new_size)
        except Exception as e:
            self.status_msg = f"Failed to load image: {e}"
            self.busy = False
            return

        def worker():
            t0 = time.time()
            try:
                b64_img = self._encode_image_b64(file_path)
                state_payload = {
                    "filename": Path(file_path).name,
                    "task": "Evaluate visual state and return typed SystemOne decisions."
                }

                # Use ollama.systemone if available, otherwise fallback to HTTP POST /v1/systemone
                if hasattr(ollama, "systemone"):
                    resp = ollama.systemone(
                        model="clef-flash",
                        state=state_payload,
                        images=[b64_img],
                        questions=DEFAULT_QUESTIONS,
                        keep_alive="30m"
                    )
                    if hasattr(resp, "model_dump"):
                        self.result_data = resp.model_dump()
                    elif isinstance(resp, dict):
                        self.result_data = resp
                    else:
                        self.result_data = json.loads(resp.json())
                else:
                    req_body = json.dumps({
                        "model": "clef-flash",
                        "state": state_payload,
                        "images": [b64_img],
                        "questions": DEFAULT_QUESTIONS,
                        "keep_alive": "30m"
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        "http://localhost:11434/v1/systemone",
                        data=req_body,
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req) as r:
                        self.result_data = json.loads(r.read().decode("utf-8"))

                self.latency_ms = int((time.time() - t0) * 1000)
                self.status_msg = f"Analysis complete ({self.latency_ms} ms) - Drop another image anytime"
            except Exception as e:
                self.status_msg = f"AI Decision Error: {e}"
            finally:
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Header & status bar
        title = self.font_title.render("Clef-Flash Vision Decision Engine (Decision-Native / Jev API)", True, ACCENT_COLOR)
        self.screen.blit(title, (24, 18))
        status_color = WARN_COLOR if self.busy else MUTED_COLOR
        status = self.font_body.render(self.status_msg, True, status_color)
        self.screen.blit(status, (24, 50))

        # Left Panel: Image Drop Zone & Preview
        left_rect = pygame.Rect(24, 85, 420, 530)
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
        right_rect = pygame.Rect(464, 85, 472, 530)
        pygame.draw.rect(self.screen, PANEL_COLOR, right_rect, border_radius=10)

        y_offset = 100
        if self.result_data and "answers" in self.result_data:
            answers = self.result_data["answers"]
            for q_key, ans in answers.items():
                q_type = ans.get("type") if isinstance(ans, dict) else "noul"
                header_str = f"[{str(q_type).upper()}] {q_key}"
                self.screen.blit(self.font_body.render(header_str, True, ACCENT_COLOR), (484, y_offset))
                y_offset += 24

                if isinstance(ans, (int, float)) or q_type == "noul":
                    prob = ans if isinstance(ans, (int, float)) else ans.get("noul", ans.get("probability", 0.0))
                    verdict = "TRUE (Yes)" if prob >= 0.5 else "FALSE (No)"
                    line = f"  -> Verdict: {verdict}  |  Prob (noul): {prob:.4f}"
                    self.screen.blit(self.font_small.render(line, True, TEXT_COLOR), (484, y_offset))
                    y_offset += 26

                elif isinstance(ans, dict) and ("choice" in ans or q_type == "choice"):
                    choice = ans.get("choice")
                    conf = ans.get("confidence", 0.0)
                    line = f"  -> Choice: {choice}  (Confidence: {conf:.3f})"
                    self.screen.blit(self.font_small.render(line, True, TEXT_COLOR), (484, y_offset))
                    y_offset += 20
                    probs = ans.get("probabilities", {})
                    for k, p in probs.items():
                        bar = "#" * int(p * 15)
                        p_line = f"     {k:<18}: {p:.3f} {bar}"
                        self.screen.blit(self.font_small.render(p_line, True, MUTED_COLOR), (484, y_offset))
                        y_offset += 18
                    y_offset += 10

                elif isinstance(ans, dict) and ("score" in ans or q_type == "score"):
                    score = ans.get("score", 0.0)
                    conf = ans.get("confidence", 0.0)
                    line = f"  -> Score: {score:.3f}  (Confidence: {conf:.3f})"
                    self.screen.blit(self.font_small.render(line, True, TEXT_COLOR), (484, y_offset))
                    y_offset += 20
                    probs = ans.get("probabilities", {})
                    legend = ans.get("legend", {})
                    for idx_str, p in probs.items():
                        lbl = legend.get(str(idx_str), f"Level {idx_str}")
                        p_line = f"     [{idx_str}] {lbl[:22]:<22}: {p:.3f}"
                        self.screen.blit(self.font_small.render(p_line, True, MUTED_COLOR), (484, y_offset))
                        y_offset += 18
                    y_offset += 10
        else:
            empty = self.font_body.render("Waiting for image input to compute probabilities...", True, MUTED_COLOR)
            self.screen.blit(empty, (484, 110))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.DROPFILE:
                    # Triggered when an image file is dragged and dropped into the window
                    self.analyze_image(event.file)

            self.draw()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    app = VisionDecisionApp()
    app.run()
