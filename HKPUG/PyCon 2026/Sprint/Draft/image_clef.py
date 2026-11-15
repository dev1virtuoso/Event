#!/usr/bin/env python3
"""
圖片拖放決策分析器 (Drag-and-Drop Vision Decision Tool)
支援模型：clef-flash (Jev / System One API 規格)
"""

import pygame
import ollama
import base64
import json
import threading
from pathlib import Path

SCREEN_WIDTH = 960
SCREEN_HEIGHT = 640

BG_COLOR = (18, 18, 24)
PANEL_COLOR = (28, 29, 38)
ACCENT_COLOR = (80, 200, 120)
TEXT_COLOR = (235, 235, 240)
MUTED_COLOR = (140, 145, 160)
WARN_COLOR = (230, 160, 60)

# 預設的 Jev / System One 決策問題結構 (可自行修改)
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
        pygame.display.set_caption("Clef-Flash 視覺決策分析器 (Drag & Drop Image)")
        self.clock = pygame.time.Clock()

        # 使用支援中英文的系統字型
        self.font_title = pygame.font.SysFont("pingfangtc,arial", 22, bold=True)
        self.font_body = pygame.font.SysFont("pingfangtc,menlo,consolas,arial", 15)
        self.font_small = pygame.font.SysFont("menlo,consolas,monospace", 13)

        self.image_surface = None
        self.image_path = None
        self.busy = False
        self.status_msg = "請將圖片 (PNG/JPG/WebP) 直接拖放至此視窗中..."
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
        self.status_msg = f"正在使用 clef-flash 分析: {Path(file_path).name} ..."

        # 載入預覽圖
        try:
            loaded = pygame.image.load(file_path)
            # 等比例縮放至左側預覽區 (最大 400x420)
            iw, ih = loaded.get_size()
            scale = min(400 / iw, 420 / ih)
            new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
            self.image_surface = pygame.transform.smoothscale(loaded, new_size)
        except Exception as e:
            self.status_msg = f"圖片載入失敗: {e}"
            self.busy = False
            return

        def worker():
            import time
            t0 = time.time()
            try:
                b64_img = self._encode_image_b64(file_path)
                state_payload = {
                    "filename": Path(file_path).name,
                    "task": "Evaluate visual state and return typed SystemOne decisions."
                }

                # 優先使用新版 ollama.systemone，若套件版本不同則自動透過 HTTP 呼叫 /v1/systemone
                if hasattr(ollama, "systemone"):
                    resp = ollama.systemone(
                        model="clef-flash",
                        state=state_payload,
                        images=[b64_img],
                        questions=DEFAULT_QUESTIONS,
                        keep_alive="30m"
                    )
                    # 將回傳物件轉為標準 dict 方便顯示
                    if hasattr(resp, "model_dump"):
                        self.result_data = resp.model_dump()
                    elif isinstance(resp, dict):
                        self.result_data = resp
                    else:
                        self.result_data = json.loads(resp.json())
                else:
                    import urllib.request
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
                self.status_msg = f"分析完成 ({self.latency_ms} ms) — 可隨時拖放新圖片"
            except Exception as e:
                self.status_msg = f"AI 決策錯誤: {e}"
            finally:
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    def draw(self):
        self.screen.fill(BG_COLOR)

        # 頂部狀態列
        title = self.font_title.render("Clef-Flash 視覺決策引擎 (Decision-Native / Jev API)", True, ACCENT_COLOR)
        self.screen.blit(title, (24, 18))
        status_color = WARN_COLOR if self.busy else MUTED_COLOR
        status = self.font_body.render(self.status_msg, True, status_color)
        self.screen.blit(status, (24, 50))

        # 左側：圖片拖放與預覽區
        left_rect = pygame.Rect(24, 85, 420, 530)
        pygame.draw.rect(self.screen, PANEL_COLOR, left_rect, border_radius=10)
        pygame.draw.rect(self.screen, (60, 65, 85), left_rect, width=2, border_radius=10)

        if self.image_surface:
            ix = left_rect.centerx - self.image_surface.get_width() // 2
            iy = left_rect.centery - self.image_surface.get_height() // 2
            self.screen.blit(self.image_surface, (ix, iy))
        else:
            hint = self.font_body.render("將圖片拖放到這裡 (Drop Image Here)", True, MUTED_COLOR)
            self.screen.blit(hint, (left_rect.centerx - hint.get_width() // 2, left_rect.centery - 10))

        # 右側：Jev / SystemOne 決策輸出區
        right_rect = pygame.Rect(464, 85, 472, 530)
        pygame.draw.rect(self.screen, PANEL_COLOR, right_rect, border_radius=
