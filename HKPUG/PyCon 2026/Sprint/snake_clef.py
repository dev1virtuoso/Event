#!/usr/bin/env python3
"""
極速 + 超穩版貪食蛇（clef-flash）
"""

import pygame
import random
import ollama
import threading
import time
from collections import deque

CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT + 40

BLACK = (20, 20, 30)
WHITE = (230, 230, 230)
GREEN = (80, 200, 120)
RED = (220, 80, 80)
DARK_GREEN = (40, 120, 70)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
DIR_NAME = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
NAME_TO_DIR = {v: k for k, v in DIR_NAME.items()}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("極速穩版貪食蛇 - clef-flash")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)

        self.reset()
        self.ai_direction = RIGHT
        self.ai_busy = False
        self.last_ai_time = 0

    def reset(self):
        self.snake = deque([(GRID_WIDTH // 2, GRID_HEIGHT // 2)])
        self.direction = RIGHT
        self.ai_direction = RIGHT
        self.food = self._spawn_food()
        self.score = 0
        self.game_over = False
        self.steps = 0

    def _spawn_food(self):
        while True:
            pos = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if pos not in self.snake:
                return pos

    def _is_safe(self, pos):
        x, y = pos
        return (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and pos not in self.snake)

    def _get_safe_directions(self):
        head = self.snake[0]
        safe = []
        for d in DIRECTIONS:
            nx, ny = head[0] + d[0], head[1] + d[1]
            if self._is_safe((nx, ny)):
                safe.append(d)
        return safe

    def _get_state(self):
        head = self.snake[0]
        body_set = set(self.snake)

        def danger(dx, dy):
            nx, ny = head[0] + dx, head[1] + dy
            return not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT) or (nx, ny) in body_set

        # 食物相對方向
        fx, fy = self.food
        hx, hy = head
        food_hint = []
        if fy < hy: food_hint.append("up")
        if fy > hy: food_hint.append("down")
        if fx < hx: food_hint.append("left")
        if fx > hx: food_hint.append("right")

        return {
            "head": head,
            "food": self.food,
            "len": len(self.snake),
            "danger": {
                "up": danger(0, -1),
                "down": danger(0, 1),
                "left": danger(-1, 0),
                "right": danger(1, 0)
            },
            "food_dir": food_hint,
            "current": DIR_NAME[self.direction]
        }

    def _ask_clef_async(self):
        if self.ai_busy:
            return
        self.ai_busy = True

        def worker():
            try:
                state = self._get_state()
                questions = {
                    "move": {
                        "type": "choice",
                        "instructions": (
                            "You are a snake. SURVIVAL FIRST. "
                            "Never choose a direction marked as danger=true. "
                            "Among safe directions, prefer the ones in food_dir. "
                            "Do not reverse into yourself."
                        ),
                        "criteria": {
                            "up": "move up",
                            "down": "move down",
                            "left": "move left",
                            "right": "move right"
                        }
                    }
                }
                resp = ollama.systemone(
                    model="clef-flash",
                    state=state,
                    questions=questions,
                    keep_alive="30m"
                )
                choice = resp.answers["move"].choice
                new_dir = NAME_TO_DIR.get(choice, self.direction)

                # 只接受非反向的方向
                if new_dir != OPPOSITE.get(self.direction):
                    self.ai_direction = new_dir
            except Exception as e:
                print("AI error:", e)
            finally:
                self.ai_busy = False

        threading.Thread(target=worker, daemon=True).start()

    def update(self):
        if self.game_over:
            return

        # 定期請求 AI
        now = time.time()
        if now - self.last_ai_time > 0.15:
            self._ask_clef_async()
            self.last_ai_time = now

        # === 核心安全機制 ===
        # 先看 AI 建議的方向是否安全
        proposed = self.ai_direction
        head = self.snake[0]
        test_pos = (head[0] + proposed[0], head[1] + proposed[1])

        if self._is_safe(test_pos) and proposed != OPPOSITE.get(self.direction):
            self.direction = proposed
        else:
            # AI 方向危險 → 立刻改用安全方向
            safe_dirs = self._get_safe_directions()
            if safe_dirs:
                # 優先選朝食物的安全方向
                fx, fy = self.food
                hx, hy = head
                best = None
                for d in safe_dirs:
                    nx, ny = hx + d[0], hy + d[1]
                    # 簡單啟發式：靠近食物
                    if abs(nx - fx) + abs(ny - fy) < abs(hx - fx) + abs(hy - fy):
                        best = d
                        break
                self.direction = best if best else random.choice(safe_dirs)
            else:
                # 無路可走
                self.game_over = True
                return

        # 實際移動
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        if not self._is_safe(new_head):
            self.game_over = True
            return

        self.snake.appendleft(new_head)
        self.steps += 1

        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            self.steps = 0
        else:
            self.snake.pop()

        if self.steps > 400:
            self.game_over = True

    def draw(self):
        self.screen.fill(BLACK)

        fx, fy = self.food
        pygame.draw.rect(self.screen, RED, (fx*CELL_SIZE, fy*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == 0 else DARK_GREEN
            pygame.draw.rect(self.screen, color, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE-1, CELL_SIZE-1))

        status = "AI thinking" if self.ai_busy else "AI ready"
        info = self.font.render(f"Score: {self.score}   {status}", True, WHITE)
        self.screen.blit(info, (10, SCREEN_HEIGHT - 32))

        if self.game_over:
            over = self.font.render("GAME OVER - Press R", True, RED)
            self.screen.blit(over, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 15))

        pygame.display.flip()

    def run(self):
        print("預熱 clef-flash...")
        try:
            ollama.systemone(
                model="clef-flash",
                state={"warmup": 1},
                questions={"t": {"type": "noul", "instructions": "ok"}},
                keep_alive="30m"
            )
            print("預熱完成")
        except Exception as e:
            print("預熱失敗:", e)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.reset()
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not self.game_over:
                self.update()

            self.draw()
            self.clock.tick(16)   # 可再調高到 18~20

        pygame.quit()

if __name__ == "__main__":
    print("啟動極速穩版貪食蛇...")
    game = SnakeGame()
    game.run()