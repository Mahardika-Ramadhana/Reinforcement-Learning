import pygame
import sys
import numpy as np
import random

class SnakeEnv:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        
        self.window_size = 600
        self.cell_size = 30
        self.grid_size = self.window_size // self.cell_size # 20
        
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("RL Snake AI - Temporal Grid Vision")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)
        
        self.reset()

    def reset(self):
        self.direction = 1 # RIGHT
        self.head = [self.window_size // 2, self.window_size // 2]
        self.snake_body = [
            self.head,
            [self.head[0] - self.cell_size, self.head[1]],
            [self.head[0] - (2 * self.cell_size), self.head[1]]
        ]
        self.score = 0
        self.total_reward = 0
        self.done = False
        self.frame_iteration = 0
        self.last_reward = 0
        self._place_food()
        return self.get_state()

    def step(self, action):
        self.frame_iteration += 1
        self._handle_events()
        
        old_dist = np.linalg.norm(np.array(self.head) - np.array(self.food_pos))
        self._update_direction(action)
        self._move_snake()
        new_dist = np.linalg.norm(np.array(self.head) - np.array(self.food_pos))
        
        reward = self._evaluate_step(old_dist, new_dist)
        self.last_reward = reward
        self.total_reward += reward
        
        return self.get_state(), reward, self.done

    def _update_direction(self, action):
        dirs = [0, 1, 2, 3] # UP, RIGHT, DOWN, LEFT
        idx = dirs.index(self.direction)
        if np.array_equal(action, [1, 0, 0]): pass
        elif np.array_equal(action, [0, 1, 0]): self.direction = dirs[(idx + 1) % 4]
        else: self.direction = dirs[(idx - 1) % 4]

    def _move_snake(self):
        x, y = self.head
        if self.direction == 0: y -= self.cell_size
        elif self.direction == 1: x += self.cell_size
        elif self.direction == 2: y += self.cell_size
        elif self.direction == 3: x -= self.cell_size
        self.head = [x, y]
        self.snake_body.insert(0, self.head)

    def _evaluate_step(self, old_dist, new_dist):
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake_body):
            self.done = True
            return -10
        if self.head == self.food_pos:
            self.score += 1
            self._place_food()
            return 20
        self.snake_body.pop()
        return 0.1 if new_dist < old_dist else -0.2

    def is_collision(self, pt=None):
        pt = pt if pt else self.head
        if pt[0] < 0 or pt[0] >= self.window_size or pt[1] < 0 or pt[1] >= self.window_size: return True
        return pt in self.snake_body[1:]

    def get_state(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        
        body_length = len(self.snake_body)
        for i, p in enumerate(reversed(self.snake_body)):
            ix, iy = p[0] // self.cell_size, p[1] // self.cell_size
            # Safety check: avoid IndexError if snake is exactly on boundary (colliding)
            if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size:
                val = 0.1 + (0.9 * (i / (body_length - 1))) if body_length > 1 else 1.0
                grid[iy][ix] = val
            
        fx, fy = self.food_pos[0] // self.cell_size, self.food_pos[1] // self.cell_size
        if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
            grid[fy][fx] = -1.0
        
        state = grid.flatten().tolist()
        state.extend([self.direction == i for i in range(4)])
        return np.array(state, dtype=float)

    def render(self, n_games=0, record=0):
        self.screen.fill((0, 0, 0))
        for i, p in enumerate(self.snake_body):
            color_val = max(50, 255 - (i * 5))
            color = (0, color_val, 0) if i == 0 else (0, max(0, color_val-50), 0)
            pygame.draw.rect(self.screen, color, (p[0]+1, p[1]+1, self.cell_size-2, self.cell_size-2))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food_pos[0], self.food_pos[1], self.cell_size, self.cell_size))
        
        msgs = [f"Game: {n_games}", f"Score: {self.score}", f"Record: {record}"]
        for i, m in enumerate(msgs):
            self.screen.blit(self.font.render(m, True, (255,255,255)), (10, 10 + i*20))
        pygame.display.flip()
        self.clock.tick(1000)

    def _handle_events(self):
        pass

    def _place_food(self):
        while True:
            self.food_pos = [random.randint(0, self.grid_size-1)*self.cell_size, random.randint(0, self.grid_size-1)*self.cell_size]
            if self.food_pos not in self.snake_body: break
