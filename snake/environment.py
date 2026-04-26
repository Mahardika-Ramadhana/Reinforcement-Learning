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
        self.grid_size = self.window_size // self.cell_size
        
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("RL Snake AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)
        
        self.reset()

    def reset(self):
        self.direction = 1 # Right
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
        
        self._update_direction(action)
        self._move_snake()
        
        reward = self._evaluate_step()
        self.last_reward = reward
        self.total_reward += reward
        
        return self.get_state(), reward, self.done

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def _update_direction(self, action):
        clock_wise_directions = [0, 1, 2, 3] # UP, RIGHT, DOWN, LEFT
        current_idx = clock_wise_directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # Straight
            new_direction = clock_wise_directions[current_idx]
        elif np.array_equal(action, [0, 1, 0]): # Turn Right
            new_direction = clock_wise_directions[(current_idx + 1) % 4]
        else: # Turn Left
            new_direction = clock_wise_directions[(current_idx - 1) % 4]
            
        self.direction = new_direction

    def _move_snake(self):
        x, y = self.head
        if self.direction == 0: y -= self.cell_size
        elif self.direction == 1: x += self.cell_size
        elif self.direction == 2: y += self.cell_size
        elif self.direction == 3: x -= self.cell_size

        self.head = [x, y]
        self.snake_body.insert(0, self.head)

    def _evaluate_step(self):
        if self.is_collision() or self._is_stuck_in_loop():
            self.done = True
            return -10

        if self.head == self.food_pos:
            self.score += 1
            self._place_food()
            return 20
        
        self.snake_body.pop()
        return -0.02

    def is_collision(self, point=None):
        point = point if point else self.head
        return self._hits_boundary(point) or self._hits_self(point)

    def _hits_boundary(self, pt):
        return pt[0] < 0 or pt[0] >= self.window_size or pt[1] < 0 or pt[1] >= self.window_size

    def _hits_self(self, pt):
        return pt in self.snake_body[1:]

    def _is_stuck_in_loop(self):
        return self.frame_iteration > 200 * len(self.snake_body)

    def get_state(self):
        head = self.snake_body[0]
        vision = self._get_vision_features(head)
        danger = self._get_danger_features(head)
        
        state = [
            *danger,
            *self._get_direction_features(),
            *self._get_food_relative_features(head),
            *vision,
            *self._get_tail_relative_features(head)
        ]
        return np.array(state, dtype=float)

    def _get_danger_features(self, head):
        adjacent_cells = [
            (head[0], head[1] - self.cell_size), # Up
            (head[0] + self.cell_size, head[1]), # Right
            (head[0], head[1] + self.cell_size), # Down
            (head[0] - self.cell_size, head[1])  # Left
        ]
        return [self.is_collision(pt) for pt in adjacent_cells]

    def _get_direction_features(self):
        return [self.direction == i for i in range(4)]

    def _get_food_relative_features(self, head):
        return [
            self.food_pos[1] < head[1], # Food Up
            self.food_pos[0] > head[0], # Food Right
            self.food_pos[1] > head[1], # Food Down
            self.food_pos[0] < head[0]  # Food Left
        ]

    def _get_tail_relative_features(self, head):
        tail = self.snake_body[-1]
        return [
            tail[1] < head[1],
            tail[0] > head[0],
            tail[1] > head[1],
            tail[0] < head[0]
        ]

    def _get_vision_features(self, head):
        ray_directions = [
            (0, -self.cell_size), (0, self.cell_size), (self.cell_size, 0), (-self.cell_size, 0),
            (self.cell_size, -self.cell_size), (-self.cell_size, -self.cell_size),
            (self.cell_size, self.cell_size), (-self.cell_size, self.cell_size)
        ]
        
        vision = []
        for dx, dy in ray_directions:
            vision.append(self._scan_direction(head, dx, dy))
        return vision

    def _scan_direction(self, head, dx, dy):
        dist = 0
        curr = [head[0] + dx, head[1] + dy]
        while not self._hits_boundary(curr):
            dist += 1
            if curr in self.snake_body[1:]: break
            curr[0] += dx
            curr[1] += dy
            
        is_diagonal = dx != 0 and dy != 0
        factor = np.sqrt(2) if is_diagonal else 1.0
        max_dist = self.grid_size * factor
        return (dist * factor) / max_dist

    def render(self, n_games=0, record=0):
        self.screen.fill((0, 0, 0))
        self._draw_grid()
        self._draw_snake()
        self._draw_food()
        self._draw_ui(n_games, record)
        pygame.display.flip()
        self.clock.tick(120)

    def _draw_grid(self):
        for i in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (20, 20, 20), (i, 0), (i, self.window_size))
            pygame.draw.line(self.screen, (20, 20, 20), (0, i), (self.window_size, i))

    def _draw_snake(self):
        for i, pos in enumerate(self.snake_body):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(self.screen, color, (pos[0]+1, pos[1]+1, self.cell_size-2, self.cell_size-2))

    def _draw_food(self):
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food_pos[0], self.food_pos[1], self.cell_size, self.cell_size))

    def _draw_ui(self, n_games, record):
        texts = [
            f"Game: {n_games}",
            f"Score: {self.score}",
            f"Record: {record}",
            f"Reward: {self.last_reward:.2f}"
        ]
        for i, text in enumerate(texts):
            img = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(img, (10, 10 + i * 25))

    def _place_food(self):
        while True:
            x = random.randint(0, self.grid_size - 1) * self.cell_size
            y = random.randint(0, self.grid_size - 1) * self.cell_size
            self.food_pos = [x, y]
            if self.food_pos not in self.snake_body:
                break
