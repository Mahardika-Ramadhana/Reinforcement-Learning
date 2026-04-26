import pygame
import sys
import numpy as np
import random


class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.window_size = 600
        self.cell_size = 60
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("RL Snake - Dynamic Sensing")
        self.clock = pygame.time.Clock()
        self.reset()

        # Tambahkan inisialisasi font
        pygame.font.init()
        self.font = pygame.font.SysFont("arial", 20)
        self.last_reward = 0

    def reset(self):
        self.direction = 1
        self.head = [self.window_size // 2, self.window_size // 2]
        # Inisialisasi badan berdasarkan cell_size agar tidak tumpang tindih
        self.snake_body = [
            self.head,
            [self.head[0] - self.cell_size, self.head[1]],
            [self.head[0] - (2 * self.cell_size), self.head[1]],
        ]
        self.food_pos = [0, 0]
        self._place_food()
        self.score = 0
        self.total_reward = 0
        self.done = False
        self.frame_iteration = 0
        return self._get_state()

    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        clock_wise = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head
        if self.direction == 0:
            y -= self.cell_size
        elif self.direction == 1:
            x += self.cell_size
        elif self.direction == 2:
            y += self.cell_size
        elif self.direction == 3:
            x -= self.cell_size

        self.head = [x, y]
        self.snake_body.insert(0, self.head)

        reward = -0.02 # Sedikit naikkan penalty agar tidak mutar-mutar
        if self._is_collision() or self.frame_iteration > 200 * len(self.snake_body):
            self.done = True
            reward = -10

        elif self.head == self.food_pos:
            reward = 20 # Naikkan dari 10 ke 20
            self.score += 1
            self._place_food()
        else:
            self.snake_body.pop()

        self.total_reward += reward
        self.last_reward = reward

        return self._get_state(), reward, self.done

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if (
            pt[0] < 0
            or pt[0] >= self.window_size
            or pt[1] < 0
            or pt[1] >= self.window_size
        ):
            return True
        if pt in self.snake_body[1:]:
            return True
        return False

    def _get_state(self):
        head = self.snake_body[0]

        # Arah Scan: N, S, E, W, NE, NW, SE, SW
        directions = [
            (0, -self.cell_size),
            (0, self.cell_size),
            (self.cell_size, 0),
            (-self.cell_size, 0),
            (self.cell_size, -self.cell_size),
            (-self.cell_size, -self.cell_size),
            (self.cell_size, self.cell_size),
            (-self.cell_size, self.cell_size),
        ]

        vision = []
        for dx, dy in directions:
            dist = 0
            curr = [head[0] + dx, head[1] + dy]
            # Scan sampai mentok tembok atau badan sendiri
            while 0 <= curr[0] < self.window_size and 0 <= curr[1] < self.window_size:
                dist += 1
                if curr in self.snake_body[1:]:
                    break
                curr[0] += dx
                curr[1] += dy
            
            # Fix Normalisasi: Pakai Euclidean Distance agar diagonal tidak melebihi 1.0
            # Straight directions (dx/dy = 0) -> factor = 1.0
            # Diagonal directions (dx/dy != 0) -> factor = sqrt(2)
            is_diagonal = dx != 0 and dy != 0
            dist_factor = np.sqrt(2) if is_diagonal else 1.0
            max_possible_dist = (self.window_size / self.cell_size) * dist_factor
            vision.append((dist * dist_factor) / max_possible_dist)

        # 1-4: Bahaya Instan (Up, Right, Down, Left)
        danger_directions = [
            (head[0], head[1] - self.cell_size),  # Up
            (head[0] + self.cell_size, head[1]),  # Right
            (head[0], head[1] + self.cell_size),  # Down
            (head[0] - self.cell_size, head[1]),  # Left
        ]
        danger = [self._is_collision(pt) for pt in danger_directions]

        state = [
            # 1-4: Bahaya Instan
            *danger,
            # 5-8: Arah Jalan
            self.direction == 0,  # UP
            self.direction == 1,  # RIGHT
            self.direction == 2,  # DOWN
            self.direction == 3,  # LEFT
            # 9-12: Lokasi Apel relatif
            self.food_pos[1] < head[1],  # Food Up
            self.food_pos[0] > head[0],  # Food Right
            self.food_pos[1] > head[1],  # Food Down
            self.food_pos[0] < head[0],  # Food Left
            # 13-20: Vision Sensors (Jarak ke rintangan di 8 arah)
            *vision,
            # 21-24: Lokasi Ekor (Penting buat hindari jebakan badan sendiri)
            self.snake_body[-1][1] < head[1],
            self.snake_body[-1][0] > head[0],
            self.snake_body[-1][1] > head[1],
            self.snake_body[-1][0] < head[0],
        ]
        return np.array(state, dtype=float)

    def render(self):
        self.screen.fill((0, 0, 0))
        for pos in self.snake_body:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (pos[0], pos[1], self.cell_size - 2, self.cell_size - 2),
            )
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (self.food_pos[0], self.food_pos[1], self.cell_size, self.cell_size),
        )

        reward_text = self.font.render(
            f"Score: {self.score} | Total Reward: {self.total_reward:.2f}", True, (255, 255, 255)
        )

        self.screen.blit(reward_text, [10, 35])

        pygame.display.flip()
        self.clock.tick(1000)

    def _place_food(self):
        x = random.randint(0, (self.window_size // self.cell_size) - 1) * self.cell_size
        y = random.randint(0, (self.window_size // self.cell_size) - 1) * self.cell_size
        self.food_pos = [x, y]
