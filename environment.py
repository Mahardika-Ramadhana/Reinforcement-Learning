import pygame
import sys
import numpy as np
import random


class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.window_size = 800
        self.cell_size = 40
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("RL Playground - Snake")
        self.clock = pygame.time.Clock()

        self.snake_body = [[400, 400], [360, 400], [320, 200]]
        self.direction = 1
        self.done = False
        self._place_food()

    def reset(self):
        self.snake_body = [[400, 400], [360, 400], [320, 200]]
        self.direction = 1
        self.done = False
        self._place_food()

        return self._get_state()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Mengambil posisi kepala saat ini
        head_x, head_y = self.snake_body[0]

        # Koordinat kepala baru berdasarkan action
        if action == 0:
            head_y -= self.cell_size  # Up
        elif action == 1:
            head_x += self.cell_size  # Right
        elif action == 2:
            head_y += self.cell_size  # Down
        elif action == 3:
            head_x -= self.cell_size  # Left

        new_head = [head_x, head_y]

        # Tambahkan kepala baru ke urutan list tubuh
        self.snake_body.insert(0, new_head)

        reward = 0

        # Cek apakah ular memakan apel
        if head_x == self.food_pos[0] and head_y == self.food_pos[1]:
            print("Ular makan apel di: ", self.food_pos)
            reward = 10
            self._place_food()
        else:
            self.snake_body.pop()

        # Cek game Over
        if (
            head_x < 0
            or head_x > self.window_size
            or head_x < 0
            or head_x > self.window_size
        ):
            self.done = True
            reward = -15

        return self._get_state(), reward, self.done

    def render(self):
        self.screen.fill((0, 0, 0))  # Black background

        for pos in self.snake_body:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (pos[0], pos[1], self.cell_size, self.cell_size),
            )

        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (self.food_pos[0], self.food_pos[1], self.cell_size, self.cell_size),
        )

        pygame.display.flip()
        self.clock.tick(10)

    def _get_state(self):
        return np.array(self.snake_body[0])

    def _place_food(self):
        x = random.randint(0, (self.window_size // self.cell_size) - 1) * self.cell_size
        y = random.randint(0, (self.window_size // self.cell_size) - 1) * self.cell_size

        self.food_pos = [x, y]


if __name__ == "__main__":
    env = SnakeEnv()
    state = env.reset()

    while not env.done:
        action = np.random.randint(0, 4)
        next_state, reward, done = env.step(action)
        env.render()

        if done:
            print("Game Over!")
            pygame.quit()
            break
