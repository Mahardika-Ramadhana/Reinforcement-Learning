import pygame
import sys
import numpy as np

class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.window_size = 800
        self.cell_size = 40
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("RL Playground - Snake")
        self.clock = pygame.time.Clock()

        self.snake_pos = [400, 400]
        self.done = False

    def reset(self):
        self.snake_pos = [400, 400]
        self.done = False
        return self._get_state()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if action == 0: self.snake_pos[1] -= self.cell_size    # Atas 
        elif action == 1: self.snake_pos[0] += self.cell_size  # Kanan
        elif action == 2: self.snake_pos[1] += self.cell_size  # Bawah.
        elif action == 3: self.snake_pos[0] -= self.cell_size  # Kiri

        reward = 0
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.window_size or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.window_size):
            self.done = True
            reward = -10 # Hukuman karena nabrak tembok

        return self._get_state(), reward, self.done

    def render(self):
        self.screen.fill((0, 0, 0)) # Black background
        pygame.draw.rect(self.screen, (0, 255, 0),
                        (self.snake_pos[0], self.snake_pos[1], self.cell_size, self.cell_size))
        pygame.display.flip()
        self.clock.tick(10)

    def _get_state(self):
        return np.array(self.snake_pos)

if __name__ == "__main__":
    env = SnakeEnv()
    state = env.reset()

    while not env.done:
        action = np.random.randint(0, 4)
        next_state, reward, done = env.step(action)
        env.render()

        if done:
            print("Game Over!")
            break
