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
        pygame.display.set_caption("RL Snake AI - Ultimate Relative")
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
        
        old_dist = self._get_dist_to_food(self.head)
        self._update_direction(action)
        self._move_snake()
        new_dist = self._get_dist_to_food(self.head)
        
        reward = self._evaluate_step(old_dist, new_dist)
        self.last_reward = reward
        self.total_reward += reward
        
        return self.get_state(), reward, self.done

    def _get_dist_to_food(self, point):
        return np.linalg.norm(np.array(point) - np.array(self.food_pos))

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def _update_direction(self, action):
        directions = [0, 1, 2, 3] # UP, RIGHT, DOWN, LEFT
        idx = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # Straight
            self.direction = directions[idx]
        elif np.array_equal(action, [0, 1, 0]): # Right Turn
            self.direction = directions[(idx + 1) % 4]
        else: # Left Turn
            self.direction = directions[(idx - 1) % 4]

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
            return 20 # High reward for food
        
        self.snake_body.pop()
        
        # Aggressive Reward Shaping
        if new_dist < old_dist:
            return 0.1
        return -0.2

    def is_collision(self, pt=None):
        pt = pt if pt else self.head
        if pt[0] < 0 or pt[0] >= self.window_size or pt[1] < 0 or pt[1] >= self.window_size:
            return True
        if pt in self.snake_body[1:]:
            return True
        return False

    def get_state(self):
        head = self.snake_body[0]
        
        # Directions: UP, RIGHT, DOWN, LEFT
        # Food Relative logic
        food_left = self.food_pos[0] < head[0]
        food_right = self.food_pos[0] > head[0]
        food_up = self.food_pos[1] < head[1]
        food_down = self.food_pos[1] > head[1]

        # Points around head
        pt_u = [head[0], head[1] - self.cell_size]
        pt_d = [head[0], head[1] + self.cell_size]
        pt_l = [head[0] - self.cell_size, head[1]]
        pt_r = [head[0] + self.cell_size, head[1]]

        # Current direction one-hot
        dir_u = self.direction == 0
        dir_r = self.direction == 1
        dir_d = self.direction == 2
        dir_l = self.direction == 3

        state = [
            # Danger straight
            (dir_r and self.is_collision(pt_r)) or (dir_l and self.is_collision(pt_l)) or 
            (dir_u and self.is_collision(pt_u)) or (dir_d and self.is_collision(pt_d)),

            # Danger right
            (dir_u and self.is_collision(pt_r)) or (dir_d and self.is_collision(pt_l)) or 
            (dir_l and self.is_collision(pt_u)) or (dir_r and self.is_collision(pt_d)),

            # Danger left
            (dir_d and self.is_collision(pt_r)) or (dir_u and self.is_collision(pt_l)) or 
            (dir_r and self.is_collision(pt_u)) or (dir_l and self.is_collision(pt_d)),
            
            # Direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food
            food_left, food_right, food_up, food_down
        ]

        return np.array(state, dtype=int)

    def render(self, n_games=0, record=0):
        self.screen.fill((0, 0, 0))
        for pos in self.snake_body:
            pygame.draw.rect(self.screen, (0, 255, 0), (pos[0], pos[1], self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food_pos[0], self.food_pos[1], self.cell_size, self.cell_size))
        
        texts = [f"Game: {n_games}", f"Score: {self.score}", f"Record: {record}"]
        for i, t in enumerate(texts):
            self.screen.blit(self.font.render(t, True, (255,255,255)), (10, 10 + i*20))
            
        pygame.display.flip()
        self.clock.tick(1000) # Maximum training speed

    def _place_food(self):
        while True:
            x = random.randint(0, self.grid_size-1) * self.cell_size
            y = random.randint(0, self.grid_size-1) * self.cell_size
            self.food_pos = [x, y]
            if self.food_pos not in self.snake_body:
                break
