import torch
import random
import numpy as np
from collections import deque
from environment import SnakeEnv
from model import Linear_QNet, QTrainer

# Parameter Global
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        # Inisialisasi Model & Trainer
        # Input: 11 (State), Hidden: 256, Output: 3 (Action)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
