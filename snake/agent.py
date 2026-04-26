import torch
import random
import os
from collections import deque
from environment import SnakeEnv
from model import Linear_QNet, QTrainer
from helper import plot


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.98  # Lebih peduli masa depan
        self.memory = deque(maxlen=100_000)
        
        # Double DQN: Model Utama dan Model Target
        self.model = Linear_QNet(24, 256, 256, 3)
        self.model.to_device()
        self.target_model = Linear_QNet(24, 256, 256, 3)
        self.target_model.to_device()
        self.update_target_model()
        
        self.trainer = QTrainer(self.model, lr=0.0005, gamma=self.gamma, target_model=self.target_model)
        self.record = 0
        self.load_model()

    def update_target_model(self):
        """Salin bobot dari model utama ke model target"""
        self.target_model.load_state_dict(self.model.state_dict())

    def load_model(self):
        file_path = "./model/model.pth"
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=self.model.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.n_games = checkpoint['n_games']
            self.record = checkpoint['record']
            self.update_target_model()
            print(f"Resuming from Game: {self.n_games}, Record: {self.record}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Experience Replay
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        # EPSILON DECAY: AI mengeksplorasi lebih lama sampai sekitar 500 games
        self.epsilon = max(5, 180 - (self.n_games / 3)) 
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.model.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    game = SnakeEnv()

    while True:
        state_old = game._get_state()
        final_move = agent.get_action(state_old)
        state_new, reward, done = game.step(final_move)
        
        # Percepat speed saat training, render tiap langkah
        game.render(n_games=agent.n_games, record=agent.record)

        agent.trainer.train_step(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            final_score = game.score
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Setiap 10 game, sinkronisasi otak target agar belajar lebih stabil
            if agent.n_games % 10 == 0:
                agent.update_target_model()

            if final_score > agent.record:
                agent.record = final_score
                agent.model.save(agent.n_games, agent.record)

            print(f"Game {agent.n_games} | Score: {final_score} | Record: {agent.record}")
            
            # Update Plot
            plot_scores.append(final_score)
            total_score += final_score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
