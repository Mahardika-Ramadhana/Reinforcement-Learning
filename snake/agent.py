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
        self.epsilon = 0
        self.gamma = 0.95 # Slightly lower for faster grid 20x20 convergence
        self.memory = deque(maxlen=100_000)
        self.record = 0
        
        # 11 inputs: 3 (Danger) + 4 (Direction) + 4 (Food)
        self.model = Linear_QNet(11, 256, 256, 3)
        self.model.to_device()
        
        self.target_model = Linear_QNet(11, 256, 256, 3)
        self.target_model.to_device()
        self.sync_target_model()
        
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma, target_model=self.target_model)
        
        self._load_checkpoint()

    def sync_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _load_checkpoint(self):
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        file_path = os.path.join(model_dir, "model.pth")
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=self.model.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.n_games = checkpoint['n_games']
            self.record = checkpoint['record']
            self.sync_target_model()
            print(f"Resuming from Game: {self.n_games}, Record: {self.record}")

    def get_action(self, state):
        self.epsilon = max(0, 80 - self.n_games)
        
        if random.randint(0, 200) < self.epsilon:
            return self._get_random_move()
        return self._get_predicted_move(state)

    def _get_random_move(self):
        move = [0, 0, 0]
        move[random.randint(0, 2)] = 1
        return move

    def _get_predicted_move(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.model.device)
        prediction = self.model(state_tensor)
        move_idx = torch.argmax(prediction).item()
        
        move = [0, 0, 0]
        move[move_idx] = 1
        return move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        batch = random.sample(self.memory, 1000) if len(self.memory) > 1000 else self.memory
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    
    agent = Agent()
    game = SnakeEnv()

    while True:
        old_state = game.get_state()
        action = agent.get_action(old_state)
        
        new_state, reward, done = game.step(action)
        game.render(n_games=agent.n_games, record=agent.record)

        agent.trainer.train_step(old_state, action, reward, new_state, done)
        agent.remember(old_state, action, reward, new_state, done)

        if done:
            final_score = game.score
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games % 10 == 0:
                agent.sync_target_model()

            if final_score > agent.record:
                agent.record = final_score
                agent.model.save(agent.n_games, agent.record)

            print(f"Game {agent.n_games} | Score: {final_score} | Record: {agent.record}")
            
            plot_scores.append(final_score)
            total_score += final_score
            plot_mean_scores.append(total_score / agent.n_games)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()
