import torch
import random
from collections import deque
from environment import SnakeEnv
from model import Linear_QNet, QTrainer


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=100_000)  # Experience Replay Buffer
        self.model = Linear_QNet(24, 256, 256, 3)
        self.model.to_device()  # Sinkronisasi ke GPU
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Experience Replay: Belajar dari kumpulan memori acak agar tidak bias
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        # EPSILON DECAY: Awalnya ngaco, lama-lama nurut sama model
        self.epsilon = max(10, 200 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # Gerak acak
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.model.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()  # Ikut prediksi otak
            final_move[move] = 1
        return final_move


def train():
    record = 0
    agent = Agent()
    game = SnakeEnv()

    while True:
        state_old = game._get_state()
        final_move = agent.get_action(state_old)
        state_new, reward, done = game.step(final_move)
        game.render()

        # Latih memori jangka pendek (tiap langkah)
        agent.trainer.train_step(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # FIX SKOR: Simpan sebelum reset agar tidak terhapus
            final_score = game.score
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()  # Latih memori jangka panjang saat mati

            if final_score > record:
                record = final_score
                agent.model.save()

            print(f"Game {agent.n_games} | Score: {final_score} | Record: {record}")


if __name__ == "__main__":
    train()
