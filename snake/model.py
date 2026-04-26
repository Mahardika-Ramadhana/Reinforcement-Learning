import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()  # Wajib panggil init parent agar parameter terdaftar
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        # Deteksi hardware: pindah ke GPU (cuda) jika tersedia
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def to_device(self):
        self.to(self.device)

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Konversi data ke Tensor dan pindahkan ke device yang sama dengan model
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.model.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(
            self.model.device
        )
        action = torch.tensor(np.array(action), dtype=torch.float).to(self.model.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(self.model.device)
        done = torch.tensor(np.array(done), dtype=torch.float).to(self.model.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        # 1. Prediksi Q-value saat ini
        pred = self.model(state)

        # 2. Hitung target Q-value
        target = pred.clone()
        
        # Hitung Q-value untuk state berikutnya dalam satu batch
        with torch.no_grad():
            next_pred = self.model(next_state)
            
        # BELLMAN EQUATION: Q_new = R + gamma * max(next_Q) * (1 - done)
        # Jika done=1, maka Q_new cuma Reward (tidak ada masa depan)
        max_next_q = torch.max(next_pred, dim=1)[0]
        Q_new = reward + (1 - done) * self.gamma * max_next_q

        # Update target hanya untuk aksi yang diambil
        # action is one-hot [batch_size, 3], so argmax gives the index
        action_indices = torch.argmax(action, dim=1)
        batch_indices = torch.arange(len(done)).to(self.model.device)
        target[batch_indices, action_indices] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
