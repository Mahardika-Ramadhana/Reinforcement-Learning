import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def to_device(self):
        self.to(self.device)

    def save(self, n_games, record, file_name="model.pth"):
        model_directory = os.path.join(os.path.dirname(__file__), "model")
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        
        file_path = os.path.join(model_directory, file_name)
        checkpoint = {
            'n_games': n_games,
            'record': record,
            'model_state_dict': self.state_dict()
        }
        torch.save(checkpoint, file_path)

class QTrainer:
    def __init__(self, model, lr, gamma, target_model=None):
        self.model = model
        self.target_model = target_model if target_model else model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = self._to_tensor(states)
        next_states = self._to_tensor(next_states)
        actions = self._to_tensor(actions)
        rewards = self._to_tensor(rewards)
        dones = self._to_tensor(dones)

        if len(states.shape) == 1:
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)

        current_predictions = self.model(states)
        target_values = current_predictions.clone()
        
        with torch.no_grad():
            next_state_predictions = self.target_model(next_states)
            
        max_next_q_values = torch.max(next_state_predictions, dim=1)[0]
        updated_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        action_indices = torch.argmax(actions, dim=1)
        batch_indices = torch.arange(len(dones)).to(self.model.device)
        target_values[batch_indices, action_indices] = updated_q_values

        self.optimizer.zero_grad()
        loss = self.criterion(target_values, current_predictions)
        loss.backward()
        self.optimizer.step()

    def _to_tensor(self, data):
        return torch.tensor(np.array(data), dtype=torch.float).to(self.model.device)
