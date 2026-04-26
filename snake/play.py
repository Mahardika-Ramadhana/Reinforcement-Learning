import torch
from environment import SnakeEnv
from model import Linear_QNet
import numpy as np
import time

def play():
    # Inisialisasi Environment
    game = SnakeEnv()
    
    # Load Model (Sesuai dengan arsitektur saat training: 24 input, 3 output)
    model = Linear_QNet(24, 256, 256, 3)
    model.load_state_dict(torch.load("model/model.pth", map_location=model.device))
    model.to_device()
    model.eval() # Set ke evaluation mode (penting untuk neural network)

    print("AI is playing... Press Ctrl+C to stop.")
    
    record = 0
    while True:
        # 1. Ambil state saat ini
        state = game._get_state()
        state_tensor = torch.tensor(state, dtype=torch.float).to(model.device)
        
        # 2. Prediksi aksi (Tanpa epsilon/randomness)
        with torch.no_grad():
            prediction = model(state_tensor)
            move_idx = torch.argmax(prediction).item()
        
        final_move = [0, 0, 0]
        final_move[move_idx] = 1
        
        # 3. Jalankan aksi
        _, _, done = game.step(final_move)
        game.render()
        
        # Berikan sedikit delay agar mata manusia bisa mengikuti (opsional)
        time.sleep(0.05) 
        
        if done:
            print(f"Game Over! Score: {game.score}")
            game.reset()

if __name__ == "__main__":
    play()
