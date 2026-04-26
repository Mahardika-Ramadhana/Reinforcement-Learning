import torch
from environment import SnakeEnv
from model import Linear_QNet
import numpy as np
import time

def play():
    game = SnakeEnv()
    
    model = Linear_QNet(24, 256, 256, 3)
    checkpoint = torch.load("model/model.pth", map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to_device()
    model.eval()

    print(f"Playing with model trained for {checkpoint['n_games']} games. Record: {checkpoint['record']}")
    
    record = 0
    while True:
        state = game._get_state()
        state_tensor = torch.tensor(state, dtype=torch.float).to(model.device)
        
        with torch.no_grad():
            prediction = model(state_tensor)
            move_idx = torch.argmax(prediction).item()
        
        final_move = [0, 0, 0]
        final_move[move_idx] = 1
    
        _, _, done = game.step(final_move)
        game.render(n_games=checkpoint['n_games'], record=checkpoint['record'])
        
        time.sleep(0.05) 
        
        if done:
            print(f"Game Over! Score: {game.score}")
            game.reset()

if __name__ == "__main__":
    play()
