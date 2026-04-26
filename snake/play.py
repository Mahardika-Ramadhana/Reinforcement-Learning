import torch
import time
from environment import SnakeEnv
from model import Linear_QNet

def play():
    game = SnakeEnv()
    
    # 32 inputs to match the new vision logic
    model = Linear_QNet(32, 256, 256, 3)
    checkpoint = torch.load("model/model.pth", map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to_device()
    model.eval()

    print(f"Watch AI playing (Trained for {checkpoint['n_games']} games. Record: {checkpoint['record']})")
    
    while True:
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float).to(model.device)
        
        with torch.no_grad():
            prediction = model(state_tensor)
            move_idx = torch.argmax(prediction).item()
        
        action = [0, 0, 0]
        action[move_idx] = 1
    
        _, _, done = game.step(action)
        game.render(n_games=checkpoint['n_games'], record=checkpoint['record'])
        
        time.sleep(0.05) 
        
        if done:
            print(f"Final Score: {game.score}")
            game.reset()

if __name__ == "__main__":
    play()
