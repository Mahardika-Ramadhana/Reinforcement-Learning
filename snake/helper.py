import matplotlib.pyplot as plt
import os

plt.ion()

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    plt.plot(scores, label='Current Score', color='blue', alpha=0.3)
    plt.plot(mean_scores, label='Average Score', color='red', linewidth=2)
    
    plt.ylim(ymin=0)
    plt.legend()
    plt.draw()
    plt.pause(0.1)
    
    _save_plot()

def _save_plot():
    plot_directory = "./plots"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    plt.savefig(os.path.join(plot_directory, "training_progress.png"))
