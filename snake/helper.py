import matplotlib.pyplot as plt
import os

# Set mode interaktif agar plot tidak menghentikan eksekusi kode
plt.ion()

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    plt.plot(scores, label='Score', color='blue', alpha=0.5)
    plt.plot(mean_scores, label='Mean Score', color='red', linewidth=2)
    
    plt.ylim(ymin=0)
    
    # Tampilkan teks skor terakhir
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    
    plt.legend()
    plt.draw()
    plt.pause(0.1)
    
    # Simpan sebagai gambar untuk LinkedIn
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    plt.savefig("./plots/training_progress.png")
