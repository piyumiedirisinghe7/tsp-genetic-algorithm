import matplotlib.pyplot as plt
from typing import List

def plot_convergence(history: List[float], title: str = "Convergence"):
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Tour Distance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/convergence.png', dpi=150)
    plt.show()
    print("Plot saved to results/convergence.png")