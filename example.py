import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import json

history = json.load(open("tests/history.json", "r"))
plot_history(history)
plt.savefig('history.png')