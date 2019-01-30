# KF 01/24/2019

#################################################################
# Load data
#################################################################
import pickle
hist_file_path = 'history.pkl'

with open(hist_file_path, 'rb') as f:
    history = pickle.load(f)

#################################################################
# Plot Accuracy and Loss
#################################################################
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Plot the Loss and Accuracy")
N = len(history["loss"])
#plt.style.use("ggplot")
#plt.figure()
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
#N = epochs
l1 = ax1.plot(np.arange(0, N), history["acc"], label="train_acc")
l2 = ax1.plot(np.arange(0, N), history["val_acc"], label="val_acc", linewidth=2)
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch #')
ax1.set_ylim(0.5, 1.05)
ax1.grid()

l3 = ax2.plot(np.arange(0, N), history["loss"], color='orchid', linestyle='dashed', label="train_loss")
l4 = ax2.plot(np.arange(0, N), history["val_loss"], color='limegreen', linestyle='dashed', label="val_loss", linewidth=2)
ax2.set_ylabel('Loss')
ax2.set_ylim(0, max(max(history['loss']), max(history['val_loss'])))
# Put all label legend together
l = l1 + l2 + l3 + l4
labels = [i.get_label() for i in l]
#plt.legend(l, labels, loc='center right')
plt.legend(l, labels, loc='center right')

plt.title("Training Loss and Accuracy")
plt_name = 'plot_acc_loss.eps' 
plt.savefig(plt_name, format='eps', dpi=1000)

#################################################################
# Plot Learning Rate`
#################################################################
print("Plot Learning Rate")
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, N), history['lr'], linewidth=6)
plt.xlabel('Epoch #')
plt.ylabel('Learning Rate')
plt.grid()

plt.title("Learning Rate")
plt_name = 'plot_learning_rate.eps' 
plt.savefig(plt_name, format='eps', dpi=1000)
