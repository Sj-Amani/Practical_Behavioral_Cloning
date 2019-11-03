'''
Draw the history of training data
'''
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

with open(os.getcwd()+'/train_hist.p', 'rb') as f:
	normal = pickle.load(f)

tr_loss = normal['loss']
val_loss = normal['val_loss']
num_epochs = len(tr_loss)

x = np.arange(num_epochs)+1

plt.plot(x, tr_loss, 'r--^', x, val_loss, 'b--^')
plt.xlim([0, 10.5])
plt.ylim([0, 0.05])
plt.grid(False)
plt.xlabel('Epoch Number')
plt.ylabel('MSE Loss')
plt.title('LOSS Results')
plt.legend(('Training Loss', 'Validation Loss'), loc='upper right', shadow=True)
fig = plt.gcf()
fig.savefig(os.getcwd()+'/results/loss_results.png')

print('LOSS Result saved in:')
print(os.getcwd()+'/results/loss_results.jpg')

plt.show()




