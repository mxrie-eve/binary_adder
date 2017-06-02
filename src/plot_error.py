import numpy as np
import matplotlib.pyplot as plt


data = np.load('../results/error.txt.npy')
plt.scatter(data[:,0], data[:,1])

plt.xlabel('Epoch')
plt.ylabel('Mean Error')
plt.title('Error as a function of the Epoch')
plt.savefig('../results/error.pdf')


