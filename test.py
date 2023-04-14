import numpy as np
from scipy import stats
from torchvision import datasets
from torchvision import datasets, transforms
import json

# train_dataset = datasets.CIFAR10( "data/cifar/", train=True, download=True, transform=None)

# x_m, alpha = 5000, 50.
# samples = np.int32((np.random.pareto(alpha, (10, 10))) * x_m)
print(np.random.normal(1, 0.2, 10))