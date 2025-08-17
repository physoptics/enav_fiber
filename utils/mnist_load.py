import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms

# 1. Define a trivial transform that just converts PIL→Tensor
to_tensor = transforms.ToTensor()

# 2. Download train and test sets
train_ds = MNIST(root='./data', train=True,  download=True, transform=to_tensor)
test_ds  = MNIST(root='./data', train=False, download=True, transform=to_tensor)

# 3. Extract as NumPy arrays
#    - train_ds.data is a torch.ByteTensor of shape (60000,28,28)
#    - train_ds.targets is a torch.LongTensor of shape (60000,)
x_train = train_ds.data.numpy().astype(np.float32) / 255.0
y_train = train_ds.targets.numpy().astype(np.int64)

x_test  = test_ds.data.numpy().astype(np.float32)  / 255.0
y_test  = test_ds.targets.numpy().astype(np.int64)

print(x_train.shape, y_train.shape)  # (60000, 28, 28), (60000,)

# create a data directory if you like
import os
os.makedirs('data', exist_ok=True)

# save each one
np.save('data/x_train.npy', x_train)
np.save('data/y_train.npy', y_train)
np.save('data/x_test.npy',  x_test)
np.save('data/y_test.npy',  y_test)
