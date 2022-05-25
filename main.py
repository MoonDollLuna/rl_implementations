import torch.nn as nn
from nns.base_nns import mlp

# CURRENTLY TESTING
# Create a NN and check that it was created properly

nn = mlp(4, [32, 64, 128], nn.ReLU, 16, nn.Softmax)
print(nn)

nn2 = mlp(4, [32, 64, 128], nn.ReLU, 16, nn.Softmax)
print(nn2)

nn3 = mlp(4, [32], nn.ReLU, 16, nn.Softmax)
print(nn3)
