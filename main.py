import torch.nn as nn
from neural_networks.base_nns import mlp

# CURRENTLY TESTING
# Create a NN and check that it was created properly

neural_n = mlp(4, [32, 64, 128], nn.ReLU, 16, nn.Softmax)
print(neural_n)

neural_n2 = mlp(4, [32, 64, 128], nn.ReLU, 16, nn.Softmax)
print(neural_n2)

neural_n3 = mlp(4, [32], nn.ReLU, 16, nn.Softmax)
print(neural_n3)
