import torch

# activate function 
def sigmoid(x):
    return 1/(1+torch.exp(-x))

# NN setup (3 input, 2 hidden, 1 output)
torch.manual_seed(7)
features = torch.randn((1,3))
n_input = features.shape[1]
n_hidden = 2
n_output = 1
#weights = torch.randn_like(features)   # torch.randn_like(A): create same tensor like A
w1 = torch.randn(n_input,n_hidden)
w2 = torch.randn(n_hidden, n_output)
b1 = torch.randn((1,n_hidden))
b2 = torch.randn((1,n_output))
# Compute
h_input = torch.matmul(features,w1) + b1
h_output = sigmoid(h_input)
output = sigmoid(torch.matmul(h_output,w2)+b2)
print(output)