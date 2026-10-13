"""A hand-written MLP with its own forward, backward and update steps."""
import torch


def relu(x):
    return torch.clamp(x, min=0.0)


# MY: It is implemented by hand (taken from the lecture notes :) )
class CustomMLP(torch.nn.Module):
    def __init__(self, layer_sizes, manual_optimization=True):
        super().__init__()
        assert len(layer_sizes) == 4
        D_in, H1, H2, D_out = layer_sizes

        if manual_optimization:
            self.W1 = torch.randn(D_in, H1)
            self.b1 = torch.zeros(1, H1)
            self.W2 = torch.randn(H1, H2)
            self.b2 = torch.zeros(1, H2)
            self.W3 = torch.randn(H2, D_out)
            self.b3 = torch.zeros(1, D_out)
            for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]:
                p.requires_grad_(False)
        else:
            self.W1 = torch.nn.Parameter(torch.randn(D_in, H1))
            self.b1 = torch.nn.Parameter(torch.zeros(1, H1))
            self.W2 = torch.nn.Parameter(torch.randn(H1, H2))
            self.b2 = torch.nn.Parameter(torch.zeros(1, H2))
            self.W3 = torch.nn.Parameter(torch.randn(H2, D_out))
            self.b3 = torch.nn.Parameter(torch.zeros(1, D_out))

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)
        pred = A2 @ self.W3 + self.b3
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "pred": pred}
        return pred, cache

    def loss(self, y_true, pred):
        return torch.sqrt(torch.mean((pred - y_true) ** 2))

    def backward(self, cache, y_true):
        X, Z1, A1, Z2, A2, pred = (
            cache["X"],
            cache["Z1"],
            cache["A1"],
            cache["Z2"],
            cache["A2"],
            cache["pred"],
        )
        N = X.shape[0]

        residual = pred - y_true
        rmsle = torch.sqrt(torch.mean(residual**2))
        dpred = residual / (N * (rmsle + 1e-12))

        dW3 = A2.T @ dpred
        db3 = dpred.sum(dim=0, keepdim=True)

        dA2 = dpred @ self.W3.T
        dZ2 = dA2 * (Z2 > 0).float()

        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(dim=0, keepdim=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0).float()

        dW1 = X.T @ dZ1
        db1 = dZ1.sum(dim=0, keepdim=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def step(self, grads, lr):
        with torch.no_grad():
            self.W1 -= lr * grads["W1"]
            self.b1 -= lr * grads["b1"]
            self.W2 -= lr * grads["W2"]
            self.b2 -= lr * grads["b2"]
            self.W3 -= lr * grads["W3"]
            self.b3 -= lr * grads["b3"]

    def predict(self, X):
        pred, _ = self.forward(X)
        return pred.clamp(min=0).view(-1)

