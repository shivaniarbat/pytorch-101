"""
This program is to learn how to write and train a simple linear regression model in
PyTorch without using the nn.Module.
"""
import matplotlib.pyplot as plt
import torch

# the class for plotting
class plot_diagram():
    """
    this function is to plot the diagram for gradient descent algorithm
    """

    # Constructor
    def __init__(self,X,Y,w,stop,go = False):
        start = w.data # w is tensor
        self.error = []
        self.parameter = []
        self.X = X
        self.Y = Y
        self.parameter_values = torch.arange(start,stop)
        self.Loss_function = [criterian(forward(X),Y) for w.data in self.parameter_values]
        w.data = start

    # execute
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel('A')
        plt.ylim(-20,20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
        plt.show()

    # Destructor
    # def __del__(self):
    #     plt.close('all')

# create a forward function for prediction
def forward(x):
    yhat = w * x
    return yhat

# cost function
def criterian(yhat, y):
    # MSE function for evaluating the result
    return torch.mean((yhat - y) ** 2)

# learning rate
lr = 0.1
LOSS = []

X = torch.arange(-3,3,0.1).view(-1,1)
f = -3 * X

Y = f + 0.1 * torch.randn(X.size())

w = torch.tensor(-10.0, requires_grad=True)

gradient_plot = plot_diagram(X,Y,w,stop=5)

# training a model

def train_model(iter):
    for epoch in range(iter):

        # make prediction
        Yhat = forward(X)

        # calculate loss
        loss = criterian(Yhat,Y)

        # plot the diagram for better idea
        gradient_plot(Yhat,w,loss.item(),epoch)

        # store the loss
        LOSS.append(loss.item())

        # backward pass
        loss.backward()

        # update parameters
        w.data = w.data - lr * w.grad.data

        # zero the gradient before running the backward pass
        w.grad.zero_()

train_model(4)

# plot the loss at each iteration
plt.plot(LOSS)
plt.tight_layout()
plt.show()


