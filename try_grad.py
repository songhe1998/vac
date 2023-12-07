import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MNISTConvNet(nn.Module):

    def __init__(self):

        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.conv1.register_backward_hook(self.backward_hook)
        self.conv1.register_forward_hook(self.forward_hook)

    def backward_hook(self, module, grad_input, grad_output):
    	print(grad_input[1].shape)

    def forward_hook(self, module, input, output):
    	print(module, input[0].shape, output[0].shape)

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

# net = MNISTConvNet()
# # print(list(net.children())[0]);exit()
# net.zero_grad()
# input = torch.randn(3, 1, 28, 28)
# out = net(input)
# grad = torch.ones_like(out)
# out.backward(grad)
# print(out.size())
# def backward_hook(module, grad_input, grad_output):
# 	print(grad_input[0].shape)
# res = resnet18(weights='IMAGENET1K_V1')
# res.eval()

# first_layer = list(res.children())[0]
# first_layer.register_backward_hook(backward_hook)

# print(first_layer)
# inputs = torch.randn(1,3,224,224)
# out = res(inputs)
# res.zero_grad()
# grad = torch.ones_like(out)
# out.backward(grad)

class VanillaBackpropResnetRep():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            #print(grad_in.shape)
            print(grad_in[0].shape)
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.children())[0]
        #print(first_layer);exit()
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class=None):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        if target_class is None:
            target_output = torch.ones_like(model_output)
        else:
            target_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
            target_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=target_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


res = resnet18(weights='IMAGENET1K_V1')
res.eval()
vbp = VanillaBackpropResnetRep(res)
input_image = torch.randn(1,3,224,224, requires_grad=True)
grads = vbp.generate_gradients(input_image)
print(grads.shape)
