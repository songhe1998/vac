"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

from misc_functions import get_example_params, get_video_example_params, save_class_activation_images
import torch.nn as nn

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        # for module_pos, module in self.model.features._modules.items():
        #     x = module(x)  # Forward
        #     if int(module_pos) == self.target_layer:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x  # Save the convolution output on that layer

        for index, layer in enumerate(list(self.model.children())):
            # if index == 1:
            #     x = layer(x)
            print(index, layer)
            x = layer(x)
            if index == 8:
                break
                x = x.squeeze(-1).squeeze(-1)
            if index == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x

        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        # print(self.model)
        # exit()

        if False:
            conv_output, x = self.forward_pass_on_convolutions(x)
            x = x.view(x.size(0), -1)  # Flatten
            # Forward pass on the classifier
            classifier = list(self.model.children())[-1]
            #print(x.shape);exit()
            x = classifier(x)
            #x = self.model.classifier(x)

        # for index, layer in enumerate(list(self.model.children())):
        #     print(index, layer)
        # exit();
        x = x[0]
        conv_layer = list(self.model.children())[0]
        for index, layer in enumerate(list(conv_layer.children())):

            x = layer(x)
            # if index == 8:
            #     break
            #     x = x.squeeze(-1).squeeze(-1)
            if index == 7:
                x.register_hook(self.save_gradient)
                conv_output = x
        x = x.squeeze()
        print(x.shape)

        x = x.unsqueeze(0).transpose(-1,-2)

        temp_conv = list(self.model.children())[1]
        #lstm = self.model.children[2]

        x = temp_conv(x, len(conv_output))['conv_logits']
        #print(x.shape)


        return conv_output, x.squeeze()


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None, len_x=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy(), axis=-1)
            #print(target);exit()
            one_hot_output = torch.zeros_like(model_output)
            for i in range(len(target_class)):
                l = target_class[i]
                one_hot_output[i][l] = 1
        # Target for backprop
        else:
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
        # Zero grads
        #self.model.features.zero_grad()
        #self.model.classifier.zero_grad()
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(
            gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # Get convolution outputs
        print(conv_output.shape, guided_gradients.shape)
        #exit()
        cams = []
        for index in range(len(conv_output)):
            
            # print(index)
            target = conv_output.data.numpy()[index]
            # Get weights from gradients
            # for
            weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
            # print(guided_gradients.shape, target.shape, weights.shape);exit()
            # Create empty numpy array for cam
            cam = np.ones(target.shape[1:], dtype=np.float32)
            # Have a look at issue #11 to check why the above is np.ones and not np.zeros
            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
            cam = np.maximum(cam, 0)

            if np.max(cam) - np.min(cam) != 0:
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[-2],
                           input_image.shape[-1]), Image.ANTIALIAS))/255

            print(np.argmin(cam));exit()
            cams.append(cam)
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cams


if __name__ == '__main__':
    # Get params
    img_path = '/Users/songhewang/Downloads/test.png'
    img_path = '/Users/songhewang/Downloads/sharpened-image.png'
    #img_path = '/Users/songhewang/Downloads/sentence1_gsl/frame_0026.jpg'
    img_path = '/Users/songhewang/Downloads/24December_2010_Friday_tagesschau-5126'
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model, len_x) =\
        get_video_example_params(img_path)
    print(prep_img.shape, original_image[0].size)
    # Grad cam
    #for i in range(5,7):
    #filename = f'images0008_{i}.png'
    target_layer = 5
    grad_cam = GradCam(pretrained_model, target_layer=target_layer)
    # Generate cam mask
    target_class = None
    cams = grad_cam.generate_cam(prep_img, target_class)
    print(cams[0].shape)
    # Save mask
    for i in range(len(cams)):
        file_name_to_export = f'24December_2010_Friday_tagesschau-5126_{i}'
        save_class_activation_images(original_image[i], cams[i], file_name_to_export)
        print(f'Grad cam on {i} completed')

    # # Get params
    # img_path = '/Users/songhewang/Downloads/24December_2010_Friday_tagesschau-5126'
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model, len_x) =\
    #     get_video_example_params(img_path)
    # # Vanilla backprop
    # grad_cam = GradCam(pretrained_model, target_layer=7)
    # # Generate gradients
    # vanilla_grads = VBP.generate_gradients(prep_img, target_class, len_x)

    # # Save colored gradients
    # for i in range(len(vanilla_grads)):
    #     save_gradient_images(vanilla_grads[i], file_name_to_export + f'_Vanilla_BP_color_{i}')
    #     # Convert to grayscale
    #     grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads[i])
    #     # Save grayscale gradients

    #     save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    # print('Vanilla backprop completed') 
