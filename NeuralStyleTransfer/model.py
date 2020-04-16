import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image 
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

class NeuralStyleTransferModel():
	
    def __init__(self, style_image, content_image,
                 cnn = models.vgg19(pretrained=True).features.eval(), 
                 normalization_mean = torch.tensor([0.485, 0.456, 0.406]), 
                 normalization_std = torch.tensor([0.229, 0.224, 0.225]),
                 content_layers=['conv_4'],
                 style_layers=['conv_1, conv_2', 'conv_3', 'conv_4', 'conv_5']):
        self.input_image = content_image.clone()
        self.model, self.style_losses, self.content_losses = NeuralStyleTransferModel.load_model(cnn, normalization_mean, normalization_std, style_image,  content_image, content_layers, style_layers)
    
   
    def load_model(cnn, cnn_normalization_mean, cnn_normalization_std, 
                    style_image, content_image, content_layers, style_layers):
                   
        content_losses = []
        style_losses = []

        cnn = copy.deepcopy(cnn)
        
        # normalization layer
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

        model = nn.Sequential(normalization)

        # add content and style loss ater each conv layer
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # replace to play nicely with content and style loss
                # see: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RunTimeError('unrecognized layer')
            
            model.add_module(name, layer)
            
            if name in content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module('content_loss_{}'.format(i), content_loss)
                content_losses.append(content_loss)
                
            if name in style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module('style_loss_{}'.format(i), style_loss)
                style_losses.append(style_loss)
                
        # trim off layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
            
        model = model[:(i + 1)] 
        return model, style_losses, content_losses
        
    def style_transfer(self, style_weight=100000, content_weight=1, num_steps=300):
    
        optimizer = optim.LBFGS([self.input_image.requires_grad_()])
    
        run = [0]
        while run[0] <= num_steps:
            
            def closure():
                
                self.input_image.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                self.model(self.input_image)
                style_score = 0
                content_score = 0
                
                for styl in self.style_losses:
                    style_score += styl.loss
                    
                for cnt in self.content_losses:
                    content_score += cnt.loss
                    
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                run[0] += 1
                
                if run[0] % 50 == 0:
                    print(f"run {run}")
                    print(f"Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")
                    print()
                  
                return style_score + content_score
            
            optimizer.step(closure)
            
        # limit value of image to be between 0 and 1
        self.input_image.data.clamp_(0, 1)
                
        return self.input_image
     
class Normalization(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
        
    def forward(self, img):
        return (img - self.mean) / self.std
   
   
class ContentLoss(nn.Module):
    
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = StyleLoss.gram_matrix(target_feature).detach()
    
    def forward(self, input):
        G = StyleLoss.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def gram_matrix(input):
        batch_num, feature_num, width, height = input.size()
        
        features = input.view(batch_num*feature_num, width*height)
        # gram matrix is result of multipying a mtrix by its transpose
        G = torch.mm(features, features.t())
        # normalize gram matrix by dividing by total number of elements
        G = G.div(batch_num * feature_num * width * height)
        return G