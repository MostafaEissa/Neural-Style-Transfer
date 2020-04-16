import torch
from PIL import Image 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models


def image_loader(image_name):
    image_size = 128
    image = Image.open(image_name)
    # resize image
    image = transforms.Resize(image_size)(image)
    # convert to tensor
    image = transforms.ToTensor()(image)
    # create a minibatch of a single image
    image = image.unsqueeze(0)
    return image
    
def image_show(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)

def image_save(tensor, path):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)