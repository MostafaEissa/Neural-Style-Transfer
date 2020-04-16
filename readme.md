# Neural Style Transfer

This is my implementation of [PyTorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) on neural style transfer. This repo is fully functional, but its main purpose is for me to learn about Nerual Style Transfer.

# Model 

The model was trained using [PyTorch](http://pytorch.org/).

## Basic Usage 

To train the model with the default hyperparameters, simply run:

```
python run_model.py PATH_TO_CONTENT_IMAGE PATH_TO_STYLE_IMAGE PATH_TO_OUTPUT_IMAGE
```

In this case the model with use the content image and style image to produce the output in the specified output directory.