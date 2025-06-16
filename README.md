# NEURAL.-STYLE-TRANSFER

COMPANY - CODTECH IT SOLUTION 

NAME - SIPRA SHARMA

INTERN ID - CT06DG741

DOMAIN - ARTIFICIAL INTELLIGENCE

DURATION - 6 WEEKS

MENTOR - NEELA SANTOSH 

DESCRIPTION -This Python script demonstrates the implementation of Neural Style Transfer (NST) using the PyTorch deep learning framework. Neural Style Transfer is a computer vision technique that merges the content of one image with the style of another to produce a visually appealing composite image. The code relies on several important libraries. PyTorch (torch, torch.nn, torch.optim) is used for building and optimizing the model, while Torchvision (torchvision.models, torchvision.transforms, torchvision.utils) provides access to pre-trained models (like VGG19) and image preprocessing utilities. The Python Imaging Library (PIL) is used to load and handle images, and if a compatible GPU is available, the script will leverage CUDA to accelerate processing.

At the core of this implementation is the pre-trained VGG19 convolutional neural network, which is used to extract feature maps from images. These feature maps help the model understand the high-level content and low-level stylistic features of an image. The VGG19 model is loaded with pretrained weights and frozen so that its parameters are not updated during training. The script preprocesses both the content and style images by resizing them to a uniform size and normalizing them using ImageNet statistics. A custom VGG class is defined to extract intermediate features from specific layers: deeper layers (like conv4_2) capture image content, while earlier layers (like conv1_1, conv2_1, etc.) capture style attributes such as textures, patterns, and colors.

The algorithm works by initializing the generated image as a copy of the content image. It then iteratively updates this image using gradient descent to minimize a loss function. The content loss ensures that the generated image retains the structure of the original content image, while the style loss (calculated using Gram matrices of feature maps) ensures the image mimics the visual style of the style image. These two losses are combined with different weights (typically, the style loss is given more importance) to guide the image transformation. Over 6000 training iterations, the generated image is refined, and intermediate results are saved every 100 epochs. The final stylized image is saved at the end. This project showcases how pre-trained deep neural networks can be used creatively for tasks beyond classification, enabling powerful artistic transformations with relatively simple code.  
I Perform this task in google colab.
