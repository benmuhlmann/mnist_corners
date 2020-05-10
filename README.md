# mnist_corners
### How will a CNN learn to classify 0s and 1s when a fraction of 0s are crammed into a corner of the image

What do I mean by 'cramming images into a corner'?

original image:

![original image](https://github.com/benmuhlmann/mnist_corners/blob/master/markdown_figures/img_1_original.jpg)

cornerized image:

![cornerized image](https://github.com/benmuhlmann/mnist_corners/blob/master/markdown_figures/img_1_cornerized.jpg)

As the proportion of cornerized zeros changes, how will the network change? 
Changes I'm interested in: 
- How will the convolutional kernels change?
- How will accuracy and F1 score change?
