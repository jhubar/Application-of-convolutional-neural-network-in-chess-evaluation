# Convolution
First, we need to understand a standard 2D convolution operation. A standard 2D convolution applies 2D filters/kernels over the inputs at **fixed perceptive fild and spacial locations** to generate the output feature map. Certain parameters are involved in the convolution opération named as kernal, stride and padding.

## Kernel size:
Defines the fieldof view of the convolution. 

## Stride: 
Defines the step size of the kernels when traversing the image. 

## PAdding:
Defines how the border of sample is handled. For instance, for a kernel of 3 with stride of 1, no padding would result in down-smpling.

<img src=Image/conv.gif width=50% />
2D convilution using a kernel size of 3, stride of 1 and no padding
