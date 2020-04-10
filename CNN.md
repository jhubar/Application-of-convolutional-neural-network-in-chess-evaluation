# Convolution
First, we need to understand a standard 2D convolution operation. A standard 2D convolution applies 2D filters/kernels over the inputs at **fixed perceptive fild and spacial locations** to generate the output feature map. Certain parameters are involved in the convolution opération named as kernal, stride and padding.

## Kernel size:
Defines the fieldof view of the convolution.

## Stride:
Defines the step size of the kernels when traversing the image.

## PAdding:
Defines how the border of sample is handled. For instance, for a kernel of 3 with stride of 1, no padding would result in down-smpling.
## Example
2D convilution using a kernel size of 3, stride of 1 and no padding
<figure>
  <p align="center">
  <img src=Image/conv.gif  with=50%/>
  </p>
</figure>

## Convolution with padding

From the simplest convolution maps, we can see that after convolution operation, the output will be smaller than the input, but sometimes we want the output size to be consistent with the input, and padding is introduced for this purpose, and this kind of padding in order to keep the input and output sizes the same, we will call it “the same paddling”. The actual performance of padding is to add 0 around the input, and the upper limit is the size of the convolution core – 1. As in the dotted line area in the figure below, generally speaking, the size of padding will not be given in the paper, and we need ourselves. Derivation, deduction formula can be seen below.
2D convilution using a kernel size of 3, stride of 1 and padding of 1


<figure>
  <p align="center">
  <img src=Image/convpadding1.gif  with=50%/>
  </p>
</figure>

-Same padding: Padding added to make the output the same as the input size, such as 3The nucleus of 3, same padding = 1,5The nucleus of 5, same padding = 2.
-full padding: padding = kernel size – 1
-valid padding: padding = 0
