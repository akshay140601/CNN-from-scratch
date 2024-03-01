import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def MyFConv2D(input_data, weight, bias=None, stride=1, padding=0):
    
    """
    My custom Convolution 2D calculation.

    [input]
    * input    : (batch_size, in_channels, input_height, input_width)
    * weight   : (you have to derive the shape :-)  # (out_channels, in_channels, kernel_height, kernel_width)
    * bias     : bias term # (out_channels)
    * stride   : stride size # 1 or a tuple
    * padding  : padding

    [output]
    * output   : (batch_size, out_channels, output_height, output_width)
    """

    assert len(input_data.shape) == len(weight.shape)
    assert len(input_data.shape) == 4
    
    ## padding x with padding parameter 
    ## HINT: use torch.nn.functional.pad()
    # ----- TODO -----

    x_pad = F.pad(input_data, (padding, padding, padding, padding), 'constant', 0)

    ## Derive the output size
    ## Create the output tensor and initialize it with 0
    # ----- TODO -----

    if type(stride) == int:
        stride_H = stride
        stride_W = stride
    else:
        stride_H = stride[0]
        stride_W = stride[1]

    output_height = int(((input_data.shape[2] + (2 * padding) - weight.shape[2]) / stride_H) + 1)
    output_width  = int(((input_data.shape[3] + (2 * padding) - weight.shape[3]) / stride_W) + 1)
    x_conv_out    = torch.zeros(input_data.shape[0], weight.shape[0], output_height, output_width)

    ## Convolution process
    ## Feel free to use for loop 

    '''im2col = F.unfold(x_pad, kernel_size=(weight.shape[-2], weight.shape[-1]), dilation=1, padding=0, stride=(stride_H, stride_W))
    print(im2col.shape)
    reshape_for_mm = im2col.reshape(-1, input_data.shape[1]*weight.shape[-2]*weight.shape[-1], output_height, output_width).transpose(0, 1)
    print(reshape_for_mm.shape)
    x_conv_out = torch.tensordot(a=weight.reshape(weight.shape[0], -1), b=reshape_for_mm, dims=1).transpose(0, 1)'''


    for i in range(output_height):
        for j in range(output_width):
            receptive_field = x_pad[:, :, i*stride_H:i*stride_H + weight.shape[-2], j*stride_W:j*stride_W + weight.shape[-1]]
            reshape_for_mm = receptive_field.reshape(input_data.shape[0], -1)
            x_conv_out[:, :, i, j] = reshape_for_mm @ weight.reshape(weight.shape[0], -1).transpose(0, 1)

    if bias is not None:
        bias = bias.to(x_conv_out.device)
        x_conv_out += bias.view(1, -1, 1, 1)

    return x_conv_out 


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ## Create the torch.nn.Parameter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- TODO -----

        '''seed = 18786
        #random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        weights = torch.tensor(np.random.rand(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        print(weights)'''

        self.W = nn.Parameter(torch.tensor(np.random.rand(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]), dtype=torch.float32))
        #print(self.W)
        if self.bias == True:
            self.b = nn.Parameter(torch.tensor(np.random.rand(self.out_channels), dtype=torch.float32))
        else:
            self.b = None
            
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        # ----- TODO -----
        
        return MyFConv2D(x, self.W, self.b, self.stride, self.padding)

    
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given? 
        ## Think about the relationship with kernel_size
        # ----- TODO -----

        if stride is not None:
            self.stride = stride
        else:
            self.stride = self.kernel_size


    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        
        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        
        ## Derive the output size
        # ----- TODO -----

        self.output_height   = int(((self.input_height - self.kernel_size[0]) / self.stride[0]) + 1)
        self.output_width    = int(((self.input_width - self.kernel_size[1]) / self.stride[1]) + 1)
        self.output_channels = self.channel
        self.x_pool_out      = torch.zeros(self.batch_size, self.output_channels, self.output_height, self.output_width)

        ## Maxpooling process
        ## Feel free to use for loop
        # ----- TODO -----

        '''im2col = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, dilation=1, padding=0)
        reshaping_for_max = im2col.reshape(self.batch_size, self.output_channels, self.kernel_size[0] * self.kernel_size[1], -1)
        max_vals, _ = torch.max(reshaping_for_max, dim=2)
        self.x_pool_out = max_vals.reshape(self.batch_size, self.output_channels, self.output_height, self.output_width)'''

        for i in range(self.output_height):
            for j in range(self.output_width):
                receptive_field = x[:, :, i*self.stride[0]:i*self.stride[1] + self.kernel_size[0], j*self.stride[1]:j*self.stride[1] + self.kernel_size[1]]
                max_values, _ = torch.max(receptive_field.reshape(receptive_field.size(0), receptive_field.size(1), -1), dim=2, keepdim=True)
                #print(max_values.shape)
                self.x_pool_out[:, :, i, j] = max_values.view(max_values.size(0), max_values.size(1))
                #self.x_pool_out[:, :, i, j], _ = torch.max(receptive_field, dim=(2,3), keepdim=True)
                #self.x_pool_out[:, :, i, j], _ = torch.max(self.x_pool_out, dim=3, keepdim=True)

        return self.x_pool_out


if __name__ == "__main__":

    ## Test your implementation of MyFConv2D.
    # ----- TODO -----

    seed = 18786
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    batch_size = 128
    in_channels = 3
    input_height = 28
    input_width = 28

    input_data = torch.tensor(np.random.rand(batch_size, in_channels, input_height, input_width), dtype=torch.float32)

    out_channels = 4
    kernel_height = 3
    kernel_width = 3
    weight = torch.tensor(np.random.rand(out_channels, in_channels, kernel_height, kernel_width), dtype=torch.float32)
    #print(weight)

    bias = torch.tensor(np.random.rand(out_channels), dtype=torch.float32)
    pool_kernel_size = (2, 2)
    pool_stride = (2, 2)

    #my_conv_layer = MyConv2D(in_channels, out_channels, (kernel_height, kernel_width), (2, 2), padding=2, bias=True)
    #custom_conv_output = my_conv_layer(input_data)
    custom_conv_output = MyFConv2D(input_data, weight, bias, (2, 2), 2)
    torch_conv_output = F.conv2d(input_data, weight, bias, (2, 2), padding=2)
    #print('Actual: ', torch_conv_output)
    #print('My implementation: ', custom_conv_output)
    assert torch.allclose(custom_conv_output, torch_conv_output, atol=1e-4), "Convolution outputs do not match!"

    custom_pool_output = MyMaxPool2D(pool_kernel_size, pool_stride)(custom_conv_output)
    torch_pool_output = F.max_pool2d(torch_conv_output, kernel_size=pool_kernel_size, stride=pool_stride)
    assert torch.allclose(custom_pool_output, torch_pool_output, atol=1e-4), "Max pooling outputs do not match!"

    print("Sanity test passed!")
