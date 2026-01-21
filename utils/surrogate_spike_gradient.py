import torch
import torch.nn as nn
import torch.nn.functional as F

def get_surrogate_spike_gradient(sigmoid_beta, fast_sigmoid=True):
    class SurrGradSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            out = torch.zeros_like(input)
            out[input > 0] = 1.0
            return out
            
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            if fast_sigmoid:
                grad = grad_output/((sigmoid_beta*torch.abs(input)+1.0)**2)
            else:
                grad = grad_output*sigmoid_beta*torch.sigmoid(sigmoid_beta*input)*(1.0-torch.sigmoid(sigmoid_beta*input))
            return grad

    return SurrGradSpike.apply