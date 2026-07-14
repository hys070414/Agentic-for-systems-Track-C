import cupy as cp
from cupy.linalg import matmul as cp_matmul
import numpy as np
from scipy.special import erf as scipy_erf


def add(inputs):
    result = inputs[0]
    for i in range(1, len(inputs)):
        result = cp.add(result, inputs[i])
    return result


def mul(inputs):
    result = inputs[0]
    for i in range(1, len(inputs)):
        result = cp.multiply(result, inputs[i])
    return result

def div(inputs):
    return cp.divide(inputs[0], inputs[1])


def relu(inputs):
    return cp.maximum(inputs[0], 0)


def erf(inputs):
    return cp.array(scipy_erf(cp.asnumpy(inputs[0])))


def flatten(inputs, axis=1):
    x = inputs[0]
    if axis < 0:
        axis = x.ndim + axis
    new_shape = (-1,) if axis == 0 else (x.shape[0], -1)
    return cp.reshape(x, new_shape)


def reshape(inputs, shape):
    if len(shape) == 0 and len(inputs) > 1:
        shape_tensor = inputs[1]
        if shape_tensor.ndim == 0:
            shape_tensor = cp.expand_dims(shape_tensor, 0)
        shape = cp.asnumpy(shape_tensor)
        shape = shape.astype(int).tolist()
        if isinstance(shape, int):
            shape = [shape]
    if len(shape) == 0:
        return inputs[0]
    return cp.reshape(inputs[0], shape)


def transpose(inputs, perm):
    x = inputs[0]
    if len(perm) == 0 or len(perm) != x.ndim:
        perm = list(range(x.ndim))[::-1]
    return cp.transpose(x, axes=perm)


def split(inputs, split_sizes, axis=0, num_outputs=1):
    x = inputs[0]
    if len(split_sizes) == 0:
        if x.shape[axis] % num_outputs == 0:
            chunk_size = x.shape[axis] // num_outputs
            split_sizes = [chunk_size] * num_outputs
        else:
            return [x]
    return cp.split(x, np.cumsum(split_sizes)[:-1], axis=axis)


def gemm(inputs, alpha=1.0, beta=1.0, transA=0, transB=0):
    a = inputs[0]
    b = inputs[1]
    
    if transA:
        a = cp.transpose(a)
    if transB:
        b = cp.transpose(b)
    
    result = alpha * cp_matmul(a, b)
    
    if len(inputs) > 2:
        result = beta * inputs[2] + result
    
    return result


def matmul(inputs):
    return cp_matmul(inputs[0], inputs[1])


def conv(inputs, kernel_shape=None, pads=None, strides=None, dilations=None):
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None
    
    if pads is None:
        pads = (0, 0, 0, 0)
    if strides is None:
        strides = (1, 1)
    if dilations is None:
        dilations = (1, 1)
    
    padding = (pads[0], pads[2]), (pads[1], pads[3])
    stride = strides
    dilation = dilations
    
    out_channels, in_channels, kh, kw = weight.shape
    
    x_padded = cp.pad(x, ((0, 0), (0, 0), padding[0], padding[1]), mode='constant')
    
    batch_size, _, h, w_dim = x_padded.shape
    out_h = (h - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    out_w = (w_dim - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    
    patches = cp.lib.stride_tricks.sliding_window_view(
        x_padded, (kh, kw), axis=(2, 3)
    )
    patches = patches[:, :, ::stride[0], ::stride[1]]
    
    patches = patches.reshape(batch_size, in_channels, out_h, out_w, kh, kw)
    patches = patches.transpose(0, 2, 3, 1, 4, 5)
    
    weight = weight.reshape(out_channels, in_channels, kh, kw)
    weight = weight.transpose(1, 2, 3, 0)
    
    result = cp.tensordot(patches, weight, axes=([3, 4, 5], [0, 1, 2]))
    result = result.transpose(0, 3, 1, 2)
    
    if bias is not None:
        result = result + bias.reshape(1, -1, 1, 1)
    
    return result


def global_average_pool(inputs):
    x = inputs[0]
    return cp.mean(x, axis=(2, 3), keepdims=True)


def softmax(inputs, axis=-1):
    x = inputs[0]
    x_max = cp.max(x, axis=axis, keepdims=True)
    exp_x = cp.exp(x - x_max)
    return exp_x / cp.sum(exp_x, axis=axis, keepdims=True)


def layer_normalization(inputs, epsilon=1e-5):
    x = inputs[0]
    scale = inputs[1] if len(inputs) > 1 else None
    bias = inputs[2] if len(inputs) > 2 else None
    
    mean = cp.mean(x, axis=-1, keepdims=True)
    variance = cp.var(x, axis=-1, keepdims=True)
    
    x_normalized = (x - mean) / cp.sqrt(variance + epsilon)
    
    if scale is not None:
        x_normalized = x_normalized * scale
    if bias is not None:
        x_normalized = x_normalized + bias
    
    return x_normalized


def gather(inputs, axis=0):
    data = inputs[0]
    indices = inputs[1]
    
    if indices.dtype != cp.int64:
        indices = indices.astype(cp.int64)
    
    return cp.take(data, indices, axis=axis)


def constant(value):
    return cp.array(value)


def identity(inputs):
    return inputs[0]


OPERATOR_REGISTRY = {
    'Add': add,
    'Constant': constant,
    'Conv': conv,
    'Div': div,
    'Erf': erf,
    'Flatten': flatten,
    'Gather': gather,
    'Gemm': gemm,
    'GlobalAveragePool': global_average_pool,
    'Identity': identity,
    'LayerNormalization': layer_normalization,
    'MatMul': matmul,
    'Mul': mul,
    'Relu': relu,
    'Reshape': reshape,
    'Softmax': softmax,
    'Split': split,
    'Transpose': transpose,
}


def get_operator(op_type):
    return OPERATOR_REGISTRY.get(op_type)
