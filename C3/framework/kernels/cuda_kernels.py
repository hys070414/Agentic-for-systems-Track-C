import torch


def matmul_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def matmul_f16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.half(), b.half()).float()


def matmul_f8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def relu_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def relu_f16(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def add_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b)


def mul_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mul(a, b)


def div_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.div(a, b)


def reduce_max_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, dim=-1, keepdim=True).values


def reduce_sum_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=-1, keepdim=True)


def reduce_mean_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=-1, keepdim=True)


def exp_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def sqrt_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)


def sub_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sub(a, b)


def conv2d_f32(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return torch.nn.functional.conv2d(input, weight, bias)


def winograd_forward_f32(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return torch.nn.functional.conv2d(input, weight, bias)


KERNEL_REGISTRY = {
    'matmul_f32': matmul_f32,
    'matmul_f16': matmul_f16,
    'matmul_f8': matmul_f8,
    'relu_f32': relu_f32,
    'relu_f16': relu_f16,
    'add_f32': add_f32,
    'mul_f32': mul_f32,
    'div_f32': div_f32,
    'reduce_max_f32': reduce_max_f32,
    'reduce_sum_f32': reduce_sum_f32,
    'reduce_mean_f32': reduce_mean_f32,
    'exp_f32': exp_f32,
    'sqrt_f32': sqrt_f32,
    'sub_f32': sub_f32,
    'conv2d_f32': conv2d_f32,
    'winograd_forward_f32': winograd_forward_f32,
}


def get_kernel(kernel_name: str):
    return KERNEL_REGISTRY.get(kernel_name)
