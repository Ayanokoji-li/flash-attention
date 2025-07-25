import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 8
n_head = 4
seq_len = 4096
head_embd = 32

# set rand seed for reproducibility
torch.manual_seed(0)
q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     manual_result = manual_attn(q, k, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=1))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
# print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=1e-2, atol=1e-02))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=1))
# print("org", minimal_result)

print("=== profiling flash decoding === ")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    decoding_result = minimal_attn.forward_decode(q, k, v)
print('attn values sanity check:', torch.allclose(minimal_result, decoding_result, rtol=1e-2, atol=1e-02))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=1))
# print("decoding", minimal_result)