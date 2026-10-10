import torch

from .triton_utils import (
    enable_tilelens_trace,
    flatten_varlen_lse,
    unflatten_varlen_lse,
)


def build_varlen_lse_example():
    cu_seqlens = [0, 15, 156, 529]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)
    batch_size = len(cu_seqlens) - 1
    lengths = cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]
    max_seqlen = int(lengths.max().item())
    n_head = 5
    lse = torch.randn((batch_size, n_head, max_seqlen), dtype=torch.float32)
    return lse, cu_seqlens_tensor, max_seqlen


def trace_varlen_lse_kernels():
    enable_tilelens_trace()
    lse, cu_seqlens, max_seqlen = build_varlen_lse_example()
    flat = flatten_varlen_lse(lse, cu_seqlens)
    flat = flat.transpose(-2, -1).unsqueeze(dim=-1)
    unflatten_varlen_lse(flat, cu_seqlens, max_seqlen)


def launch_visualizer(share=False, port=None):
    import tilelens

    trace_varlen_lse_kernels()
    return tilelens.launch(share=share, port=port)
