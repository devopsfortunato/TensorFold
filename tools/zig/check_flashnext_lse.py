"""torch.logsumexp(x, -1, keepdim=True) on fp32 rows against fn_logsumexp.cu, by raw bytes, at Flash Next's shapes."""

import ctypes

from ops_ffi import pointer, bind, P, U

# the draft head (79,591 ids; two ranks' shares 39,796 / 39,795), a rank's lm_head share at two ranks, the full head
WIDTHS = (79591, 39796, 39795, 124160, 248320, 1000, 4096)
ROWS = tuple(range(1, 17))

OUTSIDE_SOURCES = [{
    'repo': 'pytorch/pytorch', 'version': '9186a08b2c12b534aa935ca92e3f94834939c06a',
    'read': ['aten/src/ATen/native/ReduceOps.cpp (logsumexp_out_impl)', 'aten/src/ATen/native/cuda/Reduce.cuh',
             'aten/src/ATen/native/cuda/ReduceSumProdKernel.cu', 'aten/src/ATen/native/cuda/UnaryOpsKernel.cu',
             'aten/src/ATen/native/cuda/UnaryLogKernels.cu'],
    'idea': 'the composite amax / masked_fill / sub / exp / sum / log / add and the reduce kernel\'s summation order '
            '(block shape, vectorized accumulators, lane, warp and CTA trees); ours recomputes the same order in three '
            'launches without a semaphore',
    'statement': 'no code copied'}]


def config(lib, rows, n):
    import torch

    fn = lib.tf_fn_logsumexp_config
    fn.argtypes = [U, U, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
    fn.restype = ctypes.c_int
    props = torch.cuda.get_device_properties(0)
    out = (ctypes.c_uint32 * 5)()
    if fn(rows, n, props.multi_processor_count, props.max_threads_per_multi_processor, out):
        raise ValueError(f'no logsumexp config for {rows} x {n}')
    return dict(zip(('bw', 'bh', 'grid_x', 'ctas', 'split'), list(out)))


def check_lse(lib, check):
    import torch

    lse = bind(lib, 'tf_fn_logsumexp', [P, P, P, U, U, P])
    handle = P(torch.cuda.current_stream().cuda_stream)
    configs = {}

    def run(name, x):
        rows, n = x.shape
        c = config(lib, rows, n)
        configs[f'{rows}x{n}'] = c
        out = torch.empty((rows, 1), device='cuda', dtype=torch.float32)
        scratch = torch.empty(rows * (1 + c['ctas']), device='cuda', dtype=torch.float32)
        rc = lse(pointer(x), pointer(out), pointer(scratch), rows, n, handle)
        check(f'logsumexp/{name}/{rows}x{n}', out, torch.logsumexp(x, dim=-1, keepdim=True), rc)

    torch.manual_seed(907)
    for n in WIDTHS:
        for rows in ROWS:
            run('randn4', torch.randn((rows, n), device='cuda', dtype=torch.float32) * 4)
        for rows in (1, 2, 7, 16):
            logits = (torch.randn((rows, n), device='cuda') * 3 + 1).to(torch.bfloat16).float()
            run('bf16-logits', logits)
            spiky = torch.randn((rows, n), device='cuda') * 0.5
            spiky[:, torch.randint(0, n, (5,))] = 30.0
            run('spiky', spiky)
            masked = torch.randn((rows, n), device='cuda') * 4
            masked[:, ::3] = float('-inf')
            masked[-1] = float('-inf')                      # a row of -inf only: -inf
            run('neg-inf', masked)
        # a row that is a view at an odd offset: the exp temporary is contiguous whatever the input's alignment
        flat = torch.randn(3 * n + 1, device='cuda') * 4
        run('offset-view', flat[1:].view(3, n))
    return configs
