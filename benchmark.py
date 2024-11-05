import torch
import torch.nn.functional as F
import logging

import os 
# os.environ['TRITON_INTERPRET'] = '1'

import triton
import triton.language as tl
from triton.runtime import driver

from kernels.beaver_kernel import kernel_lbmm_fwd, beaver_forward
from kernels.beaver_kernel import rwkv_inner as rwkv_inner_triton

from kernels.fla.fla_rwkv6 import chunk_rwkv6 as rwkv_fla

from kernels.pt_kernel import rwkv_inner as rwkv_inner_pt
from kernels.pt_kernel import rwkv_inner_compiled as rwkv_inner_pt_compiled

if __name__ == '__main__':
    
    torch.set_printoptions(threshold=1000, edgeitems=8, linewidth=160, sci_mode=False)
    torch.manual_seed(42)
    
    # checking gradients of backward implementation
    """
    B = 2
    H = 2
    K = 32
    L = 2
    chunk_len = L
    
    r = torch.randn( [ B, H, L, K ], device='cuda', requires_grad=True )
    k = torch.randn( [ B, H, L, K ], device='cuda', requires_grad=True )
    v = torch.randn( [ B, H, L, K ], device='cuda', requires_grad=True )
    w = torch.rand( [ B, H, L, K ], device='cuda', requires_grad=True )
    u = torch.randn( [ 1, H, 1, K ], device='cuda', requires_grad=True )
    kv_states = torch.zeros( [ B, H, K, K ],  device='cuda' )
    
    # sets the last arg check_grad_impl to True to run gradcheck. But the gradcheck doesn't work with torch.compile, so comment out the line
    # that is torch compiling rwkv_inner_triton, and also requires you to manually go through the kernel_fwd and set o = tl.zeros( ... , dtype=tl.float64 )
    # for additional precision required for gradcheck
    rwkv_inner_triton( r, k, v, w, u, kv_states, chunk_len, 64, True )
    """
    
    
    # opcheck for beavermm torch.library.custom_op, will check both forward and backward pass
    temp_r, temp_k = torch.randn( [256, 192, 64], device='cuda', requires_grad=True ), torch.randn( [256, 64, 192], device='cuda', requires_grad=True )
    temp_r_sign, temp_k_sign = temp_r > 0, temp_k > 0
    torch.library.opcheck(beaver_forward, (temp_r, temp_k.contiguous(), temp_r_sign, temp_k_sign.contiguous()))
    

    # Speed benchmark
    bench_configs = []
    
    bench_configs.append(
        triton.testing.Benchmark(
            x_names=["L"],  # Argument names to use as an x-axis for the plot
            x_vals=[64 * i for i in range(1, 5)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["torch", "torch_compiled", "triton", "fla"],  # Label name for the lines
            line_names=["Torch", "Torch_Compiled", "Triton", "FLA"],  # Line styles
            styles=[("green", "-"), ("green", "-."), ("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-k64",
            args={ 'K': 64 }
        ))


    @triton.testing.perf_report(bench_configs)
    def benchmark(L, provider, K):
        B = 8
        H = 32
        
        chunk_len = L
        
        r = torch.randn( [ B, H, L, K ], device='cuda' )
        k = torch.randn( [ B, H, L, K ], device='cuda' )
        v = torch.randn( [ B, H, L, K ] , device='cuda' )
        w = torch.rand( [ B, H, L, K ], device='cuda' )
        u = torch.randn( [ 1, H, 1, K ], device='cuda' )
        kv_states = torch.zeros( [ B, H, K, K ], device='cuda' )
        
        # songlin's FLA code expects w is already in logspace (referred to as g)
        g = torch.log(w)

        def bmm():
            return rwkv_inner_pt( r, k, v, w, u, kv_states, chunk_len, 64 )
        
        def bmm_compiled():
            return rwkv_inner_pt_compiled( r, k, v, w, u, kv_states, chunk_len, 64 )
        
        def lbmm():
            return rwkv_inner_triton( r, k, v, w, u, kv_states, chunk_len, 32 )
        
        def fla():
            return rwkv_fla( r, k, v, g, u, None, kv_states )
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: bmm(), quantiles=quantiles)
        if provider == 'torch_compiled':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: bmm_compiled(), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: lbmm(), quantiles=quantiles)
        if provider == 'fla':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: fla(), quantiles=quantiles)
        perf = lambda ms: ms
        
        if provider == 'triton':
            bmm_res = bmm()
            bmm_comp_res = bmm_compiled()
            lbmm_res = lbmm()
            fla_res = fla()
        
            #print(lbmm_res[0])
            print(f"torch vs torch_compiled outputs close: {torch.dist( bmm_res[0], bmm_comp_res[0] )}")
            print(f"beaver vs torch outputs close: {torch.dist( bmm_res[0], lbmm_res[0] )}")
            print(f"beaver vs FLA outputs close: {torch.dist( lbmm_res[0], fla_res[0] )}")
            # print( torch.dist( bmm_res[0], lbmm_res[0] ), torch.dist( bmm_res[1], lbmm_res[1] ), torch.dist( bmm_res[2], lbmm_res[2] ) )
            # print( torch.dist( bmm_res[0], lbmm_res[0] ) )
        
        torch.cuda.empty_cache()
        
        
        return perf(ms), perf(max_ms), perf(min_ms)


    benchmark.run(print_data=True)
    
    print()
    print( 'M,N,K,BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,GROUP_SIZE_M,num_warps,num_stages' )
    for k, v in kernel_lbmm_fwd.cache.items():
        print(
            f"{k[0]},{k[1]},{k[2]},"
            f"{v.kwargs['BLOCK_SIZE_M']},"
            f"{v.kwargs['BLOCK_SIZE_N']},"
            f"{v.kwargs['BLOCK_SIZE_K']},"
            f"{v.kwargs['GROUP_SIZE_M']},"
            f"{v.num_warps},"
            f"{v.num_stages}"
        )
    print()
    