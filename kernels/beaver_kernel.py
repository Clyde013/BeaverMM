import torch
import torch.nn.functional as F

import os 
# os.environ['TRITON_INTERPRET'] = '1'

import triton
import triton.language as tl
from triton.runtime import driver

def cfg( M, N, K, G, num ):
    return {
        'BLOCK_SIZE_M': M,
        'BLOCK_SIZE_N': N,
        'BLOCK_SIZE_K': K,
        'GROUP_SIZE_M': G,
        'NUM_STAGES': num,
    }


# configs = [
#     triton.Config( cfg( M=m, N=n, K=k, G=g, num=s ), num_stages=s, num_warps=w)
#     for m in [ 32, 16 ]
#     for n in [ 64, 32, 16 ]
#     for k in [ 32, 16 ]
#     for g in [ 1 ]
#     for s in [ 2 ]
#     for w in [ 4 ]
# ]
# configs = [
#     triton.Config( cfg( M=m, N=n, K=k, G=g, num=s ), num_stages=s, num_warps=w)
#     for m in [ 16 ]
#     for n in [ 16 ]
#     for k in [ 64, 32, 16, 8 ]
#     for g in [ 1 ]
#     for s in [ 2 ]
#     for w in [ 4 ]
# ]
configs = [
    triton.Config( cfg( M=m, N=n, K=k, G=g, num=s ), num_stages=s, num_warps=w)
    for m in [ 16 ]
    for n in [ 16 ]
    for k in [ 32 ]
    for g in [ 1 ]
    for s in [ 2 ]
    for w in [ 4 ]
]


def keeper( conf ):
    M = conf.kwargs[ 'BLOCK_SIZE_M' ]
    N = conf.kwargs[ 'BLOCK_SIZE_N' ]
    K = conf.kwargs[ 'BLOCK_SIZE_K' ]
    G = conf.kwargs[ 'GROUP_SIZE_M' ]
    w = conf.num_warps
    
    # if M * N * K > 32768:
    #     return False
    
    return True

print( len( list( filter( keeper, configs ) ) ) )


@triton.autotune(
    configs=list( filter( keeper, configs ) ),
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_lbmm_fwd(
    x_log_ptr,  # [B,M,K] log abs LHS
    y_log_ptr,  # [B,K,N] log abs RHS
    x_sign_ptr, # [B,M,K] sign LHS
    y_sign_ptr, # [B,K,N] sign RHS
    output_ptr, # [B,M,N] linspace output
    M, N, K, # sizes
    stride_x_b, stride_x_m, stride_x_k, # LHS strides
    stride_y_b, stride_y_k, stride_y_n, # RHS strides
    stride_xs_b, stride_xs_m, stride_xs_k, # LHS sign strides
    stride_ys_b, stride_ys_k, stride_ys_n, # RHS sign strides
    stride_o_b, stride_o_m, stride_o_n, # out strides
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):

    pid_batch = tl.program_id( 0 ) # Batch index
    x_log_ptr += pid_batch * stride_x_b 
    x_sign_ptr += pid_batch * stride_xs_b 
    y_log_ptr += pid_batch * stride_y_b 
    y_sign_ptr += pid_batch * stride_ys_b 
    output_ptr += pid_batch * stride_o_b

    pid_m = tl.program_id( 1 ) # M index
    m = pid_m * BLOCK_SIZE_M
    offs_x_m = (m + tl.arange(0, BLOCK_SIZE_M))[:, None]    # (M, 1)

    for n in tl.range( 0, (pid_m + 1) * BLOCK_SIZE_M, BLOCK_SIZE_N):
        o = tl.zeros( ( BLOCK_SIZE_M, BLOCK_SIZE_N ), dtype=tl.float32 )
        # uncomment this if ur checking gradients (and comment above line), gradcheck requires float64 numerical precision
        # o = tl.zeros( ( BLOCK_SIZE_M, BLOCK_SIZE_N ), dtype=tl.float64 )

        offs_y_n = (n + tl.arange(0, BLOCK_SIZE_N))[:, None]    # (N, 1)

        for k in tl.range( 0, K, step=BLOCK_SIZE_K, num_stages=NUM_STAGES ):
            offs_k = (k + tl.arange( 0, BLOCK_SIZE_K ))[None, :]    # (1, K)
            x_log_ptrs = x_log_ptr + offs_x_m * stride_x_m + offs_k * stride_x_k     # (M,1) + (1,K) -> (M,K)
            x_sign_ptrs = x_sign_ptr + offs_x_m * stride_xs_m + offs_k * stride_xs_k # (M,1) + (1,K) -> (M,K)

            y_log_ptrs = y_log_ptr + offs_y_n * stride_y_n + offs_k * stride_y_k     # (N,1) + (1,K) -> (N,K)
            y_sign_ptrs = y_sign_ptr + offs_y_n * stride_ys_n + offs_k * stride_ys_k # (N,1) + (1,K) -> (N,K)

            x_log = tl.load( x_log_ptrs )[:,None,:] # (M,1,K)
            y_log = tl.load( y_log_ptrs )[None,:,:] # (1,N,K)

            x_sign = tl.load( x_sign_ptrs )[:,None,:] # (M,1,K)
            y_sign = tl.load( y_sign_ptrs )[None,:,:] # (1,N,K)
            
            z_sign = x_sign == y_sign

            z_exp = tl.exp( x_log + y_log ) * ( z_sign * 2 - 1 ) # (M,N,K)
            o += tl.sum( z_exp, -1 )

        # apply triangular mask
        offs_y_n = (n + tl.arange(0, BLOCK_SIZE_N))[None, :]
        o = tl.where( offs_x_m > offs_y_n, o, 0.0 )
        
        output_ptrs = output_ptr + stride_o_m * offs_x_m + stride_o_n * offs_y_n # (M,N)
        # construct normal mask to ensure pointers don't overflow
        mask = (offs_x_m < M) & (offs_y_n < N)
        tl.store(output_ptrs, o, mask=mask)
    
@triton.autotune(
    configs=list( filter( keeper, configs ) ),
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_lbmm_bwd(
    x_log_ptr,  # [B,M,K] log abs LHS
    y_log_ptr,  # [B,K,N] log abs RHS
    x_sign_ptr, # [B,M,K] sign LHS
    y_sign_ptr, # [B,K,N] sign RHS
    grad_out_ptr,	# [B,M,N] gradient output
    dx_ptr,	# [B,M,K] dx output
    dy_ptr,	# [B,N,K] dy output
    M, N, K, # sizes
    stride_x_b, stride_x_m, stride_x_k, # LHS strides
    stride_y_b, stride_y_k, stride_y_n, # RHS strides
    stride_xs_b, stride_xs_m, stride_xs_k, # LHS sign strides
    stride_ys_b, stride_ys_k, stride_ys_n, # RHS sign strides
    stride_grad_b, stride_grad_m, stride_grad_n,    # grad_out strides
    stride_dx_b, stride_dx_m, stride_dx_k,  # dx strides
    stride_dy_b, stride_dy_n, stride_dy_k,  # dy strides
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    This function should recompute z_exp, then use that to compute dx and dy via multiplication with grad_out
     """
    pid_batch = tl.program_id( 0 ) # Batch index
    x_log_ptr += pid_batch * stride_x_b 
    x_sign_ptr += pid_batch * stride_xs_b 
    y_log_ptr += pid_batch * stride_y_b 
    y_sign_ptr += pid_batch * stride_ys_b
    grad_out_ptr += pid_batch * stride_grad_b
    dx_ptr += pid_batch * stride_dx_b
    dy_ptr += pid_batch * stride_dy_b

    pid_m = tl.program_id( 1 ) # M index
    m = pid_m * BLOCK_SIZE_M
    offs_x_m = (m + tl.arange(0, BLOCK_SIZE_M))[:, None]    # (M, 1)

    for n in tl.range( 0, (pid_m + 1) * BLOCK_SIZE_M, BLOCK_SIZE_N):
        offs_y_n = (n + tl.arange(0, BLOCK_SIZE_N))[:, None]    # (N, 1)

        for k in tl.range( 0, K, step=BLOCK_SIZE_K, num_stages=NUM_STAGES ):
            offs_k = (k + tl.arange( 0, BLOCK_SIZE_K ))[None, :]    # (1, K)
            x_log_ptrs = x_log_ptr + offs_x_m * stride_x_m + offs_k * stride_x_k     # (M,1) + (1,K) -> (M,K)
            x_sign_ptrs = x_sign_ptr + offs_x_m * stride_xs_m + offs_k * stride_xs_k # (M,1) + (1,K) -> (M,K)

            y_log_ptrs = y_log_ptr + offs_y_n * stride_y_n + offs_k * stride_y_k     # (N,1) + (1,K) -> (N,K)
            y_sign_ptrs = y_sign_ptr + offs_y_n * stride_ys_n + offs_k * stride_ys_k # (N,1) + (1,K) -> (N,K)

            x_log = tl.load( x_log_ptrs )[:,None,:] # (M,1,K)
            y_log = tl.load( y_log_ptrs )[None,:,:] # (1,N,K)

            x_sign = tl.load( x_sign_ptrs )[:,None,:] # (M,1,K)
            y_sign = tl.load( y_sign_ptrs )[None,:,:] # (1,N,K)
            
            z_sign = x_sign == y_sign

            z_exp = tl.exp( x_log + y_log ) * ( z_sign * 2 - 1 ) # (M,N,K)
            # construct normal mask to ensure pointers don't overflow
            mask = (offs_x_m[:, :, None] < M) & (offs_y_n[None, :, :] < N) & (offs_k[None, :, :] < K)
            # apply triangular mask & normal mask
            z_exp_masked = tl.where( (offs_x_m[:, :, None] > offs_y_n[None, :, :]) & mask, z_exp, 0.0 )
            
            # random mem leaks might occur if there is no masking here
            grad_out_ptrs = grad_out_ptr + offs_x_m * stride_grad_m + offs_y_n.trans(1, 0) * stride_grad_n  # (M, 1) + (1, N)
            grad_mask = (offs_x_m < M) & (offs_y_n < N)
            grad_out = tl.load( grad_out_ptrs, mask=grad_mask )[:,:,None]   # (M,N,1)
            
            dxy = z_exp_masked * grad_out   # (M,N,K) * (M,N,1)
            dx = tl.sum(dxy, -2)    # (M,K)
            dy = tl.sum(dxy, -3)    # (N,K)
            
            dx_ptrs = dx_ptr + offs_x_m * stride_dx_m + offs_k * stride_dx_k     # (M,1) + (1,K) -> (M,K)
            dx_mask = (offs_x_m < M) & (offs_k < K)
            dy_ptrs = dy_ptr + offs_y_n * stride_dy_n + offs_k * stride_dy_k     # (N,1) + (1,K) -> (N,K)
            dy_mask = (offs_y_n < N) & (offs_k < K)
            
            tl.store(dx_ptrs, dx, mask=dx_mask)
            tl.store(dy_ptrs, dy, mask=dy_mask)
    
    
class BeaverMM(torch.autograd.Function):
    @staticmethod
    def forward( ctx, x_log, y_log, x_sign, y_sign ):
        B, M, K = x_log.shape
        _, K, N = y_log.shape
        ctx.save_for_backward(x_log, y_log, x_sign, y_sign)

        assert (K % 32 == 0), "K must be divisible by 32"

        output = torch.zeros((B, M, N), device = x_log.device, dtype = x_log.dtype)
        
        grid = lambda META: (
            B, triton.cdiv(M, META['BLOCK_SIZE_M']),
        )

        kernel_lbmm_fwd[grid](
            x_log, y_log,
            x_sign, y_sign,
            output,
            M, N, K,
            x_log.stride(0), x_log.stride(1), x_log.stride(2),
            y_log.stride(0), y_log.stride(1), y_log.stride(2),
            x_sign.stride(0), x_sign.stride(1), x_sign.stride(2),
            y_sign.stride(0), y_sign.stride(1), y_sign.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
        )
        return output
    
    @staticmethod
    def backward( ctx, grad_output ):
        x_log, y_log, x_sign, y_sign = ctx.saved_tensors
        
        B, M, K = x_log.shape
        _, K, N = y_log.shape
        dx = torch.zeros((B, M, K), device = x_log.device, dtype = x_log.dtype)
        dy = torch.zeros((B, N, K), device = x_log.device, dtype = x_log.dtype)
        
        grid = lambda META: (
            B, triton.cdiv(M, META['BLOCK_SIZE_M']),
        )

        kernel_lbmm_bwd[grid](
            x_log, y_log,
            x_sign, y_sign,
            grad_output,
            dx, dy,
            M, N, K,
            x_log.stride(0), x_log.stride(1), x_log.stride(2),
            y_log.stride(0), y_log.stride(1), y_log.stride(2),
            x_sign.stride(0), x_sign.stride(1), x_sign.stride(2),
            y_sign.stride(0), y_sign.stride(1), y_sign.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
        )
        
        return dx, dy.transpose(-1, -2), None, None

def rwkv_inner(r,k,v,w,u,kv_state,chunk_len:int=128,precision:int=64, do_log=False):
    # assert(chunk_len <= 24 or precision == 64)
    """
    expects
    r : (B,H,L,K)
    k : (B,H,L,K)
    v : (B,H,L,V)
    w : (B,H,L,K) or (1,H,L,K)
    u : (1,H,1,K)
    kv_state : (B,H,K,V)
    """
    B,H,L,K = k.size()
    V = v.size(-1)
    T = chunk_len

    if L == 1:
        kv = k.mT @ v
        out = r @ (kv_state + u.mT * kv)
        kv_state = w.mT * kv_state + kv
        return out, kv_state
    else:
        # FIXME - support fast path for non-exact multiples
        # ensure it's an exact multiple
        assert L%T == 0, "fast non-cuda rwkv5.2+ requires ctxlen to be an exact multiple of chunk_len"

        N = L // T

        # this has to be done to avoid numerical instability (inf/NaN) when w is used as a divisor up to chunk_length//2 places away (so precision_min_val^(T//2) has to be in fp range)
        # NOTE - this does not account for the impact of the size of R, K so we currently use the chunk_len=32 numbers for chunk_len=24
        assert(precision == 32 or precision == 64)
        precision_min_val = 0.005 # good for fp32 (1.175e-38 ^ (1/16.0) < 0.00426)
        if precision == 32:
            precision_dtype = torch.float32
        else: #elif precision_dtype == torch.float64:
            precision_dtype = torch.float64
        w = w.clamp(precision_min_val)

        # calculate cumulative decay in log space where it won't overflow
        w_log = w.float().log() # (1,H,L,K) or (B,H,L,K)

        # print( f'{nan_percent(w_log)=}' )

        # chunked view of w_log
        wc_log = w_log.view(w.size(0),H,N,T,K)
        wc_log_cum = wc_log.cumsum(dim=-2)

        # chunked view of shifted_w_log
        shifted_wc_log_cum = F.pad(wc_log_cum, (0, 0, 1, -1))


        # NOTE - we have to apply the decay weight from TWO ahead.. ONE ahead gets no decay (log==0)
        # pre-applied weights
        # left side is prior chunk (w_inter), right side is current chunk (w_intra)
        # without u...
        # w0   w1   w2   w3   | w4   w5   w6   w7
        # w1:4 w2:4 w3:4 w4:4 | w4:5 w4:6 w4:7 w4:8
        # with u...
        # w0   w1   w2   w3   | w4   w5   w6   w7
        # w1:4 w2:4 w3:4 w4:4 | w4:4 w4:5 w4:6 w4:7

        # ws decays the entire current state (representing t-1) to the prior block (t-2)
        ws = wc_log.sum(dim=-2, keepdim=True) # 1HN1K or BHN1K
        # w_inter is the decay to the end of the current block, since it will be applied at the next iteration when current (t) becomes prior (t-1)
        # this formula because e.g. w1:4 = w0:4 - w0:1
        w_inter = ws - wc_log_cum # 1HNTK or BHNTK (w^(T-1) ... w^0)
        # w_intra is the decay from the beginning of the current block (t), since it will be applied to current queries (t) against prior state (representing keys+values up to but not including block t)
        # this formula because e.g. w1:3 = w0:3 - w0
        w_intra = wc_log_cum - wc_log # 1HNTK or BHNTK (w^0 ... w^(T-2))

        ws = list(ws.mT.exp().to(r.dtype).unbind(dim=-3)) # N x 1HK1 or BHK1 !!NOTE THE .mT HERE!!
        w_inter = w_inter.exp().to(r.dtype) # 1HNTK or BHNTK
        w_intra = w_intra.exp().to(r.dtype) # 1HNTK or BHNTK

        # chunked view of r, k, v
        r = r.view(B,H,N,T,K)
        k = k.view(B,H,N,T,K)
        v = v.view(B,H,N,T,V)
        u = u.unsqueeze(2).to(r.dtype) # (1,H,1,1,K)

        # parallel calculation of all intra-chunk attention contributions
        wc_log_offset = shifted_wc_log_cum[...,T//2:T//2+1,:] # B,H,N,1,K


        if do_log:
            r_decay_logged = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype)
            k_inv_decay_logged = (wc_log_offset - wc_log_cum).to(precision_dtype)

            r_log, r_sign = r.abs().log(), r > 0
            rxr = ( r_log + r_decay_logged )

            k_log, k_sign = k.abs().log(), k > 0
            kxk = ( k_log + k_inv_decay_logged )

            # a = lse_mm_real( rxr, kxk, r_sign, k_sign ).to( r.dtype ).tril(-1)

            r_shape = rxr.shape
            k_shape = kxk.shape

            r_shape2 = [ -1, *r_shape[ -2 : ] ]
            k_shape2 = [ -1, *k_shape[ -2 : ] ]
            o_shape = [ *r_shape[ : -2 ], r_shape[-2], k_shape[-2] ]

            # uncomment the following if u want to run gradcheck in main loop, and also uncomment the one line in the kernel for float64
            """
            check_grad_impl = True
            if check_grad_impl:
                input = (rxr.view( r_shape2 ),
                            kxk.view( k_shape2 ).mT.contiguous(),
                            r_sign.view( r_shape2 ),
                            k_sign.view( k_shape2 ).mT.contiguous())
                print("checking BeaverMM gradients")
                test = torch.autograd.gradcheck(BeaverMM.apply, input, eps=1e-6, atol=1e-4)
                print(f"result of gradcheck: {test}")
            """
            
            a = BeaverMM.apply(
                rxr.view( r_shape2 ),
                kxk.view( k_shape2 ).mT.contiguous(),
                r_sign.view( r_shape2 ),
                k_sign.view( k_shape2 ).mT.contiguous()
            ).view( o_shape ).to(r.dtype)
        else:
            r_decay = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype).exp() # B,H,N,T,K
            k_inv_decay = (wc_log_offset - wc_log_cum).to(precision_dtype).exp() # B,H,N,T,K
            a = ((r*r_decay) @ (k*k_inv_decay).mT).to(r.dtype).tril(-1) # B,H,N,T,T


        # add u term to attention (NOTE - the tril(-1) above zeroed the diagonal)
        a = a + torch.einsum('bhntk,bhntk->bhnt', r, u * k).diag_embed()
        out = a @ v # BHNTV
        # alternate way of adding in u
        # out = out + torch.einsum('bhntk,bhntk,bhntv->bhntv', r, u * k, v)

        # parallel precalculation of chunked (k*wk).mT@v for use in recurrent state calc below
        wkv = (k * w_inter).mT @ v # BHNKV
        wkv = list(wkv.unbind(dim=-3)) # N x BHKV

        # recurrent calculation of all states
        states = []
        for i in range(N):
            states.append(kv_state)
            kv_state = kv_state * ws[i] + wkv[i] # BHKV
            # equivalent non-precalced version
            #wkv = (k[...,i,:,:] * wk[...,i,:,:]).mT @ v[...,i,:,:]
            #kv_state = kv_state * ws[i] + wkv
        states = torch.stack(states, dim=2) # BHNKV

        # parallel application of all r to states
        out = out + (r * w_intra) @ states # BHNTV
        out = out.view(B,H,L,V)
        return out, kv_state, a