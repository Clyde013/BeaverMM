# BeaverMM
Triton kernel implementation of logspace matrix multiplication implemented for RWKV-6.

<details>
	<summary>Why's it called Beaver?</summary>
	Because we're using <i>logs</i> to prevent <i>overflow</i>
	<br>
	<b>BA-DUM-TSSS</b>
</details>

## Results
```
matmul-performance-k64:
       L     Torch  Torch_Compiled    Triton       FLA
0   64.0  0.416768        0.275456  0.190464  0.074752
1  128.0  0.990208        0.799744  0.517120  0.147456
2  192.0  2.095104        1.625088  1.009696  0.230400
3  256.0  3.501104        2.677760  1.558528  0.348064
```

## TODO
1. More modular debugging tools like gradcheck unit tests, just QoL script to improve the experience of iterating on the kernel code.
1. Some form of swizzling for beaver kernel implementation to increase L2 cache hitrate and maybe close the gap with FLA a bit.
