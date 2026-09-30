"""W4A16 kernel latency and correctness across medium/large M; run after build.py."""
import ctypes
import torch
import w4a16


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / iters


def main():
    torch.manual_seed(11)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    for m in (128, 512, 777, 1024, 1536, 2048, 8192):
        n, k = 4096, 3840
        weight = torch.randn((n, k), device='cuda') * 0.02
        packed, scales = w4a16.quantize(weight)
        del weight
        x = torch.randn((m, k), device='cuda').to(torch.bfloat16)
        out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        lib = w4a16._load()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        def kernel():
            rc = lib.w4a16_gemm(x.data_ptr(), packed.data_ptr(), scales.data_ptr(), out.data_ptr(),
                                 m, n, k, x.stride(0), out.stride(0), stream)
            if rc:
                raise RuntimeError(f'launch failed: {rc}')

        kernel()
        ref = torch.nn.functional.linear(x, w4a16.dequantize(packed, scales))
        torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(out.float(), ref.float(), atol=0.125, rtol=0.02)
        t_ms = bench(kernel)
        print(f'M={m:5} N={n} K={k} latency={t_ms * 1000:.1f} us max_abs={err:.4g}', flush=True)


if __name__ == '__main__':
    main()
