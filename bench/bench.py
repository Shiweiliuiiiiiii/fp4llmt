import torch

def benchmark_fp4(iterations = 100):
    print("Benchmarking FP4 matmul")
    M, K, N = 1024, 1024, 1024
    # Create matrices directly in half precision.
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    A = A.view(torch.float4_e2m1fn_x2)
    B = B.view(torch.float4_e2m1fn_x2)

    # Warm-up iterations.
    for _ in range(10):
        C = torch.matmul(A.view(torch.bfloat16), B.view(torch.bfloat16)).view(torch.float4_e2m1fn_x2)
        torch.cuda.synchronize()

    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        C = torch.matmul(A.view(torch.bfloat16), B.view(torch.bfloat16)).view(torch.float4_e2m1fn_x2)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP4 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))


def benchmark_fp8(iterations = 100):
    print("Benchmarking FP8 matmul")
    M, K, N = 1024, 1024, 1024
    # Create matrices directly in half precision.
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    A = A.view(torch.float8_e4m3fn)
    B = B.view(torch.float8_e4m3fn)

    # Warm-up iterations.
    for _ in range(10):
        C = torch.matmul(A.view(torch.bfloat16), B.view(torch.bfloat16)).view(torch.float8_e4m3fn)
        torch.cuda.synchronize()

    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        C = torch.matmul(A.view(torch.bfloat16), B.view(torch.bfloat16)).view(torch.float8_e4m3fn)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP8 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))

def benchmark_fp16(iterations = 100):
    print("Benchmarking FP16 matmul")
    M, K, N = 1024, 1024, 1024
    # Create matrices directly in half precision.
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    # Warm-up iterations.
    for _ in range(10):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()

    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP16 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))

def benchmark_fp32(iterations = 100):
    print("Benchmarking FP32 matmul")
    M, K, N = 1024, 1024, 1024
    # Create matrices directly in half precision.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # Warm-up iterations.
    for _ in range(10):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()

    # Benchmark over multiple iterations using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print("FP32 matmul average time: {:.3f} ms".format(elapsed_ms / iterations))

if __name__ == "__main__":
    print("Bench 32")
    benchmark_fp32(1000)
    print("Bench 16")
    benchmark_fp16(1000)
    print("Bench 8")
    benchmark_fp8(10000)
    print("Bench 4")
    benchmark_fp4(10000)

"""
Analyzed timings A100:
FP32 matmul average time: 0.129 ms
FP16 matmul average time: 0.019 ms
FP8 matmul average time: 0.019 ms
FP4 matmul average time: 0.018 ms

Analyzed timings H100:
FP32 matmul average time: 0.056 ms
FP16 matmul average time: 0.011 ms
FP8 matmul average time: 0.010 ms
FP4 matmul average time: 0.439 ms

"""

