import torch

print(f"torch: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"Built with CUDA? {torch.backends.cuda.is_built()}")
print(f"cuDNN enabled? {torch.backends.cudnn.enabled}")

if torch.cuda.is_available():
    num = torch.cuda.device_count()
    print(f"# of CUDA devices: {num}")
    for i in range(num):
        props = torch.cuda.get_device_properties(i)
        print(f"[{i}] {props.name} | CC {props.major}.{props.minor} | "
              f"{props.total_memory/1024**3:.2f} GB VRAM")
    cur = torch.cuda.current_device()
    print(f"Current device index: {cur} ({torch.cuda.get_device_name(cur)})")

    # 메모리/연산 간단 검증
    try:
        a = torch.randn(1024, 1024, device='cuda')
        b = torch.matmul(a, a.t())  # 작은 연산
        free, total = torch.cuda.mem_get_info()
        print(f"CUDA mem: free {free/1024**3:.2f} GB / total {total/1024**3:.2f} GB")
        print("CUDA tensor op OK:", b.is_cuda)
    except Exception as e:
        print("CUDA compute test failed:", e)
else:
    print("CUDA not available. Likely causes:")
    print("- CUDA 지원 빌드가 아닌 PyTorch 설치")
    print("- NVIDIA 드라이버/툴킷 미설치 또는 버전 불일치")
    print("- GPU가 없거나 WSL/컨테이너에서 패스스루 미설정")
