import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

def main():
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    x = torch.randn(2048, 2048, device='cuda', requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(2048, 128, device='cuda') * .01)
    before = weight.detach().clone()
    opt = torch.optim.SGD([weight], lr=.1)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = (x @ weight).square().mean()
        loss.backward()
        assert torch.isfinite(weight.grad).all()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    delta = (weight.detach()-before).abs().max().item()
    assert delta > 0 and losses[-1] < losses[0]
    ckpt = ROOT / 'models' / 'gpu-verification.pt'
    torch.save({'weight':weight.detach().cpu(), 'losses':losses}, ckpt)
    result = {'python':platform.python_version(), 'torch':torch.__version__,
              'cuda_runtime':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
              'gpu':torch.cuda.get_device_name(0), 'capability':torch.cuda.get_device_capability(0),
              'compiled_architectures':torch.cuda.get_arch_list(),
              'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,
              'losses':losses,'max_parameter_delta':delta,
              'seconds':time.perf_counter()-start,
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'checkpoint_sha256':hashlib.file_digest(ckpt.open('rb'),'sha256').hexdigest(),
              'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'], text=True)}
    (ROOT / 'reports' / 'environment.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
