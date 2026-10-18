"""Candidate gradient-lifetime optimization; no training entry or GPU evidence.

Same MasterAdamW constructor/state API and AdamW math. Only model-grad lifetime
changes: retain the converted/aliased grad on each FP32 master before dropping
its model reference. Callers must finish clipping/probes before step, and must
not rely on model.grad after step. A live FP32 alias stays owned by master.grad.
"""
from experiments.pi05_g1.training import MasterAdamW as _OriginalMasterAdamW


def release_unused_cuda_cache(devices):
    """Only existing CUDA devices owned by this optimizer; no CPU CUDA context."""
    import torch
    for device in devices:
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


class MasterAdamW(_OriginalMasterAdamW):
    """Identical optimizer math; release model grads before lazy Adam moments."""
    def step(self):
        import torch
        devices = []
        for parameter, master in zip(self.model_parameters, self.master_parameters, strict=True):
            master.grad = None if parameter.grad is None else parameter.grad.detach().float()
            # For FP32 this aliases storage, retained by master.grad; for BF16
            # this drops the original gradient after its FP32 copy was queued.
            parameter.grad = None
            if parameter.device.type == "cuda" and parameter.device not in devices:
                devices.append(parameter.device)
        release_unused_cuda_cache(devices)
        result = self.optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(self.model_parameters, self.master_parameters, strict=True):
                parameter.copy_(master)
        return result
