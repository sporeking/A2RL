import torch
import gc
import psutil
import numpy as np
from collections import defaultdict

class MemoryTracker:
    def __init__(self, device):
        self.device = device
        self.memory_stats = defaultdict(list)
    
    def get_tensor_details(self):
        """获取所有tensor的详细信息"""
        tensor_details = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == 'cuda':
                    tensor_info = {
                        'size': tuple(obj.size()),
                        'dtype': str(obj.dtype),
                        'memory_mb': obj.element_size() * obj.nelement() / 1024**2,
                        'requires_grad': obj.requires_grad,
                        'grad_fn': str(obj.grad_fn) if obj.grad_fn else None
                    }
                    tensor_details.append(tensor_info)
            except:
                continue
        return tensor_details

    def log_memory(self, tag=''):
        """记录当前内存使用情况"""
        gpu_mem = self.get_gpu_memory()
        if gpu_mem:
            self.memory_stats[f'{tag}_gpu'].append(gpu_mem)
            tensor_details = self.get_tensor_details()
            self.memory_stats[f'{tag}_tensors'].append(tensor_details)
            
            print(f"\n=== Memory Usage ({tag}) ===")
            print(f"GPU {self.device}:")
            print(f"  Allocated: {gpu_mem['allocated']:.1f}MB")
            print(f"  Reserved:  {gpu_mem['reserved']:.1f}MB")
            print(f"  Cached:    {gpu_mem['cached']:.1f}MB")
            
            if tensor_details:
                print("\nLargest tensors:")
                # 按内存使用量排序
                sorted_tensors = sorted(tensor_details, key=lambda x: x['memory_mb'], reverse=True)[:5]
                for tensor in sorted_tensors:
                    print(f"  Size: {tensor['size']}")
                    print(f"    Memory: {tensor['memory_mb']:.1f}MB")
                    print(f"    dtype: {tensor['dtype']}")
                    print(f"    requires_grad: {tensor['requires_grad']}")
                    print(f"    grad_fn: {tensor['grad_fn']}")
            print("========================\n")
            
    def get_gpu_memory(self):
        """获取GPU显存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            return {
                'allocated': allocated,
                'reserved': reserved,
                'cached': reserved - allocated
            }
        return None 