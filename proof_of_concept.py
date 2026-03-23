import torch
import time
from typing import Optional, Dict

class STAllocator:
  
    def __init__(self, device: int = 0):
        self.device = f'cuda:{device}'
        self.pools = {
            "persistent": [], # Weights, KV-Cache
            "transient": [],  # Activations, Gradients
        }
        self.stats = {"fragmentation_ratio": 0.0}

    def predict_lifespan(self, tensor_metadata: Dict) -> str:
       
        if tensor_metadata.get('is_parameter', False):
            return "persistent"
        return "transient"

    def allocate(self, size: int, is_parameter: bool = False):
        category = self.predict_lifespan({'is_parameter': is_parameter})
     
        print(f"[ST-Alloc] Routing {size/1e6:.2f}MB to {category} pool")
        
        return torch.empty(size, dtype=torch.uint8, device=self.device)

 
def patch_pytorch_allocator():
   pass
  # code is in private repo
 
