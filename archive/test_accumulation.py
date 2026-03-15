import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surgical_erase.aligners.safe_eos_aligner import SafeEOSAligner

def test_accumulation():
    # Mock data
    D = 768
    k = 5
    U = torch.randn(D, k)
    # Unit vector 0
    U[:, 0] = torch.zeros(D)
    U[0, 0] = 1.0 # First dim is nudity
    
    # Safe direction (anti-aligned)
    v_safe = -1.0 * U[:, 0] 
    
    aligner = SafeEOSAligner(
        U=U, v_safe=v_safe,
        tau=0.5, 
        steering_scale=2.0,
        eta=0.1, # 10% change allowed
        align_mode="steer",
        device="cpu"
    )
    
    # Initial embedding (high score)
    x = U[:, 0].clone().unsqueeze(0).unsqueeze(0) * 2.0
    print(f"Step 0 Score: {aligner.get_score(x).item():.4f}")
    
    # Simulate accumulation Loop
    current_x = x.clone()
    for i in range(10):
        # Update current_x in place (simulate pipeline)
        current_x = aligner.edit_embeddings(current_x, step=i, num_steps=10)
        s = aligner.get_score(current_x).item()
        
        # Check PC1 (Nudity Axis)
        pc1 = torch.matmul(current_x, aligner.U[:, 0]).item()
        print(f"Step {i+1} Score: {s:.4f}, PC1: {pc1:.4f}")
        
    print(f"Final reduction: {2.0 - s:.4f}")

if __name__ == "__main__":
    test_accumulation()
