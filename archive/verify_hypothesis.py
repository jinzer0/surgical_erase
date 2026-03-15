import sys
from pathlib import Path

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surgical_erase.subspace.builder import SubspaceBuilder

def verify_geometric_hypothesis(
    subspace_path=str(ROOT_DIR / "data/subspace.pt"),
    device="cuda"
):
    print(f"Loading subspace from {subspace_path}...")
    if not torch.cuda.is_available():
        device = "cpu"
        
    data = torch.load(subspace_path, map_location=device)
    U = data["U"].to(device) # (768, k)
    
    # 1. Define Test Prompts representing the spectrum
    prompts = [
        "a photo of a naked person",    # Unsafe
        "a photo of a person",          # Neutral
        "a photo of a clothed person",  # Safe
        "a running dog", # Irrelevant
        "Shirtless Putin at pride"
    ]
    
    print("\n--- Geometric Verification ---")
    print(f"Test Prompts: {prompts}")
    
    # 2. Get Embeddings
    builder = SubspaceBuilder(device=device)
    embeddings = builder.get_embeddings(prompts) # (B, 768)
    
    # 3. Project onto the Subspace (1st Principal Component)
    # The 1st PC represents the strongest direction of "Nudity" (according to our construction)
    u0 = U[:, 0] # (768,)
    
    # Calculate scalar projection: <e, u0>
    projections = torch.matmul(embeddings, u0).cpu().numpy()
    
    # 4. Analyze Results
    print("\nScalar Projection on Nudity Axis (First Component):")
    print("(Positive = More Nude, Negative = More Safe/Opposite)")
    
    results = []
    for p, val in zip(prompts, projections):
        print(f"  '{p}': {val:.4f}")
        results.append((p, val))
        
    # Check Hypothesis: Naked > Person > Clothed
    val_naked = results[0][1]
    val_neutral = results[1][1]
    val_clothed = results[2][1]
    
    if val_naked > val_neutral:
        print("\n[SUCCESS] Unsafe > Neutral: Nudity direction is correctly oriented.")
    else:
        print("\n[FAIL] Unsafe < Neutral: Direction might be flipped.")
        
    if val_neutral > val_clothed:
        print("[SUCCESS] Neutral > Safe: 'Clothed' is further in the negative direction.")
    else:
        print("[WARNING] Neutral <= Safe: 'Clothed' is not more negative than Neutral. (Subspace might be dominated by other features)")

    # 5. Check v_safe
    if "v_safe" in data:
        v_safe = data["v_safe"].to(device)
        v_safe_proj = torch.dot(v_safe, u0).item()
        print(f"\nProjection of v_safe on Nudity Axis: {v_safe_proj:.4f}")
        
        if v_safe_proj < 0:
             print("[SUCCESS] v_safe is effectively anti-aligned with Nudity.")
        else:
             print("[WARNING] v_safe has positive projection on Nudity axis.")

    # Check Variance Ratio (if eigenvalues stored)
    if "lam" in data:
        lam = data["lam"].cpu()
        print(f"\nTop Eigenvalues: {lam}")
        # We can't know total variance without full decomposition, 
        # but we can see relative strength of top components.
        if len(lam) > 1:
            ratio_1_2 = lam[0] / lam[1]
            print(f"Ratio 1st/2nd Eigenvalue: {ratio_1_2:.2f}")
            # Note: Explicit explained variance (lam/sum(eig)) might be low if ridge is high,
            # because ridge adds a large constant to ALL eigenvalues (increasing denominator).
            # But the relative ratio (lam[0]/lam[1]) should still show dominance.
    
    # 6. Visualization (Simple Plot)
    plt.figure(figsize=(10, 4))
    y_pos = np.arange(len(prompts) + 1)
    vals = [r[1] for r in results]
    labels = [r[0] for r in results]
    
    if "v_safe" in data:
        vals.append(v_safe_proj)
        labels.append("v_safe (Direction)")
        
    colors = ['red', 'gray', 'blue', 'green', 'purple']
    
    plt.barh(y_pos, vals, align='center', color=colors[:len(vals)])
    plt.yticks(y_pos, labels)
    plt.xlabel('Projection on Nudity Axis (u0)')
    plt.title('Geometric Verification of Nudity Subspace')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "outputs/hypothesis_verification.png")
    print("\nSaved visualization to 'hypothesis_verification.png'")

if __name__ == "__main__":
    verify_geometric_hypothesis()
