import sys
from pathlib import Path

import torch
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surgical_erase.subspace.builder import SubspaceBuilder

def inspect_direction():
    device = "cpu"
    # Load subspace or build it
    # We assume the user is using default pairs or modifiers.
    # Let's try to load if exists, or simple build.
    # The command used --ridge 60.
    
    print("Building subspace with ridge=60, num_pairs=10...")
    builder = SubspaceBuilder(device=device)
    # Generate standard pairs
    pairs, safety_pairs = builder.generate_pairs_from_json(str(ROOT_DIR / "data/modifiers_v2.json"), 10)
    U, lam, v_safe = builder.build(pairs, safety_pairs=safety_pairs, k=5, ridge=60.0)
    
    # Check PC1 (Nudity Axis)
    u0 = U[:, 0]
    
    # Check v_safe projection on u0
    # v_safe . u0
    proj = torch.dot(v_safe, u0).item()
    v_norm = torch.norm(v_safe).item()
    cosine = proj / (v_norm + 1e-8)
    
    print(f"v_safe . U[:, 0] (PC1): {proj:.4f}")
    print(f"v_safe Norm: {v_norm:.4f}")
    print(f"Cosine Similarity (Efficiency): {cosine:.4f}")
    
    if abs(cosine) < 0.1:
         print("WARNING: v_safe is nearly orthogonal to Nudity Axis! Steering is inefficient.")
    
    if proj > 0:
        print("WARNING: v_safe is ALIGNED with Nudity Axis (Positive). Steering will ADD nudity.")
    else:
        print("OK: v_safe is ANTI-ALIGNED with Nudity Axis (Negative). Steering should REMOVE nudity.")
        
    # Also check "Shirtless Putin" projection
    from transformers import CLIPTokenizer, CLIPTextModel
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    
    prompt = "Shirtless Putin at pride"
    inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        emb = text_encoder(inputs.input_ids)[0] # (1, 77, 768)
        
    # Check projection of some tokens
    # "Shirtless"
    # Tokenize words
    words = ["Shirtless", "Putin", "pride"]
    for w in words:
        w_id = tokenizer(w, add_special_tokens=False).input_ids[0]
        # Find index in prompt? 
        # Just approximate by encoding single word? No context mismatch.
        # Let's use the full embedding.
        pass

    # Project full embedding
    # (1, 77, D) @ (D, k) -> (1, 77, k)
    coeffs = torch.matmul(emb, U)
    pc1_scores = coeffs[0, :, 0] # (77,)
    
    print(f"\nPrompt: {prompt}")
    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0]]
    
    for i, t in enumerate(tokens):
        if t == "<|endoftext|>": break
        if pc1_scores[i].abs() > 0.5:
            print(f"Token: {t:<15} PC1: {pc1_scores[i]:.4f}")

if __name__ == "__main__":
    inspect_direction()
