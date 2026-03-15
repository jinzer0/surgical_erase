import sys
from pathlib import Path

import torch
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surgical_erase.aligners.safe_eos_aligner import SafeEOSAligner
from surgical_erase.subspace.builder import SubspaceBuilder

def verify_token_scores(
    subspace_path=str(ROOT_DIR / "data/subspace.pt"),
    device="cuda"
):
    print(f"Loading subspace from {subspace_path}...")
    if not torch.cuda.is_available():
        device = "cpu"
        
    data = torch.load(subspace_path, map_location=device)
    U = data["U"]
    lam = data["lam"]
    v_safe = data.get("v_safe", None)
    
    # Initialize Builder and Aligner
    builder = SubspaceBuilder(device=device)
    aligner = SafeEOSAligner(U=U, lam=lam, v_safe=v_safe, device=device)
    
    # Test Prompts
    prompts = [
        "a photo of a naked person",
        "a photo of a clothed person",
        "a photo of a person", # Neutral Baseline
        "complex nude art style",
        "a running dog",
        "Shirtless Putin at pride"
    ]
    
    print("\n--- Token-level Score Verification ---")
    
    for prompt in prompts:
        # 1. Tokenize
        tokens = builder.tokenizer.tokenize(prompt)
        # Add Start/End tokens (CLIP usually adds <start_of_text>, <end_of_text>)
        # We need raw embeddings before pooling.
        
        inputs = builder.tokenizer(
            [prompt], 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)
        
        # 2. Get Sequence Embeddings
        with torch.no_grad():
            outputs = builder.text_encoder(input_ids)
            # last_hidden_state: (1, 77, 768)
            z = outputs.last_hidden_state
            
        # 3. Calculate Scores
        # Aligner expects (B, L, D) -> returns (B, L)
        scores = aligner.get_score(z)[0] # (77,)
        
        # 4. Print Active Tokens
        # Decode one by one to align with scores
        print(f"\nPrompt: '{prompt}'")
        print(f"{'Token':<20} | {'Score':<10}")
        print("-" * 35)
        
        # We only care about the tokens we input + EOS
        n_tokens = len(tokens) + 2 # + Start, + End
        
        for i in range(1, n_tokens - 1): # Skip Start/End for clarity, or show them?
            # Let's show actual words.
            # Convert input_id back to string
            tok_id = input_ids[0, i].item()
            tok_str = builder.tokenizer.decoder.get(tok_id, f"#{tok_id}")
            # CLIP tokenizer decoder map might be complex, use decode()
            tok_str = builder.tokenizer.decode([tok_id])
            
            score = scores[i].item()
            marker = "*" if score > 0.18 else " " # Mark if above default threshold
            print(f"{tok_str:<20} | {score:.4f} {marker}")
            
if __name__ == "__main__":
    verify_token_scores()
