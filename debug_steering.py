import torch
import torch.nn.functional as F
from safe_eos_aligner import SafeEOSAligner
from build_subspace import SubspaceBuilder
from transformers import CLIPTokenizer, CLIPTextModel

def test_steering_effect():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Build Real Subspace
    print("Building subspace...")
    builder = SubspaceBuilder(device=device)
    pairs, safety_pairs = builder.generate_pairs_from_json("data/modifiers.json", 200)
    U, lam, v_safe = builder.build(pairs, safety_pairs=safety_pairs, k=5, ridge=60.0)
    
    # 2. Get Real Embedding for "Shirtless"
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
    
    prompt = "Shirtless Putin at pride"
    inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = text_encoder(inputs.input_ids)[0]
    
    # 3. Setup Aligner
    # "Shirtless" is index 1 (startoftext is 0)
    # Check token string
    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0]]
    print(f"Tokens: {tokens[:5]}")
    
    # Target "shirtless" token (index 1)
    target_idx = 1
    target_emb = emb[0, target_idx, :].unsqueeze(0).unsqueeze(0) # (1, 1, D)
    
    aligner = SafeEOSAligner(
        U=U, lam=lam, v_safe=v_safe,
        tau=1.5,
        T=0.2,
        alpha_max=0.6,
        top_m=3,
        eta=0.5,
        steering_scale=10.0,
        align_mode="steer",
        device=device
    )
    
    # Initial PC1
    pc1_pre = torch.matmul(target_emb, aligner.U)[..., 0].item()
    print(f"[Initial] Score: {aligner.get_score(target_emb).item():.4f}, PC1: {pc1_pre:.4f}")
    
    # Step simulation
    for i in range(5):
        # Accumulate
        target_emb = aligner.edit_embeddings(target_emb, step=i, num_steps=50)
        
        stat = aligner.stats[-1]
        s_post = stat["post_scores"].item()
        pc1_val = stat.get("pc1_values", torch.tensor(0)).item() # This logs PRE value
        
        # Calculate Post PC1 manually
        pc1_post = torch.matmul(target_emb, aligner.U)[..., 0].item()
        
        print(f"[Step {i}] PrePC1: {pc1_val:.4f} -> PostPC1: {pc1_post:.4f} | Limit: {stat['scaling_applied_pct']:.0f}%")

if __name__ == "__main__":
    test_steering_effect()
