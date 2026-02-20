import torch
import torch.nn as nn
import json
import argparse

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

class SubspaceBuilder:
    """
    Builds a concept subspace (U, lam) from contrastive prompt pairs.
    """
    def __init__(self, device="cuda"):
        self.device = device
        # Load CLIP Text Encoder & Tokenizer (SD v1.4 default)
        self.model_id = "CompVis/stable-diffusion-v1-4"
        print(f"[SubspaceBuilder] Loading CLIP Text Encoder from {self.model_id}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder").to(self.device)
        self.text_encoder.eval()

    def generate_pairs_from_json(self, json_path, num_pairs=200, pair_mode="neutral_unsafe"):
        """
        Generates prompt pairs from a JSON file.
        
        Args:
            json_path: Path to json file.
            num_pairs: Number of pairs to generate.
            pair_mode: "safe_unsafe" (Legacy) or "neutral_unsafe" (New).
                       - safe_unsafe: (Unsafe, Safe) pairs.
                       - neutral_unsafe: (Unsafe, Neutral) pairs for Subspace,
                                         plus (Safe, Neutral) pairs for Safe Direction.
        
        Returns:
            subspace_pairs: List of (target, baseline) for subspace PCA.
            safety_pairs: List of (safe, neutral) for safe direction (only in neutral_unsafe mode), else None.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        unsafe_mods = data.get("unsafe", [])
        safe_mods = data.get("safe", [])
        subjects = data.get("subjects", ["person"])
        templates = data.get("templates", ["a photo of a {modifier} {subject}"])
        
        subspace_pairs = []
        safety_pairs = []
        
        import random
        
        candidates = []
        
        # Generate enough candidates
        # We will sample on the fly to avoid explosion
        
        for _ in range(num_pairs * 2): # Over-sample
            t = random.choice(templates)
            s = random.choice(subjects)
            u = random.choice(unsafe_mods)
            
            # Construct Unsafe Prompt
            p_unsafe = t.format(modifier=u, subject=s)
            
            if pair_mode == "safe_unsafe":
                # Legacy: Unsafe vs Safe
                s_mod = random.choice(safe_mods)
                p_safe = t.format(modifier=s_mod, subject=s)
                candidates.append(((p_unsafe, p_safe), None))
                
            elif pair_mode == "neutral_unsafe":
                # New: Unsafe vs Neutral (Baseline)
                # Neutral: remove modifier. 
                # formatting "{modifier} {subject}" with "" might leave space.
                # Clean up double spaces.
                p_neutral = t.format(modifier="", subject=s).replace("  ", " ").strip()
                
                # Subspace pair: (Unsafe, Neutral)
                pair_sub = (p_unsafe, p_neutral)
                
                # Safety pair: (Safe, Neutral)
                s_mod = random.choice(safe_mods)
                p_safe = t.format(modifier=s_mod, subject=s)
                pair_safe = (p_safe, p_neutral)
                
                candidates.append((pair_sub, pair_safe))
            else:
                raise ValueError(f"Unknown pair_mode: {pair_mode}")

        # Unique & Shuffle
        unique_candidates = list(set(candidates))
        random.shuffle(unique_candidates)
        selected = unique_candidates[:num_pairs]
        
        subspace_pairs = [x[0] for x in selected]
        safety_pairs = [x[1] for x in selected] if pair_mode == "neutral_unsafe" else None
        
        return subspace_pairs, safety_pairs

    def get_embeddings(self, prompts):
        """
        Get EOS token embeddings for a list of prompts.
        Shape: (B, 768)
        """
        inputs = self.tokenizer(
            prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
            # last_hidden_state: (B, 77, 768)
            z = outputs.last_hidden_state
            
        eos_token_id = self.tokenizer.eos_token_id
        # Find position of EOS
        eos_indices = (input_ids == eos_token_id).int().argmax(dim=-1) # First occurrence
        
        e_eos = z[torch.arange(z.shape[0]), eos_indices] # (B, 768)
        return e_eos

    def build(self, 
              pairs, 
              safety_pairs=None,
              k=1, 
              ridge=0.0):
        """
        Compute subspace U (eigenvectors) and lam (eigenvalues).
        Also computes v_safe if safety_pairs is provided.
        """
        if not pairs:
            raise ValueError("No pairs provided")

        print(f"[SubspaceBuilder] Computing embeddings for {len(pairs)} pairs...")
        
        # Batch processing to avoid OOM
        batch_size = 32
        
        # 1. Process Subspace Pairs
        unsafe_prompts = [p[0] for p in pairs]
        base_prompts = [p[1] for p in pairs]
        
        E_unsafe_list = []
        E_base_list = []
        
        for i in range(0, len(pairs), batch_size):
            batch_u = unsafe_prompts[i:i+batch_size]
            batch_b = base_prompts[i:i+batch_size]
            
            E_unsafe_list.append(self.get_embeddings(batch_u))
            E_base_list.append(self.get_embeddings(batch_b))
            
        E_unsafe = torch.cat(E_unsafe_list, dim=0) # (N, 768)
        E_base = torch.cat(E_base_list, dim=0)     # (N, 768)
        
        # Difference Vectors
        D = E_unsafe - E_base # (N, 768)
        
        # PCA
        D = D.float() # Ensure fp32
        N = D.shape[0]
        
        # Centering
        D_mean = torch.mean(D, dim=0, keepdim=True)
        D_centered = D - D_mean
        
        # Covariance Matrix: (768, 768)
        C = (D_centered.T @ D_centered) / (N - 1)
        
        if ridge > 0:
            C = C + ridge * torch.eye(C.shape[0], device=self.device)
            
        # Eigen decomposition
        eigvals, eigvecs = torch.linalg.eigh(C)
        
        # Sort descending
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Select Top-k
        U = eigvecs[:, :k]   # (768, k)
        lam = eigvals[:k]    # (k,)
        
        # Enforce direction consistency
        # Align U such that the mean difference vector has positive projection
        # D_mean: (1, 768)
        # U: (768, k)
        proj = torch.matmul(D_mean, U) # (1, k)
        signs = torch.sign(proj) # (1, k)
        U = U * signs # Flip columns where projection is negative
        
        # Explain variance
        total_var = torch.sum(eigvals)
        explained_var = torch.sum(lam)
        ratio = explained_var / total_var
        print(f"[SubspaceBuilder] First {k} components explain {ratio.item()*100:.2f}% of variance.")
        
        # 2. Compute Safe Direction (if available)
        v_safe = None
        if safety_pairs is not None:
            print(f"[SubspaceBuilder] Computing Safe Direction from {len(safety_pairs)} pairs...")
            safe_prompts = [p[0] for p in safety_pairs]
            neut_prompts = [p[1] for p in safety_pairs]
            
            D_safe_list = []
            for i in range(0, len(safety_pairs), batch_size):
                b_s = safe_prompts[i:i+batch_size]
                b_n = neut_prompts[i:i+batch_size]
                
                e_s = self.get_embeddings(b_s)
                e_n = self.get_embeddings(b_n)
                D_safe_list.append(e_s - e_n)
            
            D_safe_all = torch.cat(D_safe_list, dim=0) # (N, 768)
            v_safe = torch.mean(D_safe_all, dim=0) # (768,)
            print("[SubspaceBuilder] Safe Direction computed.")

        return U, lam, v_safe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="data/modifiers.json")
    parser.add_argument("--num_pairs", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pair_mode", type=str, default="neutral_unsafe", choices=["safe_unsafe", "neutral_unsafe"])
    parser.add_argument("--output_path", type=str, default="data/subspace.pt")
    args = parser.parse_args()
    
    builder = SubspaceBuilder()
    pairs, safety_pairs = builder.generate_pairs_from_json(args.json_path, args.num_pairs, pair_mode=args.pair_mode)
    U, lam, v_safe = builder.build(pairs, safety_pairs=safety_pairs, k=args.k)
    
    # Save
    data = {"U": U, "lam": lam}
    if v_safe is not None:
        data["v_safe"] = v_safe
        
    torch.save(data, args.output_path)
    print(f"Saved subspace to {args.output_path}")

