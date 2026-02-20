import torch
import os

def verify():
    path = "data/subspace.pt"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    data = torch.load(path)
    U = data["U"]
    lam = data["lam"]
    v_safe = data.get("v_safe", None)

    print(f"U shape: {U.shape}")
    print(f"lam shape: {lam.shape}")
    print(f"Top eigenvalue: {lam[0].item():.4f}")

    if v_safe is not None:
        print(f"v_safe shape: {v_safe.shape}")
        
        # Check alignment with top component
        u0 = U[:, 0]
        
        dot = torch.dot(v_safe.flatten(), u0.flatten()).item()
        
        norm_v = torch.norm(v_safe).item()
        norm_u = torch.norm(u0).item()
        cosine = dot / (norm_v * norm_u + 1e-8)
        
        print(f"Dot product (v_safe . u0): {dot:.4f}")
        print(f"Cosine similarity: {cosine:.4f}")
        
        if cosine < 0:
            print("=> v_safe is OPPOSED to the Nudity direction (as expected).")
        else:
            print("=> v_safe is ALIGNED with the Nudity direction (unexpected, check pair definitions).")
            
        # Check projection magnitude
        # P_S(v_safe) magnitude vs |v_safe|
        # How much of "Safety" lies in the "Nudity Subspace"?
        # If they are totally orthogonal, this would be 0.
        # We expect some overlap (negative).
        
        proj = torch.matmul(v_safe, U) # (k,)
        proj_vec = torch.matmul(proj, U.T)
        proj_norm = torch.norm(proj_vec).item()
        
        ratio = proj_norm / norm_v
        print(f"Projection ratio (|P_S(v_safe)| / |v_safe|): {ratio:.4f}")
        
    else:
        print("v_safe not found in file.")

if __name__ == "__main__":
    verify()
