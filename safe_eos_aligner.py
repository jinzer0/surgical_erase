import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeEOSAligner:
    """
    Implements Prompt-adaptive Safe-EOS Anchor Alignment.
    Modifies text embeddings to suppress nudity/sexual concepts during diffusion.
    """
    def __init__(self, 
                 U, 
                 lam=None, 
                 v_safe=None, 
                 tau=0.15, 
                 T=0.1, 
                 alpha_max=0.8, 
                 top_m=5, 
                 eta=0.05, 
                 device="cuda",
                 dtype=torch.float16,
                 temporal_mode="instant",
                 schedule_mode="constant",
                 align_mode="steer", # eradicate or steer
                 steering_scale=2.0, # Scale factor for steering vector
                 start_step=0, # Delayed steering
                 end_step=1000, # Range-Bounded steering (Early Stop)
                 beta=0.5):
        """
        Args:
            U: Nudity subspace eigenvectors (D, k)
            lam: Eigenvalues (k,)
            v_safe: Safety direction vector (D,) - Difference between Safe and Neutral
            tau: Activation threshold
            T: Temperature for sigmoid gating
            alpha_max: Maximum intervention strength
            top_m: Number of tokens to modify
            eta: Trust region constraint radius factor
            temporal_mode: 'instant', 'momentum', 'fixed'
            schedule_mode: 'constant', 'increasing', 'decreasing', 'bell'
            align_mode: 'eradicate' (zero out), 'steer' (align to safe direction), 'combined' (erase + inject)
            start_step: Step to start intervention (delayed steering)
            end_step: Step to end intervention (range-bounded steering)
        """
        self.device = device
        self.dtype = dtype
        self.U = U.to(device=device, dtype=torch.float32) # Operations in fp32 for stability
        self.lam = lam.to(device=device, dtype=torch.float32) if lam is not None else None
        self.v_safe = v_safe.to(device=device, dtype=torch.float32) if v_safe is not None else None

        if align_mode == "steer" and self.v_safe is None:
            print("[SafeEOSAligner] Warning: align_mode='steer' but v_safe is None. Fallback to 'eradicate'.")
            self.align_mode = "eradicate"
        else:
            self.align_mode = align_mode
        
        self.tau = tau
        self.T = T
        self.alpha_max = alpha_max
        self.top_m = top_m
        self.eta = eta
        
        self.temporal_mode = temporal_mode
        self.schedule_mode = schedule_mode
        self.steering_scale = steering_scale
        self.start_step = start_step
        self.end_step = end_step
        self.beta = beta
        
        # State for temporal consistency
        self.s_smooth = None
        self.fixed_mask_indices = None
        self.cached_delta_par = None
        
        # Logging
        self.stats = []

    def reset_state(self):
        self.s_smooth = None
        self.fixed_mask_indices = None
        self.cached_delta_par = None
        self.stats = []

    def project(self, x):
        """
        Project x onto subspace U.
        P_S(x) = x @ U @ U.T
        x: (..., D)
        """
        # x is (B, L, D) or (B, D)
        # U is (D, k)
        # x @ U -> (..., k)
        coeff = torch.matmul(x, self.U)
        # (..., k) @ U.T -> (..., D)
        proj = torch.matmul(coeff, self.U.T)
        return proj


    def get_score(self, x):
        """
        Compute nudity activation score s_i (0 ~ 1 scaling).
        """
        # x: (B, L, D)
        # Unweighted: ||P_S(e_i)||_2
        # Weighted: sqrt(sum lam_j * coeff_j^2)
        
        coeff = torch.matmul(x, self.U) # (B, L, k)
        
        if self.lam is not None:
             # Weighted L2 norm of coefficients
             # s_i^2 = sum(lam[j] * coeff[j]^2)
             weighted_sq = (coeff ** 2) * self.lam.view(1, 1, -1)
             s_raw = torch.sqrt(torch.sum(weighted_sq, dim=-1)) # (B, L)
        else:
            # Simple L2 norm of projected vector
            # ||P_S(x)|| = ||coeff @ U.T||. Since U is orthonormal, ||coeff @ U.T|| = ||coeff||
            s_raw = torch.norm(coeff, dim=-1) # (B, L)
            
        # Normalize score to [0, 1] range relative to original embedding magnitude
        # s = ||P_S(x)|| / (||x|| + eps)
        # This makes 'tau' (0~1) meaningful.
        x_norm = torch.norm(x, dim=-1) + 1e-8
        s = s_raw / x_norm
            
        return s

    def get_schedule_factor(self, step, num_steps):
        t_norm = step / max(1, num_steps - 1)
        import math
        if self.schedule_mode == "constant":
            return 1.0
        elif self.schedule_mode == "increasing":
            return (1 - math.cos(math.pi * t_norm)) / 2
        elif self.schedule_mode == "decreasing":
            return (1 + math.cos(math.pi * t_norm)) / 2
        elif self.schedule_mode == "bell":
            return math.sin(math.pi * t_norm)
        return 1.0

    def edit_embeddings(self, 
                        E, 
                        step, 
                        num_steps, 
                        eos_idx=None):
        """
        Main editing function.
        E: (B, L, D) - Text embeddings (batch size is typically 2*prompt_batch for cfg, or just prompt_batch)
        """
        # 0. Check Start/End Step (Range-Bounded Intervention)
        if step < self.start_step or step > self.end_step:
            return E

        # Ensure fp32 for calculations
        orig_dtype = E.dtype
        E_fp32 = E.to(torch.float32)
        B, L, D = E_fp32.shape
        
        # 1. Compute Safe Anchor (using EOS)
        # Only compute anchor from conditional branch usually (first half if batched cond/uncond together)
        # But here we assume E might contain uncond. 
        # Plan says: "cond만 수정" is default.
        # So we assume E comes in as [uncond, cond] (standard diffusers).
        # We only want to edit the second half (cond/prompt).
        
        # Let's target only the second half.
        chunk_size = B // 2
        
        cond_indices = slice(chunk_size, B) if B > 1 else slice(0, B)
        E_cond = E_fp32[cond_indices]
        
        # Identify EOS
        if eos_idx is None:
             # Fallback
             idx = L - 1
             e_eos = E_cond[:, idx, :] 
        else:
             # If eos_idx passed
             idx = L - 1
             e_eos = E_cond[:, idx, :]

        # 2. Safe Anchor
        # a = e_eos - P_S(e_eos) (Not used for steering, just concept)
        # e_par_eos = self.project(e_eos)
        # safe_anchor = e_eos - e_par_eos 
        
        # 3. Nudity Score
        s = self.get_score(E_cond) # (B_cond, L)
        
        # Temporal consistency
        if self.temporal_mode == "momentum":
            if self.s_smooth is None:
                self.s_smooth = s
            else:
                self.s_smooth = self.beta * self.s_smooth + (1 - self.beta) * s
            score_to_use = self.s_smooth
        else:
            score_to_use = s
            
        # 4. Gating & Masking
        # alpha = sigmoid((s - tau)/T)
        alpha = torch.sigmoid((score_to_use - self.tau) / self.T)
        alpha = torch.clamp(alpha, 0, self.alpha_max)
        
        # Apply schedule
        g = self.get_schedule_factor(step, num_steps)
        alpha = alpha * g
        
        # Top-m selection
        # Identify indices to mask
        # We want top-m scores per sample.
        vals, indices = torch.topk(score_to_use, k=min(self.top_m, L), dim=-1)
        
        # Create a mask
        mask = torch.zeros_like(alpha)
        # Scatter 1s to top-k indices
        mask.scatter_(-1, indices, 1.0)
        
        # Combine with threshold filtering?
        threshold_mask = (score_to_use > self.tau).float()
        final_mask = mask * threshold_mask
        
        if self.temporal_mode == "fixed":
            if step == 0 or self.fixed_mask_indices is None:
                 self.fixed_mask_indices = final_mask
            final_mask = self.fixed_mask_indices
            
        alpha_masked = alpha * final_mask
        
        # 5. Alignment Update
        # e_par = P_S(e_i)
        e_par = self.project(E_cond) # (B, L, D)
        e_perp = E_cond - e_par

        # Define alpha_exp for use in all branches
        alpha_exp = alpha_masked.unsqueeze(-1)
        
        if self.align_mode == "steer":
            # Steering Logic:
            # We want to replace the current projection e_par with the projection of v_safe.
            # target = P_S(v_safe)
            # This 'target' represents the "Safe Coordinates" on the Nudity Subspace.
            target_par = self.project(self.v_safe) # (D,)
            
            # Apply scaling to boost weak safety signal
            target_par = target_par * self.steering_scale
            
            # Broadcast to (B, L, D)
            target_par = target_par.view(1, 1, -1).expand_as(e_par)
            
            # Standard Update:
            e_par_new = (1 - alpha_exp) * e_par + alpha_exp * target_par
            E_cond_new = e_perp + e_par_new
            
        elif self.align_mode == "combined":
             # Combined Logic (Novel):
             # 1. Erase: "Clamp" magnitude of projection to tau instead of zeroing.
             #    This preserves structure (direction) while limiting intensity.
             #    v_par = P_U(x)
             #    if ||v_par|| > tau: v_par_new = v_par * (tau / ||v_par||)
             
             # Calculate magnitude of current projection e_par
             e_par_norm = torch.norm(e_par, dim=-1, keepdim=True) + 1e-8
             
             # Target magnitude is clamped to self.tau (or a separate limit?)
             # Using self.tau as the "Safe Limit" consistent with activation threshold.
             # We want to reduce it towards this limit.
             
             # Soft Clamp:
             # scale_factor = min(1.0, self.tau / e_par_norm)
             # But we apply this via alpha blending for smooth intervention ?
             # No, "Combined" is usually direct. Let's use alpha to blend between Original and Clamped.
             
             target_norm = torch.minimum(e_par_norm, torch.tensor(self.tau, device=self.device))
             target_par = e_par * (target_norm / e_par_norm)
             
             # e_par_new = (1 - alpha) * e_par + alpha * target_par
             # This effectively scales down the projection towards the clamp limit.
             e_par_new = (1 - alpha_exp) * e_par + alpha_exp * target_par
             
             
             # 2. Inject: Orthogonal Component
             # We want to add "Clothes" (v_safe) but NOT disturb the Nudity Subspace U.
             # Since U contains "Personhood", and v_safe might have small overlap.
             # Ideally v_safe is mostly orthogonal.
             # Let's explicitly inject ONLY the orthogonal part to be safe.
             # v_safe_ortho = v_safe - P_U(v_safe)
             
             v_safe_proj = self.project(self.v_safe)
             v_safe_ortho = self.v_safe - v_safe_proj
             
             # Inject this orthogonal vector
             v_safe_inject = v_safe_ortho.view(1, 1, -1).expand_as(E_cond)
             d_inject = alpha_exp * self.steering_scale * v_safe_inject
             
             # Final = Perp + Par_New + Inject
             # Note: Perp is E_cond - e_par (Orthogonal to U).
             # We assume d_inject is also largely orthogonal to U (by construction).
             E_cond_new = e_perp + e_par_new + d_inject

        elif self.align_mode == "eos_delta":
            # EOS Anchor Steering Logic (Subspace-Preserving):
            # User concept: Safe_EOS_Anchor = e_eos - P_U(e_eos) + v_safe
            # Delta_EOS = Safe_EOS_Anchor - e_eos = v_safe - P_U(e_eos)
            
            # Cache the original anchor displacement at the beginning so it points 
            # steadily towards the fixed target across all denoising steps.
            if self.cached_delta_par is None:
                e_eos_nudity = self.project(e_eos) # (B, D)
                v_safe_scaled = self.v_safe * self.steering_scale
                
                # The 'Correction Vector' derived from the original EOS context
                delta_eos = v_safe_scaled - e_eos_nudity # (B, D)
                delta_eos_expanded = delta_eos.unsqueeze(1) # (B, 1, D)
                
                # CRITICAL STABILITY FIX:
                # Apply the EOS-Delta ONLY to the Subspace projection (e_par).
                self.cached_delta_par = self.project(delta_eos_expanded) # (B, 1, D)
            
            delta_par = self.cached_delta_par
            e_par_new_raw = e_par + (alpha_exp * delta_par)
            
            # OPTION (B): Max Norm Clamp (to prevent noise explosion!)
            # Even if steering_scale is massive (e.g., 30.0), this ensures the new embedding
            # never exceeds a safe boundary. We clamp the magnitude of the new projection 
            # to self.tau (the safe threshold limit for nudity).
            new_norm = torch.norm(e_par_new_raw, dim=-1, keepdim=True) + 1e-8
            
            # The maximum allowed norm on the Subspace U is self.tau.
            max_allowed_norm = torch.tensor(self.tau, device=self.device)
            scale_factor = torch.minimum(torch.tensor(1.0, device=self.device), max_allowed_norm / new_norm)
            
            e_par_new = e_par_new_raw * scale_factor
            
            E_cond_new = e_perp + e_par_new

        else: # eradicate
            target_par = torch.zeros_like(e_par)
            
            # Standard Update:
            e_par_new = (1 - alpha_exp) * e_par + alpha_exp * target_par
            E_cond_new = e_perp + e_par_new
        
        # 6. Trust Region
        # d = new - old
        d = E_cond_new - E_cond
        d_norm = torch.norm(d, dim=-1, keepdim=True)
        e_norm = torch.norm(E_cond, dim=-1, keepdim=True)
        
        eps = self.eta * e_norm
        
        # r = min(1, eps / (||d|| + 1e-8))
        r = torch.minimum(torch.ones_like(d_norm), eps / (d_norm + 1e-8))
        
        E_cond_final = E_cond + r * d
        
        # Save stats
        with torch.no_grad():
            # Compute score AFTER modification to see effect
            s_post = self.get_score(E_cond_final)
            
            # Compute Signed Projection on 1st Component (Assuming U[:, 0] is Nudity Axis)
            # coeff = x @ U -> (B, L, k)
            # We want coeff[..., 0]
            coeff = torch.matmul(E_cond, self.U)
            pc1 = coeff[..., 0] # (B, L)
            
            scaling_applied = (r < 1.0).float().mean().item() * 100
            mean_score = score_to_use.mean().item()
            max_score = score_to_use.max().item()
            
            self.stats.append({
                "step": step,
                "mean_score": mean_score,
                "max_score": max_score,
                "scaling_applied_pct": scaling_applied,
                "active_tokens": final_mask.sum().item(),
                "token_scores": score_to_use.detach().cpu(), # Pre-edit Score
                "post_scores": s_post.detach().cpu(),        # Post-edit Score
                "pc1_values": pc1.detach().cpu()             # Signed PC1 (Pre-edit)
            })
            
        # Update original batch
        E_out = E_fp32.clone()
        E_out[cond_indices] = E_cond_final
        
        return E_out.to(dtype=orig_dtype)

    def get_aggregated_scores(self):
        """
        Returns the mean score across all stored steps for visualization.
        Returns: (B, L) tensor
        """
        if not self.stats:
            return None
        
        # Stack all token_scores: (Steps, B, L)
        all_scores = torch.stack([s["token_scores"] for s in self.stats], dim=0)
        # Mean across steps
        mean_scores = all_scores.mean(dim=0)
        return mean_scores
