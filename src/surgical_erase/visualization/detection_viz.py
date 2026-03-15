import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def save_attention_map(
    scores, 
    tokens, 
    output_path,
    title=None
):
    """
    Generate and save a heatmap of detection scores over tokens.
    
    Args:
        scores: (L,) tensor of scores for each token.
        tokens: List of token strings corresponding to scores.
        output_path: Path to save the image.
        title: Optional title for the plot.
    """
    # Ensure scores are cpu numpy
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
        
    # Handle [B, L] if passed, assume batch size 1 or take mean? 
    # Usually we pass single sequence scores (L,). 
    # If (1, L), squeeze.
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores.squeeze(0)
        
    n_tokens = len(tokens)
    if len(scores) != n_tokens:
        # Mismatch handling: truncate or pad?
        # Usually CLIP tokenizer pads. Scores might include padding.
        # We should only visualize up to active tokens if possible, or just all.
        minimum = min(len(scores), n_tokens)
        scores = scores[:minimum]
        tokens = tokens[:minimum]

    # Dynamic figsize: ~0.4 inches per token, min 15 inches.
    # Height slightly increased for label space.
    fig_width = max(15, n_tokens * 0.4)
    fig_height = 4 

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create a heatmap
    # Reshape scores to (1, L) for imshow
    # vmin=0, vmax=3.0 to show absolute scale in [0, 3.0] range (Weighted scores can go up to ~2.6).
    im = ax.imshow(scores[np.newaxis, :], cmap='Reds', aspect='auto', vmin=0, vmax=3.0)
    
    # Add labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks([])
    # Rotate 45 -> 60, larger font
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
        
    # Add colorbar (vertical on the right is cleaner for long strips)
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight') # bbox_inches='tight' helps with cropped labels
    plt.close()

def save_step_analysis_graph(stats, tokens, output_path):
    """
    Generate and save a heatmap of detection scores over steps (time).
    X-axis: Tokens
    Y-axis: Diffusion Steps (0 to T)
    Plots both Pre-Steering and Post-Steering scores side-by-side or difference.
    Actually, let's plot Pre-Steering (Input) as usual, but maybe add a line plot for the max score?
    
    Let's stick to the Heatmap, but use Post-Steering if available? 
    The user wants to see the effect. 
    If Pre-Steering (Input to Step i) is high, and Post-Steering (Output of Step i) is low, 
    then Input to Step i+1 should be low.
    
    If Input i is high, Output i is low, but Input i+1 is high again... that's weird.
    
    Let's plot TWO heatmaps side-by-side: "Before Steering" and "After Steering".
    """
    if not stats:
        return
        
    step_scores_pre = []
    step_scores_post = []
    steps = []
    
    has_post = "post_scores" in stats[0]
    
    for s in stats:
        steps.append(s["step"])
        
        # Pre
        scr = s["token_scores"]
        if scr.ndim == 2: scr = scr[0]
        step_scores_pre.append(scr.numpy())
        
        # Post
        if has_post:
            scr_p = s["post_scores"]
            if scr_p.ndim == 2: scr_p = scr_p[0]
            step_scores_post.append(scr_p.numpy())
            
    step_scores_pre = np.stack(step_scores_pre) # (Steps, L)
    if has_post:
        step_scores_post = np.stack(step_scores_post)
    
    # Check token length match
    n_tokens = len(tokens)
    if step_scores_pre.shape[1] != n_tokens:
        minimum = min(step_scores_pre.shape[1], n_tokens)
        step_scores_pre = step_scores_pre[:, :minimum]
        tokens = tokens[:minimum]
        if has_post:
            step_scores_post = step_scores_post[:, :minimum]
            
    # Plot
    if has_post:
        fig, axes = plt.subplots(1, 2, figsize=(max(20, n_tokens * 0.8), max(8, len(steps) * 0.2)))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(figsize=(max(12, n_tokens * 0.4), max(8, len(steps) * 0.2)))
        ax2 = None
        
    # 1. Pre-Steering
    im1 = ax1.imshow(step_scores_pre, cmap='Reds', aspect='auto', vmin=0, vmax=3.0)
    ax1.set_title("Nudity Score (Pre-Steering)", fontsize=14)
    ax1.set_xticks(np.arange(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=60, ha="right", fontsize=10)
    ax1.set_yticks(np.arange(len(steps)))
    ax1.set_yticklabels(steps, fontsize=8)
    ax1.set_ylabel("Step")
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Post-Steering
    if ax2 is not None:
        im2 = ax2.imshow(step_scores_post, cmap='Reds', aspect='auto', vmin=0, vmax=3.0)
        ax2.set_title("Nudity Score (Post-Steering)", fontsize=14)
        ax2.set_xticks(np.arange(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=60, ha="right", fontsize=10)
        ax2.set_yticks([]) # Hide Y ticks
        
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_step_analysis_graph(stats, tokens, output_path):
    """
    Generate and save a heatmap of detection scores over steps (time).
    X-axis: Tokens
    Y-axis: Diffusion Steps (0 to T)
    """
    if not stats:
        return
        
    # Stack scores: (Steps, L)
    # stats[i]["token_scores"] is a tensor (L,) or (B, L)
    # We assume batch size 1 for analysis or take the first sample
    step_scores = []
    steps = []
    
    for s in stats:
        steps.append(s["step"])
        scr = s["token_scores"]
        if scr.ndim == 2:
            scr = scr[0] # Take first sample
        step_scores.append(scr.numpy())
        
    step_scores = np.stack(step_scores) # (Steps, L)
    
    # Check token length match
    n_tokens = len(tokens)
    if step_scores.shape[1] != n_tokens:
        minimum = min(step_scores.shape[1], n_tokens)
        step_scores = step_scores[:, :minimum]
        tokens = tokens[:minimum]
        
    # Plot Heatmap
    # X: Tokens
    # Y: Steps
    
    fig_width = max(12, n_tokens * 0.4)
    fig_height = max(8, len(steps) * 0.2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # vmin=0, vmax=3.0 (Consistent with detection map)
    im = ax.imshow(step_scores, cmap='Reds', aspect='auto', vmin=0, vmax=3.0)
    
    # X-axis Labels (Tokens)
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=10)
    ax.set_xlabel("Tokens")
    
    # Y-axis Labels (Steps)
    # If too many steps, show every Nth
    ax.set_yticks(np.arange(len(steps)))
    ax.set_yticklabels(steps, fontsize=10)
    ax.set_ylabel("Inference Step")
    
    ax.set_title("Nudity Score Evolution per Step", fontsize=14)
    
    plt.colorbar(im, ax=ax, label="Nudity Score")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_token_trajectory_graph(stats, tokens, output_path, top_k=5):
    """
    Generate and save a line graph of detection scores over steps (time).
    X-axis: Diffusion Steps (0 to T)
    Y-axis: Nudity Score
    Lines: Specific tokens (filtering low scoring ones to avoid clutter)
    
    Args:
        top_k: Number of top scoring tokens to visualize
    """
    if not stats:
        return
        
    step_scores = []
    steps = []
    
    for s in stats:
        steps.append(s["step"])
        scr = s["token_scores"]
        if scr.ndim == 2: scr = scr[0]
        step_scores.append(scr.numpy())
        
    step_scores = np.stack(step_scores) # (Steps, L)
    
    # Check token length match
    n_tokens = len(tokens)
    if step_scores.shape[1] != n_tokens:
        minimum = min(step_scores.shape[1], n_tokens)
        step_scores = step_scores[:, :minimum]
        tokens = tokens[:minimum]
        
    # Identify important tokens to plot
    # Logic: Max score across all steps
    max_scores = step_scores.max(axis=0) # (L,)
    
    # Get top k indices
    if top_k is None or top_k >= n_tokens:
        top_indices = np.arange(n_tokens)
    else:
        top_indices = np.argsort(max_scores)[::-1][:top_k]
        
    # Filter tokens and scores
    target_tokens = [tokens[i] for i in top_indices]
    target_scores = step_scores[:, top_indices] # (Steps, K)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*']
    
    for i in range(len(target_tokens)):
        token_str = target_tokens[i]
        # Clean up token str
        token_str = token_str.replace("</w>", "")
        
        ax.plot(steps, target_scores[:, i], 
                label=token_str, 
                marker=markers[i % len(markers)], 
                markersize=4,
                alpha=0.8)
        
    ax.set_xlabel("Inference Step")
    ax.set_ylabel("Nudity Score")
    ax.set_title(f"Token Score Trajectory (Top {len(target_tokens)})")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_pc1_trajectory_graph(stats, tokens, output_path, top_k=5):
    """
    Generate and save a line graph of Squared PC1 Values over steps.
    Basically same as token trajectory but for PC1.
    """
    if not stats:
        return
        
    step_vals = []
    steps = []
    
    # Check if pc1_values exists
    if "pc1_values" not in stats[0]:
        return

    for s in stats:
        steps.append(s["step"])
        val = s["pc1_values"]
        if val.ndim == 2: val = val[0]
        step_vals.append(val.numpy())
        
    step_vals = np.stack(step_vals) # (Steps, L)
    
    n_tokens = len(tokens)
    if step_vals.shape[1] != n_tokens:
        minimum = min(step_vals.shape[1], n_tokens)
        step_vals = step_vals[:, :minimum]
        tokens = tokens[:minimum]
        
    # Filter high magnitude PC1 tokens
    # We care about tokens that have high POSITIVE PC1 (Nudity)
    # top_indices based on max(PC1)
    max_vals = step_vals.max(axis=0)
    
    if top_k is None or top_k >= n_tokens:
        top_indices = np.arange(n_tokens)
    else:
        # Get top k by max positive value
        top_indices = np.argsort(max_vals)[::-1][:top_k]
        
    target_tokens = [tokens[i] for i in top_indices]
    target_vals = step_vals[:, top_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*']
    
    for i in range(len(target_tokens)):
        token_str = target_tokens[i].replace("</w>", "")
        ax.plot(steps, target_vals[:, i], 
                label=token_str, 
                marker=markers[i % len(markers)], 
                markersize=4,
                alpha=0.8)
        
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Inference Step")
    ax.set_ylabel("PC1 Value (Signed Nudity)")
    ax.set_title(f"PC1 Trajectory (Top {len(target_tokens)} Nude Tokens)")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
