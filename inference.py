import argparse
import torch
import os
import pandas as pd
import numpy as np
import random
from diffusers import StableDiffusionPipeline
from pipeline_sa_diffusion import SADiffusersPipeline
from safe_eos_aligner import SafeEOSAligner
from build_subspace import SubspaceBuilder
from notify import get_notified
from visualize_detection import save_attention_map, save_step_analysis_graph, save_token_trajectory_graph, save_pc1_trajectory_graph

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Safe-EOS Anchor Alignment")
    
    # Input/Output
    parser.add_argument("--prompts", type=str, nargs="+", help="List of prompts")
    parser.add_argument("--csvfile", type=str, help="Path to CSV file with prompts")
    parser.add_argument("--num_prompts", type=int, default=None, help="Number of prompts to process from CSV")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    
    # Model
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    
    # Subspace
    parser.add_argument("--modifiers_json", type=str, default="data/modifiers_v2.json")
    parser.add_argument("--num_pairs", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=50.0) # Ridge default 50 based on typically usage
    parser.add_argument("--subspace_path", type=str, help="Path to pre-computed subspace (optional)")
    
    # Aligner
    parser.add_argument("--tau", type=float, default=0.18)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--top_m", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--temporal_mode", type=str, default="instant", choices=["instant", "momentum", "fixed"])
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum factor (beta) for temporal smoothing")
    parser.add_argument("--schedule_mode", type=str, default="constant", choices=["constant", "increasing", "decreasing", "bell"])
    parser.add_argument("--align_mode", type=str, default="steer", choices=["eradicate", "steer", "combined", "eos_delta"])
    parser.add_argument("--steering_scale", type=float, default=1.0, help="Scale factor for steering vector")
    parser.add_argument("--start_step", type=int, default=0, help="Step to start intervention")
    parser.add_argument("--end_step", type=int, default=50, help="Step to end intervention")
    
    # Diffusion
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", help="Visualize detection map")
    parser.add_argument("--analysis", action="store_true", help="Save step-wise analysis graph")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--show_progress", action="store_true", help="Show progress bar")
    
    return parser.parse_args()


@get_notified(task_name="EOS Alignment Inference")
def main():
    set_seed(42)
    args = parse_args()
    
    if not args.verbose:
        import logging
        import warnings
        from diffusers.utils import logging as dlogging
        from transformers.utils import logging as tlogging

        warnings.filterwarnings("ignore")

        dlogging.set_verbosity_error()
        tlogging.set_verbosity_error()

        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("diffusers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = args.device
    dtype = torch.float16 if args.fp16 else torch.float32
    
    # 1. Load or Build Subspace
    if args.subspace_path and os.path.exists(args.subspace_path):
        print(f"Loading subspace from {args.subspace_path}...")
        data = torch.load(args.subspace_path, map_location=device)
        U, lam = data["U"], data["lam"]
        v_safe = data.get("v_safe", None)
    else:
        print("Building subspace...")
        builder = SubspaceBuilder(device=device)
        pairs, safety_pairs = builder.generate_pairs_from_json(args.modifiers_json, args.num_pairs)
        U, lam, v_safe = builder.build(pairs, safety_pairs=safety_pairs, k=args.k, ridge=args.ridge)
    
    # 2. Initialize Aligner
    aligner = SafeEOSAligner(
        U=U,
        lam=lam,
        v_safe=v_safe,
        tau=args.tau,
        T=args.T,
        alpha_max=args.alpha_max,
        top_m=args.top_m,
        eta=args.eta,
        device=device,
        dtype=dtype,
        temporal_mode=args.temporal_mode,
        schedule_mode=args.schedule_mode,
        align_mode=args.align_mode,
        steering_scale=args.steering_scale,
        start_step=args.start_step,
        end_step=args.end_step,
        beta=args.momentum
    )
    
    # 3. Initialize Pipeline
    print(f"Loading Stable Diffusion from {args.model_id}...")
    pipeline = SADiffusersPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype
    ).to(device)
    
    if not args.show_progress:
        pipeline.set_progress_bar_config(disable=True)

    pipeline.set_aligner(aligner)
    
    # 4. Prepare Prompts
    prompts = []
    if args.csvfile:
        df = pd.read_csv(args.csvfile, keep_default_na=False)
        if "prompt" in df.columns:
            prompts = df["prompt"].tolist()
        else:
             # Fallback if no header or different name? 
             # The user provided csv has 'prompt' column.
             print("Warning: 'prompt' column not found, trying first column.")
             prompts = df.iloc[:, 1].tolist() # based on user file 0 is id, 1 is prompt

        if "evaluation_seed" in df.columns:
            args.seed = df["evaluation_seed"].tolist()
            if args.num_prompts:
                args.seed = args.seed[:args.num_prompts]


        if args.num_prompts:
            prompts = prompts[:args.num_prompts]
    
    if args.prompts:
        prompts.extend(args.prompts)
        
    if not prompts:
        raise ValueError("No prompts provided via --prompts or --csvfile")
        
    print(f"Processing {len(prompts)} prompts...")
    
    # 5. Inference Loop
    log_data = []
    
    for i, prompt in enumerate(prompts):
        if type(prompt) is not str:
            prompt = "Null"

        print(f"[{i+1}/{len(prompts)}] generating: {prompt[:20]}...")
        if type(args.seed) is list:
            generator = torch.Generator(device=device).manual_seed(args.seed[i])
        else:
            generator = torch.Generator(device=device).manual_seed(args.seed)
        
        # Run pipeline
        out = pipeline(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        )
        
        image = out.images[0]
        
        # Save image
        safe_prompt_name = "".join([c if c.isalnum() else "_" for c in prompt[:20]])
        img_path = os.path.join(args.output_dir, f"{i}_{safe_prompt_name}.png")
        
        # Defensive check: Ensure directory exists (in case of weird deletion or race)
        if not os.path.exists(args.output_dir):
            print(f"Warning: Output directory {args.output_dir} vanished! Re-creating.")
            os.makedirs(args.output_dir, exist_ok=True)
            
        try:
            image.save(img_path)
        except Exception as e:
            print(f"Error saving image to {img_path}: {e}")
            # Try to save to absolute path just in case
            abs_path = os.path.abspath(img_path)
            print(f"Absolute path: {abs_path}")
            raise e
        
        # Collect stats
        # The aligner stores stats in self.stats
        # Aggregate stats for this run
        run_stats = aligner.stats
        if run_stats:
            avg_score = sum(s["mean_score"] for s in run_stats) / len(run_stats)
            avg_scaling = sum(s["scaling_applied_pct"] for s in run_stats) / len(run_stats)
            avg_active = sum(s["active_tokens"] for s in run_stats) / len(run_stats)
        else:
            avg_score, avg_scaling, avg_active = 0, 0, 0
            
        print(f"  Stats: Score={avg_score:.4f}, Scaling%={avg_scaling:.2f}, ActiveTokens={avg_active:.2f}")
        
        log_data.append({
            "id": i,
            "prompt": prompt,
            "avg_score": avg_score,
            "avg_scaling_pct": avg_scaling,
            "avg_active_tokens": avg_active,
            "image_path": img_path
        })
        
        # Reset aligner stats (handled in pipeline but good to be sure or if logic changes)
        # Pipeline calls aligner.reset_state() at start of __call__
        
        if args.visualize:
            # 1. Get scores
            scores = aligner.get_aggregated_scores() # (B, L)
            if scores is not None:
                # 2. Get tokens
                # We need tokenizer to decode. Pipeline has tokenizer.
                tokenizer = pipeline.tokenizer
                # Re-tokenize prompt to get tokens match
                # Use same modulation as pipeline: 
                # prompt -> tokenizer -> tokens
                token_ids = tokenizer(
                    prompt, 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length, 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids[0]
                
                # Convert ids to tokens
                tokens = [tokenizer.decode([t]) for t in token_ids]
                
                # 3. Save map
                heatmap_path = os.path.join(args.output_dir, f"{i}_{safe_prompt_name}_heatmap.png")
                # Assuming batch size 1 for visualization loop
                save_attention_map(scores[0], tokens, heatmap_path, title=prompt[:50])
                
        if args.analysis and aligner.stats:
            # We reuse tokens from above if visualize was on, otherwise we need to tokenize
            if not args.visualize:
                tokenizer = pipeline.tokenizer
                token_ids = tokenizer(
                    prompt, 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length, 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids[0]
                tokens = [tokenizer.decode([t]) for t in token_ids]
            
            os.makedirs(os.path.join(args.output_dir, f"{safe_prompt_name[:10]}"), exist_ok=True)
            analysis_path = os.path.join(args.output_dir, f"{safe_prompt_name[:10]}/{i}_step_analysis.png")
            save_step_analysis_graph(aligner.stats, tokens, analysis_path)
            
            traj_path = os.path.join(args.output_dir, f"{safe_prompt_name[:10]}/{i}_token_trajectory.png")
            save_token_trajectory_graph(aligner.stats, tokens, traj_path, top_k=10)
            
            pc1_path = os.path.join(args.output_dir, f"{safe_prompt_name[:10]}/{i}_pc1_trajectory.png")
            save_pc1_trajectory_graph(aligner.stats, tokens, pc1_path, top_k=10)
            
            print(f"Saved step analysis to {analysis_path}, {traj_path}, {pc1_path}")
        
    # Save log
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(os.path.join(args.output_dir, "inference_log.csv"), index=False)
    print("Inference completed.")

if __name__ == "__main__":
    
    main()
