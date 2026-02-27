from diffusers import StableDiffusionPipeline
import torch

from safe_eos_aligner import SafeEOSAligner

class SADiffusersPipeline(StableDiffusionPipeline):
    """
    Stable Diffusion Pipeline with Prompt-adaptive Safe-EOS Anchor Alignment.
    """
    def set_aligner(self, aligner):
        self.aligner = aligner
    
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        **kwargs,
    ):
        # 0. Default height/width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, None, None)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1 # prompt_embeds used?

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if hasattr(self, "encode_prompt"):
             prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                # prompt_embeds=prompt_embeds, # Removed to support older diffusers if needed, but standard has it
                # negative_prompt_embeds=negative_prompt_embeds,
            )
        else:
            # Fallback for older diffusers versions if _encode_prompt signature varies
            # But let's assume standard SD pipeline structure.
            # If _encode_prompt returns tuple or single tensor?
            # Recent diffusers returns (prompt_embeds, negative_prompt_embeds) if requested or concat.
            # Let's rely on standard behavior: concat [neg, pos]
            prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt
            ) 
            # In some versions it returns a tuple, in others a tensor.
            # If it's a tuple, take the first element.
            if isinstance(prompt_embeds, tuple) or isinstance(prompt_embeds, list):
                prompt_embeds = prompt_embeds[0] 
            # Note: encode_prompt in v0.10+ returns tuple if we ask? 
            # Actually standard pipeline `encode_prompt` returns a single concatenated tensor `text_embeddings` 
            # if we look at older implementations. Newer ones return `prompt_embeds`.
            # Let's try to check `text_encoder` logic.
            # To be safe and compatible with "StableDiffusionPipeline", we mimic its `__call__` flow.
            # We assume it returns `prompt_embeds` which is [uncond, cond].

        # Reset aligner state
        if hasattr(self, "aligner"):
            self.aligner.reset_state()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        # EDIT: Pre-cache token_ids if possible? No, we don't have them here easily unless we encoded them.
        # But we can assume standard behavior.

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # --- ALIGNMENT INTERVENTION ---
                # We modify `prompt_embeds` (encoder_hidden_states) dynamically per step?
                # Or we modify it once? 
                # Requirement: "denoising step마다 수정" (Modify at every denoising step).
                # So we pass modified embeddings to UNet.
                
                current_prompt_embeds = prompt_embeds
                
                if hasattr(self, "aligner"):
                    # Apply editing on fresh original embeddings each step!
                    # Do NOT accumulate, as that explodes additive methods ('eos_delta', 'combined') into noise.
                    current_prompt_embeds = self.aligner.edit_embeddings(
                        prompt_embeds, 
                        step=i, 
                        num_steps=num_inference_steps
                    )
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)

        if not return_dict:
            return (image, False)

        from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

