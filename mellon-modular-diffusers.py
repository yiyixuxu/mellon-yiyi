import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLModularPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.guider import PAGGuider
from diffusers.pipelines.modular_pipeline_builder import SequentialPipelineBlocks
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_modular import (
    StableDiffusionXLAutoDenoiseStep,
    StableDiffusionXLAutoPrepareAdditionalConditioningStep,
    StableDiffusionXLAutoPrepareLatentsStep,
    StableDiffusionXLAutoSetTimestepsStep,
    StableDiffusionXLDecodeLatentsStep,
    StableDiffusionXLInputStep,
    StableDiffusionXLTextEncoderStep,
    StableDiffusionXLVAEEncoderStep,
)
from image_gen_aux import DepthPreprocessor

from mellon.NodeBase import NodeBase


class ModelLoader(NodeBase):
    def execute(self, model_id, variant, dtype):
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, variant=variant, torch_dtype=dtype
        )
        return {
            "unet": pipeline.unet,
            "text_encoders": {
                "unet": pipeline.unet,
                "text_encoder": pipeline.text_encoder,
                "text_encoder_2": pipeline.text_encoder_2,
                "tokenizer": pipeline.tokenizer,
                "tokenizer_2": pipeline.tokenizer_2,
            },
            "vae": pipeline.vae,
            "scheduler": pipeline.scheduler,
        }


class EncodePrompts(NodeBase):
    def execute(self, text_encoders, device, positive_prompt, negative_prompt):
        text_encoder_workflow = StableDiffusionXLTextEncoderStep()
        text_encoder_workflow.update_states(
            text_encoder=text_encoders["text_encoder"],
            tokenizer=text_encoders["tokenizer"],
            text_encoder_2=text_encoders["text_encoder_2"],
            tokenizer_2=text_encoders["tokenizer_2"],
        )

        text_node = StableDiffusionXLModularPipeline()
        text_node.add_blocks(text_encoder_workflow)

        text_node.to(device)

        text_state = text_node.run_blocks(
            prompt=positive_prompt, negative_prompt=negative_prompt
        )
        text_node.to("cpu")

        return {"embeddings": text_state}


class EncodeImage(NodeBase):
    def execute(self, vae, image, device):
        vae_encoder_workflow = StableDiffusionXLVAEEncoderStep()
        vae_encoder_workflow.update_states(vae=vae)
        image_node = StableDiffusionXLModularPipeline()
        image_node.add_blocks(vae_encoder_workflow)
        image_node.to(device)

        image_state = image_node.run_blocks(image=image, batch_size=1)
        latents = image_state.get_intermediate("image_latents")

        return {"latents": latents}


class DenoiseLoop(NodeBase):
    def execute(
        self,
        unet,
        scheduler,
        device,
        embeddings,
        steps,
        cfg,
        seed,
        height,
        width,
        strength,
        image_latents,
        controlnet,
        guider,
    ):
        class StableDiffusionXLMainSteps(SequentialPipelineBlocks):
            block_classes = [
                StableDiffusionXLInputStep,
                StableDiffusionXLAutoSetTimestepsStep,
                StableDiffusionXLAutoPrepareLatentsStep,
                StableDiffusionXLAutoPrepareAdditionalConditioningStep,
                StableDiffusionXLAutoDenoiseStep,
            ]
            block_prefixes = [
                "input",
                "set_timesteps",
                "prepare_latents",
                "prepare_add_cond",
                "denoise",
            ]

        sdxl_workflow = StableDiffusionXLMainSteps()

        generator = torch.Generator(device="cuda").manual_seed(seed)

        modules_kwargs = {
            "unet": unet,
            "scheduler": scheduler,
        }

        denoise_kwargs = {
            **embeddings.intermediates,
            "generator": generator,
            "guidance_scale": cfg,
            "height": height,
            "width": width,
        }

        if controlnet is not None:
            # TODO: see how to do multicontrolnet
            modules_kwargs["controlnet"] = controlnet["controlnet_model"]
            denoise_kwargs["control_image"] = controlnet["image"]
            denoise_kwargs["controlnet_conditioning_scale"] = controlnet[
                "conditioning_scale"
            ]
            denoise_kwargs["control_guidance_start"] = controlnet["guidance_start"]
            denoise_kwargs["control_guidance_end"] = controlnet["guidance_end"]

        if guider is not None:
            modules_kwargs["guider"] = guider["guider"]
            denoise_kwargs["guider_kwargs"] = {"pag_scale": guider["scale"]}

            if controlnet is not None:
                modules_kwargs["controlnet_guider"] = guider["guider"]

        sdxl_workflow.update_states(**modules_kwargs)
        sdxl_node = StableDiffusionXLModularPipeline()
        sdxl_node.add_blocks(sdxl_workflow)

        sdxl_node.to(device)

        if image_latents is not None:
            denoise_kwargs["image"] = image_latents
            denoise_kwargs["strength"] = strength
            denoise_kwargs["num_inference_steps"] = round(steps / strength)
        else:
            denoise_kwargs["num_inference_steps"] = steps

        state_text2img = sdxl_node.run_blocks(**denoise_kwargs)

        latents = state_text2img.get_intermediate("latents")

        return {"latents": latents}


class DecodeLatents(NodeBase):
    def execute(self, vae, latents, device):
        decoder_workflow = StableDiffusionXLDecodeLatentsStep()
        decoder_workflow.update_states(vae=vae)

        decoder_node = StableDiffusionXLModularPipeline()
        decoder_node.add_blocks(decoder_workflow)

        decoder_node.to(device)
        image_text2img = decoder_node.run_blocks(latents=latents)

        images = image_text2img.get_output("images").images[0]

        return {"images": images}


class LoadControlnetModel(NodeBase):
    def execute(self, model_id, variant, dtype):
        controlnet_model = ControlNetModel.from_pretrained(
            model_id, variant=variant, torch_dtype=dtype
        )

        return {"controlnet_model": controlnet_model}


class Controlnet(NodeBase):
    def execute(
        self, image, conditioning_scale, controlnet_model, guidance_start, guidance_end
    ):
        controlnet = {
            "image": image,  # Maybe do some checks, e.g. it's a one or three channels image
            "conditioning_scale": conditioning_scale,
            "controlnet_model": controlnet_model,
            "guidance_start": guidance_start,
            "guidance_end": guidance_end,
        }
        return {"controlnet": controlnet}


class PAGOptionalGuider(NodeBase):
    def execute(self, pag_scale, pag_layers):
        # TODO: Maybe do some validations to ensure correct layers
        layers = [item.strip() for item in pag_layers.split(",")]
        guider = PAGGuider(pag_applied_layers=layers)

        return {"guider": {"guider": guider, "scale": pag_scale}}
