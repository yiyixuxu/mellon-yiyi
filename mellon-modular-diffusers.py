import os

import torch
from diffusers import ControlNetModel, ControlNetUnionModel, ModularPipeline
from diffusers.guider import APGGuider, PAGGuider
from diffusers.pipelines.components_manager import ComponentsManager
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_modular import (
    AUTO_BLOCKS,
    StableDiffusionXLIPAdapterStep,
    StableDiffusionXLLoraStep,
)
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mellon.NodeBase import NodeBase

all_blocks_map = AUTO_BLOCKS.copy()
text_block = all_blocks_map.pop("text_encoder")()
decoder_block = all_blocks_map.pop("decode")()
image_encoder_block = all_blocks_map.pop("image_encoder")()


class SDXLAutoBlocks(SequentialPipelineBlocks):
    block_classes = list(all_blocks_map.values())
    block_names = list(all_blocks_map.keys())


class ComponentsLoader(NodeBase):
    def execute(self, repo_id, variant, device, dtype):
        components = ComponentsManager()
        components.add_from_pretrained(repo_id, torch_dtype=dtype, variant=variant)
        components.enable_auto_cpu_offload(device=device)

        return {"components": components}


class TextEncoder(NodeBase):
    def execute(self, components, positive_prompt, negative_prompt):
        text_encoder = ModularPipeline.from_block(text_block)
        text_encoder.update_states(
            **components.get(
                ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
            )
        )

        text_state = text_encoder(
            prompt=positive_prompt, negative_prompt=negative_prompt
        )

        return {"embeddings": text_state}


class Denoise(NodeBase):
    def execute(
        self,
        components,
        embeddings,
        steps,
        cfg,
        seed,
        width,
        height,
        scheduler,
        karras,
        trailing,
        v_prediction,
        image,
        strength,
        guider,
        lora,
        controlnet,
        ip_adapter,
    ):
        class SDXLAutoBlocks(SequentialPipelineBlocks):
            block_classes = list(all_blocks_map.values())
            block_names = list(all_blocks_map.keys())

        sdxl_auto_blocks = SDXLAutoBlocks()

        denoise = ModularPipeline.from_block(sdxl_auto_blocks)
        denoise.update_states(**components.get(["unet", "scheduler", "vae"]))

        # since we need to check always if there are loras loaded
        lora_step = StableDiffusionXLLoraStep()
        lora_node = ModularPipeline.from_block(lora_step)
        lora_node.update_states(
            **components.get(["text_encoder", "text_encoder_2", "unet"])
        )
        active_adapters = lora_node.get_active_adapters()

        if lora:
            if isinstance(lora, dict):
                lora = [lora]

            set_adapters_list = []
            set_adapters_scale_list = []

            active_adapters = lora_node.get_active_adapters()
            active_adapter_names = set(active_adapters)
            lora_names = set()

            for single_lora in lora:
                adapter_name = single_lora["adapter_name"]
                adapter_scale = single_lora["scale"]

                set_adapters_list.append(adapter_name)
                set_adapters_scale_list.append(adapter_scale)
                lora_names.add(adapter_name)

                if adapter_name not in active_adapters:
                    lora_node.load_lora_weights(
                        single_lora["lora_path"],
                        weight_name=single_lora["weight_name"],
                        adapter_name=single_lora["adapter_name"],
                    )

            # Unload adapters that are in active_adapters but not in lora
            adapters_to_unload = active_adapter_names - lora_names
            lora_node.delete_adapters(adapters_to_unload)

            lora_node.set_adapters(
                set_adapters_list, adapter_weights=set_adapters_scale_list
            )
        else:
            # if there are no loras in the workflow, no need to check, just unload
            if len(active_adapters) > 0:
                lora_node.unload_lora_weights()

        scheduler_options = {}

        if karras:
            scheduler_options["use_karras_sigmas"] = karras

        if v_prediction:
            scheduler_options["prediction_type"] = "v_prediction"
            scheduler_options["rescale_betas_zero_snr"] = True

        if trailing:
            scheduler_options["timestep_spacing"] = "trailing"

        scheduler_cls = getattr(
            __import__("diffusers", fromlist=[scheduler]), scheduler
        )
        denoise.scheduler = scheduler_cls.from_config(
            denoise.scheduler.config, **scheduler_options
        )

        generator = torch.Generator(device="cuda").manual_seed(seed)

        denoise_kwargs = {
            **embeddings.intermediates,
            "generator": generator,
            "guidance_scale": cfg,
            "height": height,
            "width": width,
            "output": "latents",
        }

        if ip_adapter:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                ip_adapter["repo_id"],
                subfolder=ip_adapter["image_encoder_path"],
                torch_dtype=torch.float16,
            )
            feature_extractor = CLIPImageProcessor(size=224, crop_size=224)

            components.add("image_encoder", image_encoder)
            components.add("feature_extractor", feature_extractor)

            ip_adapter_block = StableDiffusionXLIPAdapterStep()
            ip_adapter_node = ModularPipeline.from_block(ip_adapter_block)

            ip_adapter_node.update_states(
                **components.get(["unet", "image_encoder", "feature_extractor"])
            )
            ip_adapter_node.load_ip_adapter(
                ip_adapter["repo_id"],
                subfolder=ip_adapter["subfolder"],
                weight_name=ip_adapter["weight_name"],
            )
            ip_adapter_node.set_ip_adapter_scale(ip_adapter["scale"])

            ip_adapter_state = ip_adapter_node(ip_adapter_image=ip_adapter["image"])
            denoise_kwargs.update(ip_adapter_state.intermediates)

        if image:
            image_node = ModularPipeline.from_block(image_encoder_block)
            image_node.update_states(**components.get(["vae"]))
            image_state = image_node(image=image, generator=generator)

            denoise_kwargs.update(image_state.intermediates)
            denoise_kwargs["strength"] = strength
            denoise_kwargs["num_inference_steps"] = round(steps / strength)
        else:
            denoise_kwargs["num_inference_steps"] = steps

        if controlnet is not None:
            if isinstance(controlnet, dict):
                controlnet = [controlnet]

            # TODO: manage unload of unused controlnets
            if len(controlnet) == 1:
                single_controlnet = controlnet[0]

                if single_controlnet.get("control_mode") is not None:
                    denoise_kwargs["control_mode"] = single_controlnet["control_mode"]

                components.add("controlnet", single_controlnet["controlnet_model"])
                denoise_kwargs["control_image"] = single_controlnet["image"]
                denoise_kwargs["controlnet_conditioning_scale"] = single_controlnet[
                    "conditioning_scale"
                ]
                denoise_kwargs["control_guidance_start"] = single_controlnet[
                    "guidance_start"
                ]
                denoise_kwargs["control_guidance_end"] = single_controlnet[
                    "guidance_end"
                ]
            else:
                controlnet_models = []
                controlnet_images = []
                controlnet_scales = []
                controlnet_start = []
                controlnet_end = []

                for single_controlnet in controlnet:
                    controlnet_models.append(single_controlnet["controlnet_model"])
                    controlnet_images.append(single_controlnet["image"])
                    controlnet_scales.append(single_controlnet["conditioning_scale"])
                    controlnet_start.append(single_controlnet["guidance_start"])
                    controlnet_end.append(single_controlnet["guidance_end"])

                controlnets = MultiControlNetModel(controlnet_models)
                components.add("controlnet", controlnets)

                denoise_kwargs["control_image"] = controlnet_images
                denoise_kwargs["controlnet_conditioning_scale"] = controlnet_scales
                denoise_kwargs["control_guidance_start"] = controlnet_start
                denoise_kwargs["control_guidance_end"] = controlnet_end

            denoise.update_states(**components.get(["controlnet"]))

        if guider is not None:
            denoise.update_states(guider=guider["guider"])
            denoise_kwargs["guider_kwargs"] = guider["guider_kwargs"]

        latents = denoise(**denoise_kwargs)

        return {"latents": latents}


class DecodeLatents(NodeBase):
    def execute(self, components, latents):
        decoder_node = ModularPipeline.from_block(decoder_block)
        decoder_node.update_states(vae=components.get("vae"))
        images_output = decoder_node(latents=latents, output="images")

        images = images_output.images

        return {"images": images}


class Lora(NodeBase):
    def execute(self, path, scale):
        lora_path = os.path.dirname(path)
        weight_name = os.path.basename(path)
        adapter_name = os.path.splitext(weight_name)[0]

        return {
            "lora": {
                "lora_path": lora_path,
                "weight_name": weight_name,
                "adapter_name": adapter_name,
                "scale": scale,
            }
        }


class MultiLora(NodeBase):
    def execute(self, lora_list):
        return {"lora": [single_lora for single_lora in lora_list]}


class PAGOptionalGuider(NodeBase):
    def execute(self, pag_scale, pag_layers):
        # TODO: Maybe do some validations to ensure correct layers
        layers = [item.strip() for item in pag_layers.split(",")]
        guider = PAGGuider(pag_applied_layers=layers)
        guider_kwargs = {"pag_scale": pag_scale}
        return {"guider": {"guider": guider, "guider_kwargs": guider_kwargs}}


class APGOptionalGuider(NodeBase):
    def execute(self, momentum, rescale_factor):
        guider = APGGuider()
        guider_kwargs = {"momentum": momentum, "rescale_factor": rescale_factor}
        return {"guider": {"guider": guider, "guider_kwargs": guider_kwargs}}


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


class MultiControlNet(NodeBase):
    def execute(self, controlnet_list):
        return {
            "controlnet": [single_controlnet for single_controlnet in controlnet_list]
        }


class LoadControlnetUnionModel(NodeBase):
    def execute(self, model_id, variant, dtype):
        controlnet_model = ControlNetUnionModel.from_pretrained(
            model_id, variant=variant, torch_dtype=dtype
        )

        return {"controlnet_union_model": controlnet_model}


class ControlnetUnion(NodeBase):
    def execute(
        self,
        pose_image,
        depth_image,
        edges_image,
        lines_image,
        normal_image,
        segment_image,
        tile_image,
        repaint_image,
        conditioning_scale,
        controlnet_model,
        guidance_start,
        guidance_end,
    ):
        # Map identifiers to their corresponding index and image
        image_map = {
            "pose_image": (pose_image, 0),
            "depth_image": (depth_image, 1),
            "edges_image": (edges_image, 2),
            "lines_image": (lines_image, 3),
            "normal_image": (normal_image, 4),
            "segment_image": (segment_image, 5),
            "tile_image": (tile_image, 6),
            "repaint_image": (repaint_image, 7),
        }

        # Initialize control_mode and control_image
        control_mode = []
        control_image = []

        # Iterate through the dictionary and add non-None images to the lists
        for key, (image, index) in image_map.items():
            if image is not None:
                control_mode.append(index)
                control_image.append(image)

        controlnet = {
            "image": control_image,
            "control_mode": control_mode,
            "conditioning_scale": conditioning_scale,
            "controlnet_model": controlnet_model,
            "guidance_start": guidance_start,
            "guidance_end": guidance_end,
        }
        return {"controlnet": controlnet}


class IPAdapter(NodeBase):
    def execute(
        self,
        image,
        scale,
        repo_id,
        subfolder,
        weight_name,
        image_encoder_path,
    ):
        ip_adapter = {
            "image": image,
            "scale": scale,
            "repo_id": repo_id,
            "subfolder": subfolder,
            "weight_name": weight_name,
            "image_encoder_path": image_encoder_path,
        }

        return {"ip_adapter": ip_adapter}
