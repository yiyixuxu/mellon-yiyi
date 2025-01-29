import os
import time
import copy

import torch
from diffusers import ControlNetModel, ControlNetUnionModel, ModularPipeline, DiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.guider import APGGuider, PAGGuider
from diffusers.pipelines.components_manager import ComponentsManager
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_modular import (
    AUTO_BLOCKS,
    StableDiffusionXLTextEncoderStep,
    StableDiffusionXLIPAdapterStep,
    StableDiffusionXLLoraStep,
    StableDiffusionXLAutoVaeEncoderStep,
    StableDiffusionXLAutoDecodeStep,
)
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mellon.NodeBase import NodeBase, are_different, format_value


# YiYi TODO: make the SDXLAutoBlocks user can directly import 
all_blocks_map = AUTO_BLOCKS.copy()
text_block = all_blocks_map.pop("text_encoder")()
decoder_block = all_blocks_map.pop("decode")()
image_encoder_block = all_blocks_map.pop("image_encoder")()
class SDXLAutoBlocks(SequentialPipelineBlocks):
    block_classes = list(all_blocks_map.values())
    block_names = list(all_blocks_map.keys())


# YiYi TODO: add it to diffusers

def update_lora_adapters(lora_node, lora_list):
    """
    Update LoRA adapters based on the provided list of LoRAs.
    
    Args:
        lora_node: ModularPipeline node containing LoRA functionality
        lora_list: List of dictionaries or single dictionary containing LoRA configurations with:
                  {'lora_path': str, 'weight_name': str, 'adapter_name': str, 'scale': float}
    """
    # Convert single lora to list if needed
    if not isinstance(lora_list, list):
        lora_list = [lora_list]
        
    # Get currently loaded adapters
    loaded_adapters = list(set().union(*lora_node.get_list_adapters().values()))
    
    # Determine which adapters to set and remove
    to_set = [lora["adapter_name"] for lora in lora_list]
    to_remove = [adapter for adapter in loaded_adapters if adapter not in to_set]
    
    # Remove unused adapters first
    for adapter_name in to_remove:
        lora_node.delete_adapters(adapter_name)
    
    # Load new LoRAs and set their scales
    scales = {}
    for lora in lora_list:
        adapter_name = lora["adapter_name"]
        if adapter_name not in loaded_adapters:
            lora_node.load_lora_weights(
                lora["lora_path"],
                weight_name=lora["weight_name"],
                adapter_name=adapter_name
            )
        scales[adapter_name] = lora["scale"]
    
    # Set adapter scales
    if scales:
        lora_node.set_adapters(
            list(scales.keys()),
            list(scales.values())
        )




components = ComponentsManager()


class ControlnetModelLoader(NodeBase):
    def __del__(self):
        components.remove(f"controlnet_{self.node_id}")
        super().__del__()

    def execute(self, model_id, variant, dtype):
        controlnet_model = ControlNetModel.from_pretrained(
            model_id, variant=variant, torch_dtype=dtype
        )
        components.add(f"controlnet_{self.node_id}", controlnet_model)
        return {"controlnet_model": components.get_model_info(f"controlnet_{self.node_id}")}

class UNetLoader(NodeBase):
    def __del__(self):
        components.remove(f"unet_{self.node_id}")
        super().__del__()

    def execute(self, model_id, subfolder, variant, dtype):
        print(f" load unet from model_id: {model_id}, subfolder: {subfolder}, variant: {variant}, dtype: {dtype}")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder=subfolder, variant=variant, torch_dtype=dtype
        )
        components.add(f"unet_{self.node_id}", unet)
        return {"unet": components.get_model_info(f"unet_{self.node_id}")}

class VAELoader(NodeBase):
    def __del__(self):
        components.remove(f"vae_{self.node_id}")
        super().__del__()

    def execute(self, model_id, subfolder=None, variant=None, dtype=None):
        # Normalize parameters
        subfolder = None if subfolder == "" else subfolder

        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder=subfolder, variant=variant, torch_dtype=dtype
        )
        components.add(f"vae_{self.node_id}", vae)
        return {"vae": components.get_model_info(f"vae_{self.node_id}")}

class SDXLModelsLoader(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        # lora
        lora_step = StableDiffusionXLLoraStep()
        self._lora_node = ModularPipeline.from_block(lora_step)
        # ip adapter
        ip_adapter_block = StableDiffusionXLIPAdapterStep()
        self._ip_adapter_node = ModularPipeline.from_block(ip_adapter_block)


    def __del__(self):
        components.remove(f"text_encoder_{self.node_id}")
        components.remove(f"text_encoder_2_{self.node_id}")
        components.remove(f"tokenizer_{self.node_id}")
        components.remove(f"tokenizer_2_{self.node_id}")
        components.remove(f"scheduler_{self.node_id}")
        components.remove(f"unet_{self.node_id}")
        components.remove(f"vae_{self.node_id}")
        components.remove(f"image_encoder_{self.node_id}")
        components.remove(f"feature_extractor_{self.node_id}")
        self._lora_node.unload_lora_weights()
        self._ip_adapter_node.unload_ip_adapter()
        super().__del__()
    
    def __call__(self, **kwargs):
        self._old_params = copy.deepcopy(self.params)
        return super().__call__(**kwargs)

    def execute(self, repo_id, variant, device, dtype, unet=None, vae=None, lora_list=None, ip_adapter_input=None):
        def _has_changed(old_params, new_params):
            for key in new_params:
                new_value = new_params.get(key)
                old_value = old_params.get(key)
                
                if new_value is not None and key not in old_params:
                    return True
                if are_different(old_value, new_value):
                    return True
            return False

        print(f" in SDXLModelsLoader execute: {self.node_id}")
        print(f" old_params: {self._old_params}")
        print(f" new params:")
        print(f" - repo_id: {repo_id}")
        print(f" - variant: {variant}")
        print(f" - dtype: {dtype}")
        print(f" - unet: {unet}")
        print(f" - vae: {vae}")
        print(f" - lora_list: {lora_list}")
        print(f" ip_adapter_input: {ip_adapter_input}")
        loaded_components = {}

        repo_changed = _has_changed(self._old_params, {'repo_id': repo_id, 'variant': variant, 'dtype': dtype})
        unet_input_changed = _has_changed(self._old_params, {'unet': unet})
        vae_input_changed = _has_changed(self._old_params, {'vae': vae})
        lora_input_changed = _has_changed(self._old_params, {'lora_list': lora_list})

        # ip_adapter_input_changed = _has_changed(self._old_params, {'ip_adapter_input': ip_adapter_input})
        if ip_adapter_input and "ip_adapter_input" in self._old_params:
            filtered_old_pram = {k: w for k, w in self._old_params["ip_adapter_input"].items() if k != "scale"}
            filtered_ip_adapter_input = {k: w for k, w in ip_adapter_input.items() if k != "scale"}
            ip_adapter_input_changed = _has_changed(filtered_old_pram, filtered_ip_adapter_input)
        else:
            ip_adapter_input_changed = _has_changed(self._old_params, {'ip_adapter_input': ip_adapter_input})
   

        print(f" Changes detected - repo: {repo_changed}, unet: {unet_input_changed}, vae: {vae_input_changed}, "
              f"lora: {lora_input_changed}, ip_adapter: {ip_adapter_input_changed}")

        unet_changed = unet_input_changed or (unet is None and repo_changed)
        vae_changed = vae_input_changed or (vae is None and repo_changed)

        # Load and update base models
        if unet is None:
            if repo_changed or unet_input_changed:
                print(f" load unet from repo_id: {repo_id}, subfolder: unet, variant: {variant}, dtype: {dtype}")
                unet = UNet2DConditionModel.from_pretrained(
                    repo_id, subfolder="unet", variant=variant, torch_dtype=dtype
                )
                # Only add to components if we loaded it ourselves
                print(f" add unet to components: unet_{self.node_id}")
                components.add(f'unet_{self.node_id}', unet)
            else:
                # Get existing unet from our components
                unet = components.get(f'unet_{self.node_id}')
            unet_model_id = f'unet_{self.node_id}'  # Always use our node_id if input is None
        else:
            # unet is always a model info dict if not None
            unet_model_id = unet["model_id"]
            unet = components.get(unet_model_id)
        
        if vae is None:
            if repo_changed or vae_input_changed:
                print(f" load vae from repo_id: {repo_id}, subfolder: vae, variant: {variant}, dtype: {dtype}")
                vae = AutoencoderKL.from_pretrained(
                    repo_id, subfolder="vae", variant=variant, torch_dtype=dtype
                )
                # Only add to components if we loaded it ourselves
                print(f" add vae to components: vae_{self.node_id}")
                components.add(f'vae_{self.node_id}', vae)
            else:
                # Get existing vae from our components
                vae = components.get(f'vae_{self.node_id}')
            vae_model_id = f'vae_{self.node_id}'  # Always use our node_id if input is None
        else:
            # vae is always a model info dict if not None
            vae_model_id = vae["model_id"]
            vae = components.get(vae_model_id)
        
        # Load text encoders and scheduler
        if repo_changed:
            print(f" load text encoders and scheduler from pipeline: {repo_id}, variant: {variant}, dtype: {dtype}")
            pipe = DiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                variant=variant,
                vae=None,
                unet=None,
            )
            components.add(f'text_encoder_{self.node_id}', pipe.text_encoder)
            components.add(f'text_encoder_2_{self.node_id}', pipe.text_encoder_2)
            components.add(f'tokenizer_{self.node_id}', pipe.tokenizer)
            components.add(f'tokenizer_2_{self.node_id}', pipe.tokenizer_2)
            components.add(f'scheduler_{self.node_id}', pipe.scheduler)

        # Handle LoRA
        if not lora_list:
            print(f" unload lora from components: lora_{self.node_id}")
            self._lora_node.unload_lora_weights()
        elif lora_input_changed or unet_changed:
            print(f" update lora from components: lora_{self.node_id}")
            
            # Unload first to clean previous model's state
            self._lora_node.unload_lora_weights()
            
            self._lora_node.update_states(
                unet=unet,  # Use actual model
                text_encoder=components.get(f'text_encoder_{self.node_id}'),
                text_encoder_2=components.get(f'text_encoder_2_{self.node_id}'),
            )
            update_lora_adapters(self._lora_node, lora_list)

        # Handle IP-Adapter
        if not ip_adapter_input:
            print(f" unload ip_adapter from components: ip_adapter_{self.node_id}")
            self._ip_adapter_node.unload_ip_adapter()
        elif ip_adapter_input_changed or unet_changed:
            # Unload first to clean previous model's state
            print(" unload ip_adapter")
            self._ip_adapter_node.unload_ip_adapter()
            
            if ip_adapter_input_changed:
                # Handle single or multiple image encoders
                repo_ids = ip_adapter_input["repo_id"]
                image_encoder_paths = ip_adapter_input["image_encoder_path"]
                
                # Convert to lists if single values
                if not isinstance(repo_ids, list):
                    repo_ids = [repo_ids]
                    image_encoder_paths = [image_encoder_paths]
                
                # Load the first image encoder (they should all be the same)
                print(f" load image_encoder from repo_id: {repo_ids[0]}, subfolder: {image_encoder_paths[0]}, dtype: {dtype}")
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    repo_ids[0],    
                    subfolder=image_encoder_paths[0],
                    torch_dtype=dtype,
                )
                feature_extractor = CLIPImageProcessor(size=224, crop_size=224)
                print(f" add image_encoder to components: image_encoder_{self.node_id}")
                components.add(f"image_encoder_{self.node_id}", image_encoder)
                print(f" add feature_extractor to components: feature_extractor_{self.node_id}")
                components.add(f"feature_extractor_{self.node_id}", feature_extractor)
                print(f" update ip_adapter from components: ip_adapter_{self.node_id}")
                self._ip_adapter_node.update_states(image_encoder=image_encoder, feature_extractor=feature_extractor)

            self._ip_adapter_node.update_states(unet=unet)  # Use actual model
            
            # Load each IP-Adapter
            print(" load ip_adapter(s)")
            if isinstance(ip_adapter_input["repo_id"], list):
                for i in range(len(ip_adapter_input["repo_id"])):
                    self._ip_adapter_node.load_ip_adapter(
                        ip_adapter_input["repo_id"][i],
                        subfolder=ip_adapter_input["subfolder"][i],
                        weight_name=ip_adapter_input["weight_name"][i],
                    )
            else:
                self._ip_adapter_node.load_ip_adapter(
                    ip_adapter_input["repo_id"],
                    subfolder=ip_adapter_input["subfolder"],
                    weight_name=ip_adapter_input["weight_name"],
                )
                
        if ip_adapter_input:
            # Set the scale(s)
            scales = ip_adapter_input["scale"]
            if not isinstance(scales, list):
                scales = [scales]
            for scale in scales:
                self._ip_adapter_node.set_ip_adapter_scale(scale)

        if unet_changed or vae_changed or repo_changed:
            components.enable_auto_cpu_offload(device=device)

        # Construct loaded_components at the end after all modifications
        loaded_components = {
            "unet_out": components.get_model_info(unet_model_id),
            "vae_out": components.get_model_info(vae_model_id),
            "text_encoders": {
                k: components.get_model_info(v) 
                for k, v in {
                    "text_encoder": f'text_encoder_{self.node_id}',
                    "text_encoder_2": f'text_encoder_2_{self.node_id}',
                    "tokenizer": f'tokenizer_{self.node_id}',
                    "tokenizer_2": f'tokenizer_2_{self.node_id}',
                }.items()
            },
            "scheduler": components.get_model_info(f'scheduler_{self.node_id}'),
            "ip_adapter": {
                "image_encoder": components.get_model_info(f"image_encoder_{self.node_id}"),
                "feature_extractor": components.get_model_info(f"feature_extractor_{self.node_id}"),
                "unet": components.get_model_info(unet_model_id, fields="model_id"),
            } if ip_adapter_input else None,
        }

        print(f" Final components state: {components}")
        return loaded_components


class EncodePrompt(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        text_block = StableDiffusionXLTextEncoderStep()
        self._text_encoder_node = ModularPipeline.from_block(text_block)

    def execute(self, text_encoders, **kwargs):
        text_encoder_components = {
            "text_encoder": components.get(text_encoders["text_encoder"]["model_id"]),
            "text_encoder_2": components.get(text_encoders["text_encoder_2"]["model_id"]), 
            "tokenizer": components.get(text_encoders["tokenizer"]["model_id"]),
            "tokenizer_2": components.get(text_encoders["tokenizer_2"]["model_id"])
        }
        
        self._text_encoder_node.update_states(**text_encoder_components)
        text_state = self._text_encoder_node(**kwargs)
        # Return the intermediates dict instead of the PipelineState
        return {"embeddings": text_state.intermediates}

# class EncodeImage(NodeBase):
#     def __init__(self, node_id=None):
#         super().__init__(node_id)
#         image_encoder_block = StableDiffusionXLAutoVaeEncoderStep()
#         self._image_encoder_node = ModularPipeline.from_block(image_encoder_block)

#     def execute(self, vae, **kwargs):
#         vae_component = components.get(vae["model_id"])
#         self._image_encoder_node.update_states(vae=vae_component)
#         image_state = self._image_encoder_node(**kwargs)
#         return {"image_embeddings": image_state.intermediates}


class Denoise(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        sdxl_auto_blocks = SDXLAutoBlocks()
        self._denoise_node = ModularPipeline.from_block(sdxl_auto_blocks)

    def execute(
        self,
        unet,
        scheduler,
        embeddings,
        steps,
        cfg,
        seed,
        width,
        height,
        guider,
        controlnet,
        ip_adapter_image_embeddings=None,
    ):
        unet_component = components.get(unet["model_id"])
        scheduler_component = components.get(scheduler["model_id"])
        
        self._denoise_node.update_states(
            unet=unet_component, 
            scheduler=scheduler_component
        )
        
        generator = torch.Generator(device="cpu").manual_seed(seed)

        denoise_kwargs = {
            **embeddings,  # Now embeddings is already a dict
            "generator": generator,
            "guidance_scale": cfg,
            "height": height,
            "width": width,
            "output": "latents",
            "num_inference_steps": steps,
        }

        if ip_adapter_image_embeddings is not None:
            denoise_kwargs.update(**ip_adapter_image_embeddings)  # Now ip_adapter_image_embeddings is already a dict

        if controlnet is not None:
            denoise_kwargs.update(**controlnet["controlnet_inputs"])
            
            # For multiple controlnets, get all models from their IDs
            model_ids = controlnet["controlnet_model"]["model_id"]
            if isinstance(model_ids, list):
                controlnet_components = [components.get(model_id) for model_id in model_ids]
                controlnet_components = MultiControlNetModel(controlnet_components)
            else:
                controlnet_components = components.get(model_ids)
            
            self._denoise_node.update_states(controlnet=controlnet_components)

        if guider is not None:
            self._denoise_node.update_states(guider=guider["guider"])
            denoise_kwargs["guider_kwargs"] = guider["guider_kwargs"]

        latents = self._denoise_node(**denoise_kwargs)
        print(f" Components after Denoise: {components}")
        return {"latents": latents}


class DecodeLatents(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        decoder_block = StableDiffusionXLAutoDecodeStep()
        self._decoder_node = ModularPipeline.from_block(decoder_block)

    def execute(self, vae, latents):
        vae_component = components.get(vae["model_id"])
        self._decoder_node.update_states(vae=vae_component)
        images_output = self._decoder_node(latents=latents, output="images")
        return {"images": images_output.images}


class Lora(NodeBase):
    def execute(self, path, scale, is_local=False):
        if is_local:
            lora_path = os.path.dirname(path)
            weight_name = os.path.basename(path)
        else:
            # Handle hub path format: "org/model_id/filename"
            parts = path.split('/')
            if len(parts) != 3:
                raise ValueError("Hub path must be in format 'org/model_id/filename'")
            lora_path = f"{parts[0]}/{parts[1]}"
            weight_name = parts[2]
        
        adapter_name = os.path.splitext(weight_name)[0]

        # Return the lora configuration directly, not wrapped in another dict
        return {
            "lora": {
                "lora_path": lora_path,
                "weight_name": weight_name,
                "adapter_name": adapter_name,
                "scale": scale,
            }
        }


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


class Controlnet(NodeBase):
    def execute(
        self, 
        control_image, 
        controlnet_conditioning_scale, 
        controlnet_model,
        control_guidance_start, 
        control_guidance_end
    ):
        controlnet = {
            "controlnet_model": controlnet_model,
            "controlnet_inputs": {
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            }
        }
        return {"controlnet": controlnet}


class MultiControlNet(NodeBase):
    def execute(self, controlnet_list):
        # Restructure controlnet_model info into lists
        controlnet_model = {
            "model_id": [],
            "added_time": [],
            "class_name": [],
            "size_gb": [],
            "adapters": []
        }
        
        # Fill the lists with values from each controlnet model
        for c in controlnet_list:
            model_info = c["controlnet_model"]
            for key in controlnet_model:
                controlnet_model[key].append(model_info[key])

        controlnet = {
            "controlnet_model": controlnet_model,
            "controlnet_inputs": {
                "control_image": [c["controlnet_inputs"]["control_image"] for c in controlnet_list],
                "controlnet_conditioning_scale": [c["controlnet_inputs"]["controlnet_conditioning_scale"] for c in controlnet_list],
                "control_guidance_start": [c["controlnet_inputs"]["control_guidance_start"] for c in controlnet_list],
                "control_guidance_end": [c["controlnet_inputs"]["control_guidance_end"] for c in controlnet_list],
            }
        }

        return {"controlnet": controlnet}




# class LoadControlnetUnionModel(NodeBase):
#     def execute(self, model_id, variant, dtype):
#         controlnet_model = ControlNetUnionModel.from_pretrained(
#             model_id, variant=variant, torch_dtype=dtype
#         )
#         components.add(f"controlnet_union_{self.node_id}", controlnet_model)
#         return {"controlnet_union_model": components.get_model_info(f"controlnet_union_{self.node_id}")}


# class ControlnetUnion(NodeBase):
#     def execute(
#         self,
#         pose_image,
#         depth_image,
#         edges_image,
#         lines_image,
#         normal_image,
#         segment_image,
#         tile_image,
#         repaint_image,
#         conditioning_scale,
#         controlnet_model,
#         guidance_start,
#         guidance_end,
#     ):
#         controlnet_component = components.get(controlnet_model["model_id"])
#         # Map identifiers to their corresponding index and image
#         image_map = {
#             "pose_image": (pose_image, 0),
#             "depth_image": (depth_image, 1),
#             "edges_image": (edges_image, 2),
#             "lines_image": (lines_image, 3),
#             "normal_image": (normal_image, 4),
#             "segment_image": (segment_image, 5),
#             "tile_image": (tile_image, 6),
#             "repaint_image": (repaint_image, 7),
#         }

#         control_mode = []
#         control_image = []

#         for key, (image, index) in image_map.items():
#             if image is not None:
#                 control_mode.append(index)
#                 control_image.append(image)

#         controlnet = {
#             "image": control_image,
#             "control_mode": control_mode,
#             "conditioning_scale": conditioning_scale,
#             "controlnet_model": controlnet_component,
#             "guidance_start": guidance_start,
#             "guidance_end": guidance_end,
#         }
#         return {"controlnet": controlnet}


class IPAdapter(NodeBase):
    def execute(
        self,
        repo_id,
        subfolder,
        weight_name,
        image_encoder_path,
        scale,
    ):
        ip_adapter = {
            "repo_id": repo_id,
            "subfolder": subfolder,
            "weight_name": weight_name,
            "image_encoder_path": image_encoder_path,
            "scale": scale,
        }

        return {"ip_adapter": ip_adapter}

class EncodeIPAdapter(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        ip_adapter_block = StableDiffusionXLIPAdapterStep()
        self._ip_adapter_node = ModularPipeline.from_block(ip_adapter_block)    

    def execute(
        self,
        image,
        ip_adapter,
    ):
        unet = components.get(ip_adapter["unet"]["model_id"])
        image_encoder = components.get(ip_adapter["image_encoder"]["model_id"])
        feature_extractor = components.get(ip_adapter["feature_extractor"]["model_id"]) 
        
        self._ip_adapter_node.update_states(
            unet=unet, 
            image_encoder=image_encoder, 
            feature_extractor=feature_extractor
        )
        
        ip_adapter_state = self._ip_adapter_node(ip_adapter_image=image)
        # Return the intermediates dict instead of the PipelineState
        return {"ip_adapter_image_embeddings": ip_adapter_state.intermediates}

class MultiIPAdapter(NodeBase):
    def execute(self, ip_adapter_list):
        # Initialize the combined ip_adapter dict with the same structure
        ip_adapter = {
            "repo_id": [],
            "subfolder": [],
            "weight_name": [],
            "image_encoder_path": [],
            "scale": [],
        }
        
        # Fill the lists with values from each ip_adapter input
        for adapter in ip_adapter_list:
            for key in ip_adapter:
                ip_adapter[key].append(adapter[key])

        return {"ip_adapter": ip_adapter}

