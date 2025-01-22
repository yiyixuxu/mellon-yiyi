import torch

from utils.hf_utils import list_local_models
from utils.torch_utils import default_device, device_list, str_to_dtype


class Scheduler:
    def __init__(self, name, scheduler_class, scheduler_args):
        self.name = name
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_args


schedulers = {
    "DDIMScheduler": "DDIM",
    "DDPMScheduler": "DDPM",
    "DEISMultistepScheduler": "DEIS",
    "DPMSolverSinglestepScheduler": "DPM++ 2S",
    "DPMSolverMultistepScheduler": "DPM++ 2M",
    "DPMSolverSDEScheduler": "DPM++ SDE",
    "EDMDPMSolverMultistepScheduler": "DPM++ 2M EDM",
    "EulerDiscreteScheduler": "Euler",
    "EulerAncestralDiscreteScheduler": "Euler Ancestral",
    "HeunDiscreteScheduler": "Heun",
    "KDPM2DiscreteScheduler": "KDPM2",
    "KDPM2AncestralDiscreteScheduler": "KDPM2 Ancestral",
    "LCMScheduler": "LCM",
    "LMSDiscreteScheduler": "LMS",
    "PNDMScheduler": "PNDM",
    "TCDScheduler": "TCD",
    "UniPCMultistepScheduler": "UniPC",
}

MODULE_MAP = {
    "ComponentsLoader": {
        "label": "Components Loader",
        "category": "Modular Diffusers",
        "params": {
            "repo_id": {
                "label": "Repo ID",
                "options": list_local_models(),
                "display": "autocomplete",
                "no_validation": True,
                "default": "stabilityai/stable-diffusion-xl-base-1.0",
            },
            "variant": {
                "label": "Variant",
                "options": ["[unset]", "fp32", "fp16"],
                "postProcess": lambda variant, params: variant
                if variant != "[unset]"
                else None,
                "default": "fp16",
            },
            "device": {
                "label": "Device",
                "type": "string",
                "options": device_list,
                "default": default_device,
            },
            "dtype": {
                "label": "Dtype",
                "type": "string",
                "options": ["auto", "float32", "float16", "bfloat16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "components": {
                "label": "Components",
                "display": "output",
                "type": "diffusers_components",
            },
        },
    },
    "TextEncoder": {
        "label": "Text Encoder",
        "category": "Modular Diffusers",
        "params": {
            "components": {
                "label": "Components",
                "display": "input",
                "type": "diffusers_components",
            },
            "positive_prompt": {
                "label": "Positive Prompt",
                "type": "string",
                "display": "textarea",
            },
            "negative_prompt": {
                "label": "Negative Prompt",
                "type": "string",
                "display": "textarea",
            },
            "embeddings": {
                "label": "Embeddings",
                "display": "output",
                "type": "prompt_embeddings",
            },
        },
    },
    "Denoise": {
        "label": "Denoise",
        "category": "Modular Diffusers",
        "params": {
            "components": {
                "label": "Components",
                "display": "input",
                "type": "diffusers_components",
            },
            "embeddings": {
                "label": "Embeddings",
                "display": "input",
                "type": "prompt_embeddings",
            },
            "steps": {
                "label": "Steps",
                "type": "int",
                "default": 25,
                "min": 1,
                "max": 1000,
            },
            "cfg": {
                "label": "Guidance",
                "type": "float",
                "display": "slider",
                "default": 7.0,
                "min": 0,
                "max": 20,
            },
            "seed": {
                "label": "Seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "display": "random",
            },
            "width": {
                "label": "Width",
                "type": "int",
                "display": "text",
                "default": 1024,
                "min": 8,
                "max": 8192,
                "step": 8,
                "group": "dimensions",
            },
            "height": {
                "label": "Height",
                "type": "int",
                "display": "text",
                "default": 1024,
                "min": 8,
                "max": 8192,
                "step": 8,
                "group": "dimensions",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "select",
                "type": ["string", "scheduler"],
                "options": schedulers,
                "default": "EulerDiscreteScheduler",
            },
            "karras": {
                "label": "Karras",
                "type": "boolean",
                "default": False,
                "group": {
                    "key": "scheduler_options",
                    "label": "Scheduler Options",
                    "display": "collapse",
                },
            },
            "trailing": {
                "label": "Trailing",
                "type": "boolean",
                "default": False,
                "group": "scheduler_options",
            },
            "v_prediction": {
                "label": "V-Prediction",
                "type": "boolean",
                "default": False,
                "group": "scheduler_options",
            },
            "image": {
                "label": "Image",
                "type": "image",
                "display": "input",
                "group": {
                    "key": "image_to_image",
                    "label": "Image to Image / Inpainting",
                    "display": "collapse",
                },
            },
            "strength": {
                "label": "Strength",
                "type": "float",
                "display": "slider",
                "default": 0.5,
                "min": 0,
                "max": 1,
                "step": 0.01,
                "group": "image_to_image",
            },
            "guider": {
                "label": "Optional Guider",
                "type": "guider",
                "display": "input",
            },
            "lora": {
                "label": "Lora",
                "display": "input",
                "type": "lora",
            },
            "controlnet": {
                "label": "Controlnet",
                "type": "controlnet",
                "display": "input",
            },
            "ip_adapter": {
                "label": "IP Adapter",
                "type": "ip_adapter",
                "display": "input",
            },
            "latents": {
                "label": "Latents",
                "type": "latents",
                "display": "output",
            },
        },
    },
    "DecodeLatents": {
        "label": "Decode Latents",
        "category": "Modular Diffusers",
        "params": {
            "components": {
                "label": "Components",
                "display": "input",
                "type": "diffusers_components",
            },
            "latents": {
                "label": "Latents",
                "type": "latents",
                "display": "input",
            },
            "images": {
                "label": "Images",
                "type": "image",
                "display": "output",
            },
        },
    },
    "Lora": {
        "label": "Lora",
        "category": "Modular Diffusers",
        "params": {
            "path": {
                "label": "Path",
                "type": "string",
            },
            "scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 1.0,
                "min": -10,
                "max": 10,
                "step": 0.1,
            },
            "lora": {
                "label": "Lora",
                "type": "lora",
                "display": "output",
            },
        },
    },
    "MultiLora": {
        "label": "Multi Lora",
        "category": "Modular Diffusers",
        "params": {
            "lora_list": {
                "label": "Lora",
                "display": "input",
                "type": "lora",
                "spawn": True,
            },
            "lora": {
                "label": "Multi Loras",
                "type": "lora",
                "display": "output",
            },
        },
    },
    "PAGOptionalGuider": {
        "label": "Perturbed Attention Guidance",
        "category": "Modular Diffusers",
        "params": {
            "pag_scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 3,
                "min": 0,
                "max": 5,
            },
            "pag_layers": {
                "label": "PAG Layers",
                "type": "string",
                "default": "mid",
            },
            "guider": {
                "label": "Guider",
                "display": "output",
                "type": "guider",
            },
        },
    },
    "APGOptionalGuider": {
        "label": "Adaptive Projected Guidance",
        "category": "Modular Diffusers",
        "params": {
            "rescale_factor": {
                "label": "Rescale Factor",
                "type": "float",
                "display": "slider",
                "default": 15,
                "min": 0,
                "max": 20,
                "step": 0.1,
            },
            "momentum": {
                "label": "Momentum",
                "type": "float",
                "display": "slider",
                "default": -0.5,
                "min": -5,
                "max": 5,
                "step": 0.1,
            },
            "guider": {
                "label": "Guider",
                "display": "output",
                "type": "guider",
            },
        },
    },
    "LoadControlnetModel": {
        "label": "Load Controlnet Model",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "type": "string",
                "description": "The ID of the model to use",
                "default": "xinsir/controlnet-depth-sdxl-1.0",
            },
            "variant": {
                "label": "Variant",
                "options": ["[unset]", "fp32", "fp16"],
                "postProcess": lambda variant, params: variant
                if variant != "[unset]"
                else None,
                "default": "fp16",
                "description": "The variant of the checkpoint to use",
            },
            "dtype": {
                "label": "dtype",
                "options": ["Auto", "fp32", "fp16", "bf16"],
                "postProcess": lambda dtype, params: {
                    "[unset]": torch.float32,
                    "fp32": torch.float32,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                }[dtype if dtype != "Auto" else params["variant"]],
                "default": "Auto",
                "description": 'The data type to convert the model to. If "Auto" is selected, the data type will be inferred from the checkpoint.',
            },
            "controlnet_model": {
                "label": "Controlnet Model",
                "display": "output",
                "type": "controlnet_model",
            },
        },
    },
    "Controlnet": {
        "label": "Controlnet",
        "category": "Modular Diffusers",
        "params": {
            "image": {
                "label": "Conditioning image",
                "type": "image",
                "display": "input",
            },
            "conditioning_scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 0.5,
                "min": 0,
                "max": 1,
            },
            "guidance_start": {
                "label": "Start",
                "type": "float",
                "display": "slider",
                "default": 0.0,
                "min": 0,
                "max": 1,
            },
            "guidance_end": {
                "label": "End",
                "type": "float",
                "display": "slider",
                "default": 1.0,
                "min": 0,
                "max": 1,
            },
            "controlnet_model": {
                "label": "Controlnet Model",
                "type": "controlnet_model",
                "display": "input",
            },
            "controlnet": {
                "label": "Controlnet",
                "display": "output",
                "type": "controlnet",
            },
        },
    },
    "MultiControlNet": {
        "label": "Multi ControlNet",
        "category": "Modular Diffusers",
        "params": {
            "controlnet_list": {
                "label": "ControlNet",
                "display": "input",
                "type": "controlnet",
                "spawn": True,
            },
            "controlnet": {
                "label": "Multi Controlnet",
                "type": "controlnet",
                "display": "output",
            },
        },
    },
    "LoadControlnetUnionModel": {
        "label": "Load Controlnet Union Model",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "type": "string",
                "description": "The ID of the model to use",
                "default": "OzzyGT/controlnet-union-promax-sdxl-1.0",
            },
            "variant": {
                "label": "Variant",
                "options": ["[unset]", "fp32", "fp16"],
                "postProcess": lambda variant, params: variant
                if variant != "[unset]"
                else None,
                "default": "fp16",
                "description": "The variant of the checkpoint to use",
            },
            "dtype": {
                "label": "dtype",
                "options": ["Auto", "fp32", "fp16", "bf16"],
                "postProcess": lambda dtype, params: {
                    "[unset]": torch.float32,
                    "fp32": torch.float32,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                }[dtype if dtype != "Auto" else params["variant"]],
                "default": "Auto",
                "description": 'The data type to convert the model to. If "Auto" is selected, the data type will be inferred from the checkpoint.',
            },
            "controlnet_union_model": {
                "label": "Controlnet Union Model",
                "display": "output",
                "type": "controlnet_union_model",
            },
        },
    },
    "ControlnetUnion": {
        "label": "Controlnet Union",
        "category": "Modular Diffusers",
        "params": {
            "pose_image": {
                "label": "Pose image",
                "type": "image",
                "display": "input",
            },
            "depth_image": {
                "label": "Depth image",
                "type": "image",
                "display": "input",
            },
            "edges_image": {
                "label": "Edges image",
                "type": "image",
                "display": "input",
            },
            "lines_image": {
                "label": "Lines image",
                "type": "image",
                "display": "input",
            },
            "normal_image": {
                "label": "Normal image",
                "type": "image",
                "display": "input",
            },
            "segment_image": {
                "label": "Segment image",
                "type": "image",
                "display": "input",
            },
            "tile_image": {
                "label": "Tile image",
                "type": "image",
                "display": "input",
            },
            "repaint_image": {
                "label": "Repaint image",
                "type": "image",
                "display": "input",
            },
            "conditioning_scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 0.5,
                "min": 0,
                "max": 1,
            },
            "guidance_start": {
                "label": "Start",
                "type": "float",
                "display": "slider",
                "default": 0.0,
                "min": 0,
                "max": 1,
            },
            "guidance_end": {
                "label": "End",
                "type": "float",
                "display": "slider",
                "default": 1.0,
                "min": 0,
                "max": 1,
            },
            "controlnet_model": {
                "label": "Controlnet Union Model",
                "type": "controlnet_union_model",
                "display": "input",
            },
            "controlnet": {
                "label": "Controlnet",
                "display": "output",
                "type": "controlnet",
            },
        },
    },
    "IPAdapter": {
        "label": "IP Adapter",
        "category": "Modular Diffusers",
        "params": {
            "image": {
                "label": "IP Adapter Image",
                "type": "image",
                "display": "input",
            },
            "scale": {
                "label": "Start",
                "type": "float",
                "display": "slider",
                "default": 0.0,
                "min": 0,
                "max": 1,
            },
            "repo_id": {
                "label": "Repo ID",
                "type": "string",
                "default": "h94/IP-Adapter",
            },
            "subfolder": {
                "label": "Subfolder",
                "type": "string",
                "default": "sdxl_models",
            },
            "weight_name": {
                "label": "Weight Name",
                "type": "string",
                "default": "ip-adapter-plus_sdxl_vit-h.safetensors",
            },
            "image_encoder_path": {
                "label": "Image Encoder Path",
                "type": "string",
                "default": "models/image_encoder",
            },
            "ip_adapter": {
                "label": "IP Adapter",
                "type": "ip_adapter",
                "display": "output",
            },
        },
    },
}
