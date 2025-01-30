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
    "UNetLoader": {
        "label": "UNet Loader",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "type": "string",
            },
            "subfolder": {
                "label": "Subfolder",
                "type": "string",
            },
            "variant": {
                "label": "Variant",
                "options": ["[unset]", "fp32", "fp16"],
                "postProcess": lambda variant, params: variant
                if variant != "[unset]"
                else None,
                "default": "fp16",
            },
            "dtype": {
                "label": "Dtype",
                "type": "string",
                "options": ["auto", "float32", "float16", "bfloat16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "unet": {
                "label": "UNet",
                "display": "output",
                "type": "model_info",
            },
        },
    },

    "VAELoader": {
        "label": "VAE Loader",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "type": "string",
            },
            "subfolder": {
                "label": "Subfolder",
                "type": "string",
            },
            "variant": {
                "label": "Variant",
                "options": ["[unset]", "fp32", "fp16"],
                "postProcess": lambda variant, params: variant
                if variant != "[unset]"
                else None,
                "default": "fp16",
            },
            "dtype": {
                "label": "Dtype",
                "type": "string",
                "options": ["auto", "float32", "float16", "bfloat16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "vae": {
                "label": "VAE",
                "display": "output",
                "type": "model_info",
            },
        },
    },

    "SDXLModelsLoader": {
        "label": "SDXL Models Loader",
        "category": "Modular Diffusers",
        "params": {
            "repo_id": {
                "label": "Repo ID",
                "type": "string",
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
            "unet": {
                "label": "UNet",
                "display": "input",
                "type": "model_info",
            },
            "vae": {
                "label": "VAE",
                "display": "input",
                "type": "model_info",
            },
            "lora_list": {
                "label": "LoRA List",
                "display": "input",
                "type": "lora",
                "spawn": True,
            },
            "text_encoders": {
                "label": "Text Encoders",
                "display": "output",
                "type": "model_info",
            },
            "unet_out": {
                "label": "UNet",
                "display": "output",
                "type": "model_info",
            },
            "vae_out": {
                "label": "VAE",
                "display": "output",
                "type": "model_info",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "output",
                "type": "model_info",
            },
        },
    },

    "EncodePrompt": {
        "label": "Encode Prompt",
        "category": "Modular Diffusers",
        "params": {
            "text_encoders": {
                "label": "Text Encoders",
                "display": "input",
                "type": "model_info",
            },
            "prompt": {
                "label": "Prompt",
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
            "unet": {
                "label": "UNet",
                "display": "input",
                "type": "model_info",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "input",
                "type": "model_info",
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
            "guider": {
                "label": "Optional Guider",
                "type": "guider",
                "display": "input",
            },
            "controlnet": {
                "label": "Controlnet",
                "type": "controlnet",
                "display": "input",
            },
            "latents": {
                "label": "Latents",
                "type": "latents",
                "display": "output",
            },
            "ip_adapter_image_embeddings": {
                "label": "IP-Adapter Embeddings",
                "type": "ip_adapter_embeddings",
                "display": "input",
            },
        },
    },

    "DecodeLatents": {
        "label": "Decode Latents",
        "category": "Modular Diffusers",
        "params": {
            "vae": {
                "label": "VAE",
                "display": "input",
                "type": "model_info",
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
                "description": "Local path or hub path (org/model_id/filename)",
            },
            "is_local": {
                "label": "Is Local Path",
                "type": "boolean",
                "default": False,
                "description": "Whether the path is a local file path",
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
            "momentum": {
                "label": "Momentum",
                "type": "float",
                "display": "slider",
                "default": -0.5,
                "min": -5,
                "max": 5,
                "step": 0.1,
            },
            "rescale_factor": {
                "label": "Rescale Factor",
                "type": "float",
                "display": "slider",
                "default": 15,
                "min": 0,
                "max": 20,
                "step": 0.1,
            },
            "guider": {
                "label": "Guider",
                "display": "output",
                "type": "guider",
            },
        },
    },

    "ControlnetModelLoader": {
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
                "type": "string",
                "options": ["auto", "float32", "float16", "bfloat16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "controlnet_model": {
                "label": "Controlnet Model",
                "display": "output",
                "type": "model_info",
            },
        },
    },

    "Controlnet": {
        "label": "Controlnet",
        "category": "Modular Diffusers",
        "params": {
            "control_image": {
                "label": "Control Image",
                "type": "image",
                "display": "input",
            },
            "controlnet_conditioning_scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 0.5,
                "min": 0,
                "max": 1,
            },
            "control_guidance_start": {
                "label": "Start",
                "type": "float",
                "display": "slider",
                "default": 0.0,
                "min": 0,
                "max": 1,
            },
            "control_guidance_end": {
                "label": "End",
                "type": "float",
                "display": "slider",
                "default": 1.0,
                "min": 0,
                "max": 1,
            },
            "controlnet_model": {
                "label": "Controlnet Model",
                "type": "model_info",
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

    "IPAdapterInput": {
        "label": "IP-Adapter Input",
        "description": "Configure IP-Adapter settings and input",
        "category": "Modular Diffusers",
        "params": {
            "repo_id": {
                "label": "Repository ID",
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
                "default": "ip-adapter_sdxl.bin",
            },
            "image_encoder_path": {
                "label": "Image Encoder Path",
                "type": "string",
                "default": "sdxl_models/image_encoder",
            },
            "image": {
                "label": "Image",
                "display": "input",
                "type": "image",
            },
            "scale": {
                "label": "Scale",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "max": 1,
                "step": 0.01,
            },
            "ip_adapter_input": {
                "label": "IP-Adapter Input",
                "display": "output",
                "type": "ip_adapter",
            },
        },
    },

    "IPAdapter": {
        "label": "IP-Adapter",
        "description": "Process images with IP-Adapter",
        "category": "Modular Diffusers",
        "params": {
            "unet": {
                "label": "UNet",
                "display": "input",
                "type": "model_info",
            },
            "ip_adapter_inputs": {
                "label": "IP-Adapter Inputs",
                "display": "input",
                "type": "ip_adapter",
                "spawn": True,
            },
            "ip_adapter_image_embeddings": {
                "label": "IP-Adapter Embeddings",
                "display": "output",
                "type": "ip_adapter_embeddings",
            },
            "unet_out": {
                "label": "UNet",
                "display": "output",
                "type": "model_info",
            },
        },
    },


}
