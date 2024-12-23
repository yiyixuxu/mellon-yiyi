import torch

from utils.hf_utils import list_local_models
from utils.torch_utils import default_device, device_list, str_to_dtype

MODULE_MAP = {
    "ModelLoader": {
        "label": "Model Loader",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
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
            "dtype": {
                "label": "Dtype",
                "type": "string",
                "options": ["auto", "float32", "float16", "bfloat16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "unet": {
                "label": "Unet",
                "display": "output",
                "type": "unet",
            },
            "text_encoders": {
                "label": "Text Encoders",
                "display": "output",
                "type": "text_encoders",
            },
            "vae": {
                "label": "Vae",
                "display": "output",
                "type": "vae",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "output",
                "type": "scheduler",
            },
        },
    },
    "EncodePrompts": {
        "label": "Encode Prompts",
        "category": "Modular Diffusers",
        "params": {
            "text_encoders": {
                "label": "Text Encoders",
                "display": "input",
                "type": "text_encoders",
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
            "device": {
                "label": "Device",
                "type": "string",
                "options": device_list,
                "default": default_device,
            },
            "embeddings": {
                "label": "Embeddings",
                "display": "output",
                "type": "prompt_embeddings",
            },
        },
    },
    "EncodeImage": {
        "label": "Encode Image",
        "category": "Modular Diffusers",
        "params": {
            "vae": {
                "label": "Vae",
                "display": "input",
                "type": "vae",
            },
            "image": {
                "label": "Image",
                "type": "image",
                "display": "input",
            },
            "device": {
                "label": "Device",
                "type": "string",
                "options": device_list,
                "default": default_device,
            },
            "latents": {
                "label": "Latents",
                "type": "image_latents",
                "display": "output",
            },
        },
    },
    "DenoiseLoop": {
        "label": "Denoise Loop",
        "category": "Modular Diffusers",
        "params": {
            "unet": {
                "label": "Unet",
                "display": "input",
                "type": "unet",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "input",
                "type": "scheduler",
            },
            "embeddings": {
                "label": "Embeddings",
                "display": "input",
                "type": "prompt_embeddings",
            },
            "controlnet": {
                "label": "Controlnet",
                "type": "controlnet",
                "display": "input",
            },
            "cfg": {
                "label": "Guidance",
                "type": "float",
                "display": "slider",
                "default": 7.0,
                "min": 0,
                "max": 20,
            },
            "steps": {
                "label": "Steps",
                "type": "int",
                "default": 25,
                "min": 1,
                "max": 1000,
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
            "image_latents": {
                "label": "Image Latents",
                "display": "input",
                "type": "image_latents",
            },
            "strength": {
                "label": "Strength)",
                "type": "float",
                "display": "slider",
                "default": 0.5,
                "min": 0,
                "max": 1,
            },
            "guider": {
                "label": "Optional Guider",
                "type": "guider",
                "display": "input",
            },
            "device": {
                "label": "Device",
                "type": "string",
                "options": device_list,
                "default": default_device,
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
            "latents": {
                "label": "Latents",
                "type": "latents",
                "display": "input",
            },
            "vae": {
                "label": "Vae",
                "display": "input",
                "type": "vae",
            },
            "device": {
                "label": "Device",
                "type": "string",
                "options": device_list,
                "default": default_device,
            },
            "images": {
                "label": "Images",
                "type": "image",
                "display": "output",
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
    "DepthEstimator": {
        "label": "Depth Estimator",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "type": "string",
                "description": "The ID of the model to use",
                "default": "depth-anything/Depth-Anything-V2-Large-hf",
            },
            "image": {
                "label": "Conditioning image",
                "type": "image",
                "display": "input",
            },
            "resolution_scale": {
                "label": "Resolution Scale",
                "type": "float",
                "display": "slider",
                "default": 1.0,
                "min": 0,
                "max": 1,
            },
            "invert": {
                "label": "Invert image",
                "type": "boolean",
                "default": False,
            },
            "device": {
                "label": "Device",
                "type": "string",
                "options": device_list,
                "default": default_device,
            },
            "depth_image": {
                "label": "Depth Image",
                "display": "output",
                "type": "image",
            },
        },
    },
}
