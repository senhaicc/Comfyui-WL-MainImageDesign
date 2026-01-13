"""
Comfyui-WL-MainImageDesign
Elite E-commerce Main Image Prompt Generator

Version: 2.0
Author: WL Design Studio
Description: Professional main image prompt generator with high visual impact optimization.
             Based on deep analysis of 13 high-converting e-commerce hero images.

Features:
- 42pt headline standard
- 1:1 and 3:4 aspect ratio optimization
- 8 professional background styles
- Advanced price & promo badge system
- Commercial photography grade output
- Visual impact score target: 9/10
"""

from .prompt_nodes import WLMainImageGenerator, WLPromptBatchConverter

NODE_CLASS_MAPPINGS = {
    "WLMainImageGenerator": WLMainImageGenerator,
    "WLPromptBatchConverter": WLPromptBatchConverter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WLMainImageGenerator": "🎨 WL Main Image Designer",
    "WLPromptBatchConverter": "🔄 WL Prompt Batch Converter"
}

__version__ = "2.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
