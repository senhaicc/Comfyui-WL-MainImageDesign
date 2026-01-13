"""
Comfyui-WL-MainImageDesign
Elite E-commerce Main Image Prompt Generator

Version: 2.1 - Enhanced Visual Impact
Author: WL Design Studio
Description: Professional main image prompt generator with high visual impact optimization.
             Features scene integration and lifestyle context for maximum conversion.

Features:
- 42pt headline standard
- 1:1 and 3:4 aspect ratio optimization
- 9 professional styles with SCENE INTEGRATION
- 3 scene modes (融合/棚拍/动态)
- Advanced price & promo badge system
- Commercial photography grade output
- Visual impact score target: 9/10

Styles (场景融合版):
1. 专业机能风 - Professional Functional (车库/检查装备场景)
2. 硬核竞技风 - Racing / Track (赛道/动态骑行场景)
3. 工业机械风 - Industrial / Mechanical (机械车间/改装场景)
4. 都市通勤风 - Urban / Daily Ride (城市街道/日常生活场景)
5. 户外冒险风 - ADV / Touring (山路/荒野探索场景)
6. 高端质感风 - Premium / Luxury (展厅/私人车库场景)
7. 安全守护风 - Safety Focused (测试/家人送别场景)
8. 改装美学风 - Custom / Style (改装车间/潮流街拍场景)
9. 参数对比风 - Spec / Data (对比展示/实验室场景)

Scene Modes:
- 场景融合（产品+使用情境）- Product in usage context
- 纯产品棚拍（干净背景）- Studio with designed background
- 骑行动态场景（人车合一）- Dynamic riding scenes
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

__version__ = "2.1.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
