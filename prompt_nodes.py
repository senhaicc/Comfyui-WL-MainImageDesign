"""
Comfyui-WL-MainImageDesign
Elite E-commerce Main Image Prompt Generator
Version: 2.1 - Enhanced Visual Impact
"""

import json
import urllib.request
import urllib.error
import ssl
import base64
import io
import os
import numpy as np
from PIL import Image

# 获取当前模块目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_prompt(filename):
    """从外部文件加载 prompt"""
    prompt_file = os.path.join(_CURRENT_DIR, "prompts", filename)
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ Prompt file not found: {prompt_file}")
        return None
    except Exception as e:
        print(f"⚠️ Error loading prompt file {filename}: {e}")
        return None

def _load_system_prompt():
    """加载主 system prompt"""
    prompt = _load_prompt("system_prompt.txt")
    if prompt is None:
        print("⚠️ Falling back to default_prompt.txt")
        prompt = _load_prompt("default_prompt.txt")
    if prompt is None:
        raise FileNotFoundError("No prompt files found in prompts/ directory.")
    return prompt


class WLMainImageGenerator:
    """
    WL Main Image Design Generator
    专业电商主图提示词生成器 - 视觉冲击力优化版
    """
    
    def __init__(self):
        pass

    def split_response_to_variants(self, text, prompt_count):
        """将响应拆分为多个变体"""
        if text is None:
            return []

        s = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not s:
            return []

        if prompt_count is None or int(prompt_count) <= 1:
            return [s]

        import re

        json_obj_pattern = r'\{\s*"prompt"\s*:\s*"'
        matches = list(re.finditer(json_obj_pattern, s))
        if len(matches) >= 2:
            parsed_objects = []
            idxs = [m.start() for m in matches]
            for i, start_idx in enumerate(idxs):
                end_idx = idxs[i + 1] if i + 1 < len(idxs) else len(s)
                chunk = s[start_idx:end_idx].strip().rstrip(',')
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict) and "prompt" in obj:
                        parsed_objects.append(obj["prompt"])
                    else:
                        parsed_objects.append(chunk)
                except json.JSONDecodeError:
                    clean = re.sub(r'^\s*\{\s*"prompt"\s*:\s*"', '', chunk)
                    clean = re.sub(r'"\s*\}\s*$', '', clean)
                    parsed_objects.append(clean)
            if parsed_objects:
                return parsed_objects

        start_markers = [
            r"(?m)^\s*变体定位\s*[：:]",
            r"(?m)^\s*Variant Role\s*:",
            r"(?m)^\s*主标题\s*[：:]",
            r"(?m)^\s*Main Headline\s*:",
        ]
        for pat in start_markers:
            matches = list(re.finditer(pat, s))
            if len(matches) >= 2:
                idxs = [m.start() for m in matches] + [len(s)]
                parts = [s[idxs[i]:idxs[i + 1]].strip() for i in range(len(idxs) - 1)]
                parts = [p for p in parts if p]
                if parts:
                    return parts

        if "\n---\n" in s:
            parts = [p.strip() for p in s.split("\n---\n")]
            parts = [p for p in parts if p]
            if parts:
                return parts

        parts = [p.strip() for p in re.split(r"\n\s*\n\s*\n+", s)]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts

        return [s]

    def _clean_code_fences(self, response_text):
        cleaned = (response_text or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _parse_response_to_prompts_list(self, response_text, expected_count):
        method = "json"
        prompts_list = []
        try:
            cleaned = self._clean_code_fences(response_text)
            prompts_list = json.loads(cleaned)
            if not isinstance(prompts_list, list):
                method = "split:not_list"
                prompts_list = self.split_response_to_variants(response_text, expected_count)
        except json.JSONDecodeError:
            method = "split:json_decode_error"
            prompts_list = self.split_response_to_variants(response_text, expected_count)
        except Exception:
            method = "split:exception"
            prompts_list = self.split_response_to_variants(response_text, expected_count)

        normalized = []
        for item in prompts_list if isinstance(prompts_list, list) else [prompts_list]:
            text = self.extract_prompt_text(item)
            if text is None:
                continue
            normalized.append(self._strip_variant_role_header(text))

        if normalized:
            prompts_list = normalized
        else:
            prompts_list = []

        prompts_list = self.enforce_prompt_count(prompts_list, expected_count, response_text)
        if len(prompts_list) > expected_count:
            prompts_list = prompts_list[:expected_count]

        return prompts_list, method

    def _strip_variant_role_header(self, prompt_text):
        if not isinstance(prompt_text, str):
            return prompt_text
        return prompt_text.strip()

    def _is_prompt_structurally_complete(self, prompt_text):
        if not isinstance(prompt_text, str):
            return False
        s = prompt_text.strip()
        if not s:
            return False
        has_main = ("主标题" in s) or ("Main Headline" in s)
        has_visual = ("视觉" in s) or ("Visual" in s) or ("光影" in s) or ("Lighting" in s) or ("背景" in s) or ("Background" in s)
        return has_main and has_visual

    def enforce_prompt_count(self, prompts_list, prompt_count, raw_response):
        try:
            pc = int(prompt_count)
        except Exception:
            pc = None
        if not pc or pc <= 0:
            return prompts_list
        if not prompts_list:
            return [str(raw_response)]
        if len(prompts_list) == pc:
            return prompts_list
        if len(prompts_list) > pc:
            return prompts_list[:pc]
        if len(prompts_list) == 1 and isinstance(prompts_list[0], str):
            parts = self.split_response_to_variants(prompts_list[0], pc)
            if parts and len(parts) >= pc:
                return parts[:pc]
        parts = self.split_response_to_variants(raw_response, pc)
        if parts and len(parts) >= pc:
            return parts[:pc]
        return prompts_list

    def extract_prompt_text(self, item):
        if item is None:
            return None
        if isinstance(item, dict):
            prompt = item.get("prompt")
            if isinstance(prompt, str):
                return prompt
            return json.dumps(item, ensure_ascii=False)
        if isinstance(item, str):
            s = item.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[{") and s.endswith("}]")):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
                        return obj["prompt"]
                except Exception:
                    pass
            return s.replace("\\n", "\n")
        return str(item)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_url": ("STRING", {
                    "multiline": False, 
                    "default": "https://api.openai.com/v1",
                }),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": "", 
                    "placeholder": "sk-..."
                }),
                "model_name": ("STRING", {
                    "multiline": False, 
                    "default": "gemini-2.0-flash-exp",
                }),
                "product_type": ("STRING", {
                    "multiline": False,
                    "default": "摩托车头盔",
                }),
                "selling_points": ("STRING", {
                    "multiline": True,
                    "default": "碳纤维材质、轻量化设计、风噪降低",
                }),
                "design_style": (
                    [
                        "专业机能风",
                        "硬核竞技风",
                        "工业机械风",
                        "都市通勤风",
                        "户外冒险风",
                        "高端质感风",
                        "安全守护风",
                        "改装美学风",
                        "参数对比风",
                    ],
                    {"default": "专业机能风"}
                ),
                "scene_mode": (
                    [
                        "场景融合（产品+使用情境）",
                        "纯产品棚拍（干净背景）",
                        "骑行动态场景（人车合一）",
                    ],
                    {"default": "场景融合（产品+使用情境）"}
                ),
                "aspect_ratio": (
                    [
                        "1:1 正方形 (800x800)",
                        "3:4 竖版 (600x800)",
                    ],
                    {"default": "1:1 正方形 (800x800)"}
                ),
                "price_display": (
                    [
                        "大促价格块 (¥XX + 划线原价)",
                        "角标促销价 (左下圆角框)",
                        "双价对比 (国补价 vs 原价)",
                        "不显示价格",
                    ],
                    {"default": "大促价格块 (¥XX + 划线原价)"}
                ),
                "price_value": ("STRING", {
                    "multiline": False,
                    "default": "¥299",
                    "placeholder": "¥299 或 $49.99"
                }),
                "original_price": ("STRING", {
                    "multiline": False,
                    "default": "¥599",
                    "placeholder": "划线原价 ¥599"
                }),
                "promo_type": (
                    [
                        "TOP排名徽章 (热销第1名)",
                        "限时折扣标签 (限时X折/立减XX)",
                        "买赠活动框 (买2赠1)",
                        "新品首发标签",
                        "官方正品徽章",
                        "认证标签 (ECE/DOT/SNELL)",
                        "无促销标签",
                    ],
                    {"default": "TOP排名徽章 (热销第1名)"}
                ),
                "trust_bar": ("STRING", {
                    "multiline": False,
                    "default": "顺丰包邮|三年质保|7天无理由",
                    "placeholder": "信任横条内容，用|分隔"
                }),
                "output_language": (
                    [
                        "中文 (Chinese)",
                        "English",
                    ],
                    {"default": "中文 (Chinese)"}
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "prompt_count": ("INT", {"default": 3, "min": 1, "max": 10, "forceInput": False})
            },
            "optional": {
                "product_image": ("IMAGE",),
                "product_image_2": ("IMAGE",),
                "product_image_3": ("IMAGE",),
                "product_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompts_list", "debug_info")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate_main_image_prompts"
    CATEGORY = "🎨 WL-MainImageDesign"

    def tensor_to_base64(self, image, index=0):
        if image is None:
            return None
        img_tensor = image
        try:
            if hasattr(image, "shape") and len(image.shape) == 4:
                img_tensor = image[index]
        except Exception:
            img_tensor = image
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        max_size = 1024
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def collect_base64_images(self, images, max_images=6):
        base64_images = []
        if images is None:
            return base64_images
        for img in images:
            if img is None:
                continue
            try:
                if hasattr(img, "shape") and len(img.shape) == 4:
                    batch = int(img.shape[0])
                    for bi in range(batch):
                        if len(base64_images) >= max_images:
                            return base64_images
                        base64_images.append(self.tensor_to_base64(img, bi))
                else:
                    if len(base64_images) >= max_images:
                        return base64_images
                    base64_images.append(self.tensor_to_base64(img, 0))
            except Exception:
                if len(base64_images) >= max_images:
                    return base64_images
                base64_images.append(self.tensor_to_base64(img, 0))
        return [b for b in base64_images if b]

    def call_llm_vision(self, api_url, api_key, model, system_prompt, user_prompt, base64_images=None, seed=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ComfyUI-WL-MainImageDesign/2.1"
        }
        url = api_url.rstrip('/')
        if url.endswith('/chat'):
            url = f"{url}/completions"
        elif not url.endswith('/chat/completions'):
            url = f"{url}/chat/completions"
        content_list = [{"type": "text", "text": user_prompt}]
        if base64_images:
            if isinstance(base64_images, str):
                base64_images = [base64_images]
            for base64_image in base64_images:
                if not base64_image:
                    continue
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": base64_image}
                })
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_list}
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.75,
            "stream": False
        }
        if seed is not None and seed > 0:
            payload["seed"] = seed
        try:
            print(f"🔗 Calling API: {url}")
            print(f"🎨 Model: {model}")
            ssl_context = ssl._create_unverified_context()
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=180, context=ssl_context) as response:
                result = json.loads(response.read().decode('utf-8'))
                return {"success": True, "content": result['choices'][0]['message']['content']}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            error_msg = f"HTTP Error {e.code}: {err_body}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        except urllib.error.URLError as e:
            error_msg = f"URL Error: {str(e)}\nAPI URL: {url}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Error: {str(e)}\nAPI URL: {url}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}

    def generate_main_image_prompts(self, api_url, api_key, model_name, product_type, selling_points, 
                                     design_style, scene_mode, aspect_ratio, price_display, price_value, original_price,
                                     promo_type, trust_bar, output_language, seed, prompt_count, 
                                     product_image=None, product_image_2=None, product_image_3=None, product_image_4=None):
        
        base64_images = self.collect_base64_images(
            [product_image, product_image_2, product_image_3, product_image_4],
            max_images=6
        )
        system_instruction = _load_system_prompt()
            
        if output_language == "中文 (Chinese)":
            lang_instruction = "请使用中文生成所有提示词内容。模块标题使用：主标题/副标题/价格促销区/排版蓝图/视觉与光影/渲染品质"
            anti_translate = "画面所有文字必须为中文，禁止出现任何英文字母或乱码，文字渲染清晰锐利。"
        else:
            lang_instruction = "Generate all prompt content in English. Use module headers: Main Headline/Sub Headline/Price & Promo Zone/Layout Blueprint/Visual & Lighting/Render Quality"
            anti_translate = "All text must be in English only. No Chinese characters. Text rendering must be crystal clear."

        if "1:1" in aspect_ratio:
            ratio_instruction = """1:1正方形(800x800)排版：
- 顶部区(0-22%)：主标题42pt等效大字，居中或左对齐
- 上中区(18-38%)：副标题/3-5个卖点图标
- 中心区(25-75%)：产品主体占45-60%，悬浮效果
- 下中区(65-82%)：规格参数/认证标签
- 底部区(78-100%)：价格+促销标签+信任横条"""
            ratio_code = "1:1"
        else:
            ratio_instruction = """3:4竖版(600x800)排版：
- 顶部区(0-18%)：主标题42pt等效大字
- 上部区(15-32%)：副标题/促销小字
- 中心区(28-72%)：产品主体占50-65%，更多垂直空间
- 下部区(68-85%)：详细规格/认证
- 底部区(82-100%)：价格+CTA+信任元素"""
            ratio_code = "3:4"

        scene_instructions = {
            "场景融合（产品+使用情境）": """【场景融合模式 - 必须遵守】
必须将产品融入真实使用情境，创造有故事感的画面：
- 摩托车头盔：骑士佩戴头盔的侧面/背影，背景是公路/赛道/城市街道，有速度感和氛围光
- 骑行手套：双手握住摩托车把手的特写，能看到手套细节和仪表盘
- 骑行服：模特穿着骑行服的帅气站姿，背景是摩托车或车库环境
- 护具：骑士穿戴护具骑行的动态场景，突出防护感
- 蓝牙耳机：骑士头盔内佩戴耳机通话/听音乐的场景
产品必须是画面主体焦点，但要有环境氛围烘托，创造"向往感"和"代入感"。
禁止：纯白底、纯产品摆放、无场景的证件照式构图。""",
            
            "纯产品棚拍（干净背景）": """【纯产品棚拍模式】
专业摄影棚风格，干净背景但不单调：
- 背景：纯色渐变或微妙纹理（非纯白），可有光效点缀
- 产品：45°经典角度，悬浮展示，专业三点布光
- 细节：材质高光、反射、阴影都要精致
- 可添加：微妙的光晕、粒子、几何线条等设计元素
禁止：纯白底无设计、平铺俯拍、无光影层次。""",
            
            "骑行动态场景（人车合一）": """【骑行动态场景模式 - 必须遵守】
必须创造有冲击力的骑行动态画面：
- 场景：公路骑行、赛道压弯、越野穿越、城市穿梭
- 动态：速度感模糊、尘土飞扬、发丝/衣角飘动
- 氛围：黄金时段逆光、夜间霓虹、雨天反光等戏剧化光线
- 产品：作为骑士装备的一部分自然融入画面
产品必须清晰可辨识，但整体画面要有电影感和冲击力。
禁止：静态摆放、无人物、无场景。"""
        }
        scene_instruction = scene_instructions.get(scene_mode, scene_instructions["场景融合（产品+使用情境）"])

        style_details = {
            "专业机能风": """【专业机能风】
视觉核心：专业、可靠、功能导向

背景与场景融合：
- 主场景：专业车库/维修间环境，或骑士正在检查装备的场景
- 色调：深灰(#2d2d2d)或冷白(#f5f5f5)为主，搭配金属质感元素
- 环境细节：可见工具墙、摩托车局部、专业设备暗示
- 光效：顶部偏前45°主光模拟车库顶灯，侧面补光

产品呈现方式：
- 若为头盔：骑士单手托起头盔检视的画面，或头盔放置在摩托车油箱上
- 若为手套：双手正在穿戴调整的特写
- 若为护具：骑士正在穿戴/调整护具的动作瞬间
- 产品必须是视觉焦点，但要有使用情境

光影氛围：整体明亮但不刺眼，产品表面可见精细材质纹理，微弱边缘光勾勒轮廓
氛围关键词：专业、可靠、功能导向、工程感、理性克制""",
            
            "硬核竞技风": """【硬核竞技风】
视觉核心：速度、力量、极限性能

背景与场景融合：
- 主场景：赛道压弯瞬间、维修区准备出发、或高速骑行的动态画面
- 色调：深黑(#0a0a0a)为底，红/橙色作为激情点缀
- 动态元素：速度线、轮胎烟雾、赛道标识、计时器元素
- 纹理：碳纤维纹理、赛道沥青质感、刹车盘热效应

产品呈现方式：
- 若为头盔：赛车手戴着头盔准备出发的紧张瞬间，面罩反射赛道
- 若为手套：紧握油门/刹车的特写，手套与车把的接触细节
- 若为皮衣：赛车手压弯的侧面动态，皮衣与赛道的视觉张力
- 可有轻微动态模糊增加速度感，但产品细节必须清晰

光影氛围：侧面硬光强烈明暗对比，锐利边缘高光，红/橙色点光源，镜头光晕
氛围关键词：速度、力量、激进、热血、极限性能""",
            
            "工业机械风": """【工业机械风】
视觉核心：结实、耐用、硬核工程

背景与场景融合：
- 主场景：重型机械车间、改装车库、或摩托车引擎特写旁
- 色调：深灰金属(#3d3d3d)、氧化铁色、工业蓝
- 质感元素：拉丝金属墙面、六角螺栓、焊接痕迹、链条齿轮
- 环境细节：工具架、千斤顶、机油桶等工业道具虚化

产品呈现方式：
- 若为头盔：放置在金属工作台上，旁边有扳手和零件
- 若为护具：与摩托车金属部件形成质感对比
- 若为配件：近距离展示与车辆的安装/连接细节
- 强调产品的结构感、材质硬度、工程精度

光影氛围：冷色调顶光突出金属质感，金属表面环境反射，硬朗投影增加厚重感
氛围关键词：结实、耐用、可靠、硬度、工程感""",
            
            "都市通勤风": """【都市通勤风】
视觉核心：真实、亲近、日常实用

背景与场景融合：
- 主场景：城市街道、咖啡店门口、写字楼停车场、地铁站旁
- 色调：自然城市色彩，灰调为主带暖色生活气息
- 环境元素：城市建筑虚化、路灯、斑马线、咖啡杯等生活道具
- 时间感：清晨通勤、傍晚下班、周末骑行的真实场景

产品呈现方式：
- 若为头盔：骑士一手提着头盔走向摩托车，或头盔挂在车把上
- 若为骑行服：都市白领穿着骑行服走出咖啡店的场景
- 若为手套：骑士在红灯前摘下手套看手机的日常瞬间
- 产品融入真实生活场景，有代入感和亲切感

光影氛围：自然日光或城市夜灯柔和漫射，浅景深效果，温暖亲切的整体调性
氛围关键词：真实、亲近、实用、现代、生活化""",
            
            "户外冒险风": """【户外冒险风】
视觉核心：耐用、探索、自由精神

背景与场景融合：
- 主场景：山路骑行、荒野休息、越野穿越、海边公路
- 色调：大地色系(土色#8B7355、军绿#4A5D23、天空蓝#2C3E50)
- 环境元素：远山轮廓、尘土飞扬、帐篷露营、地图指南针
- 氛围：开阔天地、自由探索、长途旅行的浪漫

产品呈现方式：
- 若为头盔：ADV骑士停在山顶俯瞰风景，头盔放在油箱上
- 若为骑行服：骑士穿着全套装备站在摩托车旁看地图
- 若为防水包：绑在摩托车上，背景是壮丽自然风光
- 产品与探索场景融合，传达"陪你走天涯"的可靠感

光影氛围：户外自然光，黄金时段暖调优先，大面积天光漫射创造开阔感
氛围关键词：耐用、探索、自由、可靠、长途、冒险精神""",
            
            "高端质感风": """【高端质感风】
视觉核心：高端、极简、材质至上

背景与场景融合：
- 主场景：高端展厅、私人车库、或极简摄影棚
- 色调：深色渐变(#1a1a1a→#2d2d2d)或深蓝黑，大面积留白
- 质感元素：丝绒质感背景、皮革暗示、大理石台面
- 环境：可见高端摩托车局部（如杜卡迪、宝马）作为身份暗示

产品呈现方式：
- 若为头盔：精致展示架上的艺术品般呈现，可见皮革内衬细节
- 若为皮手套：展示精湛缝线工艺和皮革纹理的特写
- 若为限量款：强调独特设计元素和品牌标识
- 产品比例大，细节丰富，如奢侈品广告般精致

光影氛围：精准控制的柔光只照亮关键区域，产品表面精细高光，整体安静克制高级
氛围关键词：高端、极简、材质、品牌、自信、奢华""",
            
            "安全守护风": """【安全守护风】
视觉核心：安全、防护、可信赖

背景与场景融合：
- 主场景：安全测试场景、家人送别骑行、或专业培训环境
- 色调：稳重蓝(#1e3a5f)或沉稳灰(#4a4a4a)，传达信任感
- 元素：安全认证标识、防护测试画面、家人关怀场景
- 氛围：可信赖、安心、守护感

产品呈现方式：
- 若为头盔：展示头盔剖面结构，突出EPS缓冲层等防护细节
- 若为护具：骑士穿戴完整护具，家人在旁送别的温馨场景
- 若为反光装备：夜间骑行场景，反光材料在车灯下闪亮
- 通过视觉引导突出关键防护区域和安全特性

光影氛围：均匀柔和主光无强烈对比，充足补光展示全貌，明亮清晰无阴暗角落
氛围关键词：安全、防护、可靠、信任、安心、守护""",
            
            "改装美学风": """【改装美学风】
视觉核心：个性、张扬、视觉表达

背景与场景融合：
- 主场景：改装车间、潮流街拍、机车聚会、霓虹夜景
- 色调：大胆对比色，霓虹粉/蓝/绿，或复古暖调
- 元素：涂鸦墙面、霓虹灯牌、改装工具、潮流贴纸
- 氛围：街头文化、个性表达、亚文化归属感

产品呈现方式：
- 若为头盔：骑士手持彩绘头盔的潮流街拍
- 若为改装配件：安装在改装车上的炫酷展示
- 若为个性装备：与改装摩托车的整体风格搭配展示
- 强调设计细节、独特涂装、个性化元素

光影氛围：戏剧化打光制造视觉焦点，霓虹色彩光源，强边缘光突出造型
氛围关键词：个性、张扬、设计感、态度、视觉表达""",
            
            "参数对比风": """【参数对比风】
视觉核心：数据、对比、理性说服

背景与场景融合：
- 主场景：干净的对比展示环境，或实验室测试场景
- 色调：中性白(#ffffff)或浅灰(#f5f5f5)，信息区分明
- 布局：产品主体+参数信息区明确分离
- 元素：大数字KPI、图形化数据、对比图表、测试数据

产品呈现方式：
- 若为头盔：重量对比（与竞品）、风洞测试数据展示
- 若为护具：防护等级对比、材料强度数据
- 若为配件：安装便捷度、兼容性参数矩阵
- 产品占画面40-50%，留出充足的参数展示区域

光影氛围：均匀明亮的产品光无干扰阴影，高亮度高清晰度的信息展示
氛围关键词：数据、对比、性能、理性、高效转化""",
        }
        style_instruction = style_details.get(design_style, "根据产品特性选择合适的视觉风格。")

        price_instructions = {
            "大促价格块 (¥XX + 划线原价)": f"""价格展示：大促价格块
- 促销价：{price_value}，红/橙色，42pt等效大字
- 原价：{original_price}，灰色划线，18pt
- 节省提示：立省XX元（绿色小字）
- 位置：底部中央或左下角""",
            "角标促销价 (左下圆角框)": f"""价格展示：角标促销价
- 价格：{price_value}，圆角矩形背景框内
- 位置：左下角
- 折扣角标：附着在价格框上""",
            "双价对比 (国补价 vs 原价)": f"""价格展示：双价对比
- 国补价/到手价：{price_value}（大字，强调）
- 原价/市场价：{original_price}（小字对比）
- 并排展示，突出优惠力度""",
            "不显示价格": "不显示任何价格信息，专注产品价值传达。",
        }
        price_instruction = price_instructions.get(price_display, "不显示价格。")

        promo_instructions = {
            "TOP排名徽章 (热销第1名)": """促销标签：TOP排名徽章
- 样式：金色/红色丝带或盾形徽章
- 文案："TOP1" / "热销第1名" / "销量冠军"
- 位置：右上角或产品旁边
- 效果：微妙金属质感，投影""",
            "限时折扣标签 (限时X折/立减XX)": """促销标签：限时折扣
- 样式：斜角飘带或醒目徽章
- 文案："限时87折" / "立减60元" / "限时特惠"
- 颜色：红/橙渐变，白色文字
- 位置：右侧竖排或右上角""",
            "买赠活动框 (买2赠1)": """促销标签：买赠活动
- 样式：圆角矩形横幅
- 文案："买2赠1" / "买3增2" / "加购送XX"
- 颜色：促销红或品牌色
- 位置：产品上方或下方""",
            "新品首发标签": """促销标签：新品首发
- 样式：简洁标签或角标
- 文案："新品" / "首发" / "NEW"
- 突出新鲜感和独特性""",
            "官方正品徽章": """促销标签：官方正品
- 样式：认证徽章风格
- 文案："官方正品" / "品牌授权" / "官方旗舰"
- 增强信任感""",
            "认证标签 (ECE/DOT/SNELL)": """促销标签：专业认证
- 样式：认证机构官方标识风格
- 类型：ECE/DOT/SNELL/3C等安全认证图标
- 位置：产品旁或规格区
- 增强专业可信度""",
            "无促销标签": "不添加促销标签，保持画面干净简洁，专注产品本身。",
        }
        promo_instruction = promo_instructions.get(promo_type, "不添加促销标签。")

        trust_instruction = f"""信任横条：
- 位置：底部10%区域
- 布局：水平均匀分布
- 内容：{trust_bar}
- 样式：小图标+文字，微妙背景条"""

        try:
            target_count = int(prompt_count)
        except Exception:
            target_count = 3
        target_count = max(1, min(10, target_count))

        if target_count == 1:
            variant_plan = """生成1个变体 - 主图首选(Hero Prime)：
- 最具视觉冲击力的场景融合画面
- 产品与使用情境完美结合
- 最强视觉冲击主标题"""
        elif target_count == 2:
            variant_plan = """生成2个变体：
V1 - 主图首选(Hero Prime)：场景融合+使用情境，最强视觉冲击
V2 - 卖点特写(Feature Spotlight)：关键差异化特写，技术细节展示"""
        elif target_count == 3:
            variant_plan = """生成3个变体：
V1 - 主图首选(Hero Prime)：场景融合+使用情境，最强视觉冲击
V2 - 卖点特写(Feature Spotlight)：关键特性近距特写
V3 - 信任背书(Trust Builder)：TOP徽章+认证+安全感场景"""
        else:
            variant_plan = f"""生成{target_count}个变体：
V1 - 主图首选(Hero Prime)：场景融合+使用情境，最强视觉冲击
V2 - 卖点特写(Feature Spotlight)：关键特性近距特写
V3 - 信任背书(Trust Builder)：TOP徽章+认证突出
V4 - 促销主打(Price Focus)：大促价格为视觉中心
V5 - 场景氛围(Lifestyle Context)：骑行生活方式场景
V6+ - 角度变化(Angle Variation)：其他使用场景/角度展示"""

        base_user_req = f"""
请为以下产品生成 {{COUNT}} 个高视觉冲击力电商主图提示词：

═══════════════════════════════════════════
【产品信息】
═══════════════════════════════════════════
- 产品类型：{product_type}
- 核心卖点：{selling_points}

═══════════════════════════════════════════
【视觉规范 - 必须严格遵守】
═══════════════════════════════════════════
【设计风格】
{design_style}
{style_instruction}

【场景模式 - 关键要求】
{scene_mode}
{scene_instruction}

【人体完整性约束 - 必须遵守】⚠️
涉及人物/骑士/模特时，严格保证解剖正确：
- 手部：5根手指完整，关节自然弯曲，无多余或缺失
- 姿态：符合物理规律，重心平衡，穿戴逻辑正确
- 建议：优先侧面/背影/剪影，或手部被产品遮挡
- 若无法保证正确，宁可选择产品单独展示+环境暗示

【画幅比例】{ratio_code}
{ratio_instruction}

【价格展示】
{price_instruction}

【促销标签】
{promo_instruction}

【信任横条】
{trust_instruction}

【语言要求】
{lang_instruction}

═══════════════════════════════════════════
【变体规划】
═══════════════════════════════════════════
{variant_plan}

═══════════════════════════════════════════
【参考图信息】
═══════════════════════════════════════════
参考图数量：{len(base64_images)}
规则：参考图仅用于锁定产品外观（形状/颜色/材质/logo/细节），必须抠出主体、忽略原图背景，为每个变体重建新的视觉场景。

═══════════════════════════════════════════
【输出格式要求 - 严格遵守】
═══════════════════════════════════════════
1. 输出纯JSON字符串列表 List[str]，长度必须等于 {{COUNT}}
2. 不要输出Markdown、代码块、任何解释文字
3. 每个变体是一个完整提示词字符串

【每个变体必须包含6个模块】
1) 主标题: "..." (52-60pt超大加粗字，4-12字，绝对视觉焦点，第一眼必看)
2) 副标题: "..." (18-24pt，3-5个卖点/规格，管道符或图标分隔)
3) 价格促销区: 按上方规范详细描述价格展示方式、标签类型位置、信任条
4) 排版蓝图: 严格按{ratio_code}比例规范，描述产品位置(占比45-60%)、文字区划分、标签位置
5) 视觉与光影: 【重点】必须详细描述场景融合方式、人物/手部交互（如有）、背景环境细节、主光/补光/轮廓光设置、氛围营造、材质渲染
6) 渲染品质: 8K超清商业摄影级、材质细节清晰可见、文字52-60pt超粗锐利无锯齿、专业调色、视觉冲击评分9/10

【变体定位标签】
每个变体开头标注定位：
中文："变体定位：主图首选" / "变体定位：卖点特写" / "变体定位：信任背书" / "变体定位：促销主打" / "变体定位：场景氛围"
英文："Variant Role: Hero Prime" / "Variant Role: Feature Spotlight" 等

【防乱码结尾】
每个提示词末尾追加：{anti_translate}

═══════════════════════════════════════════
【质量标准 - 不可妥协】
═══════════════════════════════════════════
□ 主标题52-60pt超大加粗，第一眼绝对醒目不可忽视
□ 产品必须融入使用场景，禁止纯白底平铺
□ 场景要有故事感和代入感
□ ⚠️人体完整性：手指5根完整、关节自然、无畸形断裂
□ ⚠️人物姿态合理：符合物理规律、穿戴逻辑正确
□ 光影设置营造立体感和氛围感
□ 背景风格严格匹配所选风格规范
□ 价格促销清晰但不抢产品风头
□ 排版严格适配{ratio_code}比例
□ 视觉冲击力评分目标：9/10
□ 对标苹果/戴森级商业摄影品质
"""

        collected = []
        raw_responses = []
        attempts = []
        max_per_call = 5
        max_calls = 5
        call_idx = 0
        last_error = None

        while len(collected) < target_count and call_idx < max_calls:
            remaining = target_count - len(collected)
            request_n = remaining if remaining <= max_per_call else max_per_call
            user_req = base_user_req.replace("{COUNT}", str(request_n))
            if len(collected) > 0:
                user_req += f"\n\n【续写要求】这是续写生成，请生成新的{request_n}个变体，使用不同的场景和角度，不要重复之前的构图。"
            print(f"🎨 Generating {request_n} premium main image variants... ({len(collected)}/{target_count})")
            result = self.call_llm_vision(api_url, api_key, model_name, system_instruction, user_req, base64_images if base64_images else None, seed)
            call_idx += 1
            if not result["success"]:
                last_error = result.get("error")
                attempts.append({"call": call_idx, "requested": request_n, "parsed": 0, "accepted": 0, "method": "api_error", "error": last_error})
                continue
            response = result.get("content", "")
            raw_responses.append(response)
            batch_prompts, method = self._parse_response_to_prompts_list(response, request_n)
            accepted = []
            rejected = 0
            for p in batch_prompts:
                if self._is_prompt_structurally_complete(p):
                    accepted.append(p)
                else:
                    rejected += 1
            if not accepted and batch_prompts:
                accepted = batch_prompts
            if len(accepted) > request_n:
                accepted = accepted[:request_n]
            collected.extend(accepted)
            attempts.append({"call": call_idx, "requested": request_n, "parsed": len(batch_prompts), "accepted": len(accepted), "rejected": rejected, "method": method, "response_chars": len(response) if isinstance(response, str) else None})

        if len(collected) > target_count:
            collected = collected[:target_count]
        if len(collected) < target_count:
            missing = target_count - len(collected)
            msg = "[GENERATION_FAILED] Unable to generate enough variants."
            if last_error:
                msg = f"[GENERATION_FAILED] {last_error}"
            collected.extend([msg] * missing)

        debug_payload = {
            "plugin": "Comfyui-WL-MainImageDesign v2.1",
            "input_summary": {"product": product_type, "style": design_style, "scene_mode": scene_mode, "ratio": ratio_code, "target_count": target_count},
            "reference_image_count": len(base64_images),
            "attempts": attempts,
        }
        if raw_responses:
            debug_payload["raw_response_preview"] = raw_responses[-1][:2000] if len(raw_responses[-1]) > 2000 else raw_responses[-1]
        if last_error:
            debug_payload["error"] = last_error
        return (collected, json.dumps(debug_payload, ensure_ascii=False, indent=2))


class WLPromptBatchConverter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts_list": ("STRING", {"forceInput": True, "multiline": True, "placeholder": "输入提示词列表"}),
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 20, "tooltip": "每批处理数量"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("batch_output",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "convert_list_to_batches"
    CATEGORY = "🎨 WL-MainImageDesign"

    def convert_list_to_batches(self, prompts_list, batch_size):
        if not prompts_list or not prompts_list.strip():
            return ("Error: No prompts list provided",)
        try:
            if prompts_list.strip().startswith('['):
                try:
                    prompts = json.loads(prompts_list)
                    prompt_items = prompts if isinstance(prompts, list) else [str(prompts)]
                except json.JSONDecodeError:
                    prompt_items = [line.strip() for line in prompts_list.split('\n') if line.strip()]
            else:
                prompt_items = [line.strip() for line in prompts_list.split('\n') if line.strip()]
            if not prompt_items:
                return ("Error: No valid prompts found",)
            batches = []
            for i in range(0, len(prompt_items), batch_size):
                batch = prompt_items[i:i + batch_size]
                batches.append('\n'.join(batch))
            result = ""
            for i, batch in enumerate(batches):
                if i > 0:
                    result += "\n---\n"
                result += batch
            return (result,)
        except Exception as e:
            return (f"Error: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "WLMainImageGenerator": WLMainImageGenerator,
    "WLPromptBatchConverter": WLPromptBatchConverter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WLMainImageGenerator": "🎨 WL Main Image Designer",
    "WLPromptBatchConverter": "🔄 WL Prompt Batch Converter"
}
