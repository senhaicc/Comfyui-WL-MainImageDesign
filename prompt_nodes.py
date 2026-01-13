"""
Comfyui-WL-MainImageDesign
Elite E-commerce Main Image Prompt Generator
Version: 2.0
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

        # 尝试匹配 JSON 对象模式
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

        # 尝试匹配变体定位标记
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

        # 尝试用分隔符分割
        if "\n---\n" in s:
            parts = [p.strip() for p in s.split("\n---\n")]
            parts = [p for p in parts if p]
            if parts:
                return parts

        # 尝试用多个空行分割
        parts = [p.strip() for p in re.split(r"\n\s*\n\s*\n+", s)]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts

        return [s]

    def _clean_code_fences(self, response_text):
        """清理代码围栏标记"""
        cleaned = (response_text or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _parse_response_to_prompts_list(self, response_text, expected_count):
        """解析响应为提示词列表"""
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
        """移除变体定位头部（可选）"""
        if not isinstance(prompt_text, str):
            return prompt_text
        return prompt_text.strip()

    def _is_prompt_structurally_complete(self, prompt_text):
        """检查提示词是否结构完整"""
        if not isinstance(prompt_text, str):
            return False
        s = prompt_text.strip()
        if not s:
            return False

        has_main = ("主标题" in s) or ("Main Headline" in s)
        has_visual = ("视觉" in s) or ("Visual" in s) or ("光影" in s) or ("Lighting" in s) or ("背景" in s) or ("Background" in s)

        return has_main and has_visual

    def enforce_prompt_count(self, prompts_list, prompt_count, raw_response):
        """确保提示词数量符合要求"""
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
        """提取提示词文本"""
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
                    "default": "蓝牙耳机",
                }),
                "selling_points": ("STRING", {
                    "multiline": True,
                    "default": "降噪、长续航、佩戴舒适",
                }),
                "design_style": (
                    [
                        "科技深邃 (Tech Deep)",
                        "温润米白 (Warm Cream)",
                        "高级金棕 (Premium Bronze)",
                        "清新天蓝 (Fresh Sky)",
                        "氛围场景 (Lifestyle Scene)",
                        "硬核深棕 (Hardcore Brown)",
                        "极简纯净 (Minimal Pure)",
                        "未来科幻 (Sci-Fi Future)",
                    ],
                    {"default": "科技深邃 (Tech Deep)"}
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
                        "认证标签 (ACS/SGS/Hi-Res)",
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
        """将 ComfyUI Tensor 图片转换为 Base64"""
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
        """收集多张图片的 Base64 编码"""
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
        """调用 LLM Vision API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ComfyUI-WL-MainImageDesign/2.0"
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
                    "image_url": {
                        "url": base64_image
                    }
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
                                     design_style, aspect_ratio, price_display, price_value, original_price,
                                     promo_type, trust_bar, output_language, seed, prompt_count, 
                                     product_image=None, product_image_2=None, product_image_3=None, product_image_4=None):
        """生成高视觉冲击力主图提示词"""
        
        base64_images = self.collect_base64_images(
            [product_image, product_image_2, product_image_3, product_image_4],
            max_images=6
        )

        # 加载 system prompt
        system_instruction = _load_system_prompt()
            
        # 语言处理
        if output_language == "中文 (Chinese)":
            lang_instruction = "请使用中文生成所有提示词内容。模块标题使用：主标题/副标题/价格促销区/排版蓝图/视觉与光影/渲染品质"
            anti_translate = "画面所有文字必须为中文，禁止出现任何英文字母或乱码，文字渲染清晰锐利。"
        else:
            lang_instruction = "Generate all prompt content in English. Use module headers: Main Headline/Sub Headline/Price & Promo Zone/Layout Blueprint/Visual & Lighting/Render Quality"
            anti_translate = "All text must be in English only. No Chinese characters. Text rendering must be crystal clear."

        # 比例处理
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

        # 风格详细描述
        style_details = {
            "科技深邃 (Tech Deep)": """背景：深蓝渐变(#1a1a2e→#16213e→#0f3460)
光效：蓝紫色边缘光晕，几何科技线条
粒子：微妙的科技网格或电路图案
氛围：未来感、专业感、高端科技""",
            
            "温润米白 (Warm Cream)": """背景：奶油渐变(#fefefe→#f5f0e8→#e8e0d5)
光效：左上角暖阳光，柔和漫射
纹理：微妙亚麻或纸张质感
氛围：温馨、家居感、品质生活""",
            
            "高级金棕 (Premium Bronze)": """背景：金棕渐变(#d4a574→#c9a067)，微妙
光效：金色边缘高光，奢华感
纹理：拉丝金属或大理石暗示
氛围：高端、奢华、品质卓越""",
            
            "清新天蓝 (Fresh Sky)": """背景：浅蓝渐变(#e8f4f8→#d4ecf7→#b8dced)
光效：明亮均匀，清爽感
元素：柔和云朵感，清新空气
氛围：夏日、清凉、轻盈透气""",
            
            "氛围场景 (Lifestyle Scene)": """场景：咖啡杯/书桌/浴室/卧室融入
产品：自然放置但明确为主角
景深：浅景深，产品锐利背景柔和
光线：自然窗光或温暖室内光""",
            
            "硬核深棕 (Hardcore Brown)": """背景：深棕渐变(#3d2b1f→#5c4033)
纹理：皮革、大地、粗犷表面
光效：橙琥珀色点光源
氛围：运动、户外、硬核男性""",
            
            "极简纯净 (Minimal Pure)": """背景：纯白#ffffff或浅灰#f8f8f8
效果：仅产品阴影，无渐变无特效
风格：亚马逊合规风格，参数为主
氛围：干净、专业、规格导向""",
            
            "未来科幻 (Sci-Fi Future)": """背景：深空色彩，星云暗示
纹理：行星表面、外星地形
光效：霓虹光晕，全息暗示
氛围：前沿科技、游戏、硬核""",
        }
        style_instruction = style_details.get(design_style, "根据产品特性选择合适的视觉风格。")

        # 价格展示处理
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

        # 促销标签处理
        promo_instructions = {
            "TOP排名徽章 (热销第1名)": """促销标签：TOP排名徽章
- 样式：金色/红色丝带或盾形徽章
- 文案："TOP1" / "热销第1名" / "降噪效果:第1名"
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
            
            "认证标签 (ACS/SGS/Hi-Res)": """促销标签：专业认证
- 样式：认证机构官方标识风格
- 类型：ACS/SGS/Hi-Res/CE等认证图标
- 位置：产品旁或规格区
- 增强专业可信度""",
            
            "无促销标签": "不添加促销标签，保持画面干净简洁，专注产品本身。",
        }
        promo_instruction = promo_instructions.get(promo_type, "不添加促销标签。")

        # 信任横条处理
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

        # 变体规划
        if target_count == 1:
            variant_plan = """生成1个变体 - 主图首选(Hero Prime)：
- 最佳45°角度展示产品立体感
- 最强视觉冲击主标题
- 最优价格促销布局组合"""
        elif target_count == 2:
            variant_plan = """生成2个变体：
V1 - 主图首选(Hero Prime)：45°最佳角度，主价值主张，完整促销布局
V2 - 卖点特写(Feature Spotlight)：关键差异化特写，技术细节展示"""
        elif target_count == 3:
            variant_plan = """生成3个变体：
V1 - 主图首选(Hero Prime)：45°最佳角度，主价值主张
V2 - 卖点特写(Feature Spotlight)：关键特性近距特写
V3 - 信任背书(Trust Builder)：TOP徽章+认证突出展示"""
        else:
            variant_plan = f"""生成{target_count}个变体：
V1 - 主图首选(Hero Prime)：45°最佳角度，主价值主张
V2 - 卖点特写(Feature Spotlight)：关键特性近距特写
V3 - 信任背书(Trust Builder)：TOP徽章+认证突出
V4 - 促销主打(Price Focus)：大促价格为视觉中心
V5 - 场景氛围(Lifestyle Context)：使用场景融入
V6+ - 角度变化(Angle Variation)：其他角度展示"""

        # 构建完整请求
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
规则：严格以参考图为产品外观依据，隔离产品主体后重建背景，所有变体保持外观一致。

═══════════════════════════════════════════
【输出格式要求 - 严格遵守】
═══════════════════════════════════════════
1. 输出纯JSON字符串列表 List[str]，长度必须等于 {{COUNT}}
2. 不要输出Markdown、代码块、任何解释文字
3. 每个变体是一个完整提示词字符串

【每个变体必须包含6个模块】
1) 主标题: "..." (42pt等效大字，4-12字，视觉冲击力)
2) 副标题: "..." (18-24pt，3-5个卖点/规格，管道符或图标分隔)
3) 价格促销区: 按上方规范详细描述价格展示方式、标签类型位置、信任条
4) 排版蓝图: 严格按{ratio_code}比例规范，描述产品位置(占比45-60%)、文字区划分、标签位置
5) 视觉与光影: 按风格规范详述背景、主光/补光/轮廓光设置、产品悬浮效果(上浮15-30px+椭圆柔影)、材质渲染、配色
6) 渲染品质: 8K超清商业摄影级、材质细节清晰可见、文字42pt锐利无锯齿、专业调色、视觉冲击评分9/10

【变体定位标签】
每个变体开头标注定位：
中文："变体定位：主图首选" / "变体定位：卖点特写" / "变体定位：信任背书" / "变体定位：促销主打" / "变体定位：场景氛围"
英文："Variant Role: Hero Prime" / "Variant Role: Feature Spotlight" 等

【防乱码结尾】
每个提示词末尾追加：{anti_translate}

═══════════════════════════════════════════
【质量标准 - 不可妥协】
═══════════════════════════════════════════
□ 主标题42pt等效，醒目程度不可忽视
□ 产品悬浮效果专业，占比45-60%
□ 三点布光营造立体感和高级感
□ 背景风格严格匹配规范
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
                user_req += f"\n\n【续写要求】这是续写生成，请生成新的{request_n}个变体，不要重复之前的角度、标题与构图。"

            print(f"🎨 Generating {request_n} premium main image variants... ({len(collected)}/{target_count})")
            result = self.call_llm_vision(api_url, api_key, model_name, system_instruction, user_req, base64_images if base64_images else None, seed)
            call_idx += 1

            if not result["success"]:
                last_error = result.get("error")
                attempts.append({
                    "call": call_idx,
                    "requested": request_n,
                    "parsed": 0,
                    "accepted": 0,
                    "method": "api_error",
                    "error": last_error,
                })
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

            attempts.append({
                "call": call_idx,
                "requested": request_n,
                "parsed": len(batch_prompts),
                "accepted": len(accepted),
                "rejected": rejected,
                "method": method,
                "response_chars": len(response) if isinstance(response, str) else None,
            })

        if len(collected) > target_count:
            collected = collected[:target_count]

        if len(collected) < target_count:
            missing = target_count - len(collected)
            msg = "[GENERATION_FAILED] Unable to generate enough variants."
            if last_error:
                msg = f"[GENERATION_FAILED] {last_error}"
            collected.extend([msg] * missing)

        debug_payload = {
            "plugin": "Comfyui-WL-MainImageDesign v2.0",
            "input_summary": {
                "product": product_type,
                "style": design_style,
                "ratio": ratio_code,
                "target_count": target_count,
            },
            "reference_image_count": len(base64_images),
            "attempts": attempts,
        }
        if raw_responses:
            debug_payload["raw_response_preview"] = raw_responses[-1][:2000] if len(raw_responses[-1]) > 2000 else raw_responses[-1]
        if last_error:
            debug_payload["error"] = last_error

        return (collected, json.dumps(debug_payload, ensure_ascii=False, indent=2))


class WLPromptBatchConverter:
    """提示词列表转批次处理器"""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts_list": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "placeholder": "输入提示词列表"
                }),
                "batch_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "tooltip": "每批处理数量"
                }),
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
                    if isinstance(prompts, list):
                        prompt_items = prompts
                    else:
                        prompt_items = [str(prompts)]
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
