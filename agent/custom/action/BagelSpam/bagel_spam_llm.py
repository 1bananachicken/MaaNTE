"""bagel_spam_llm.py — LLM 文本生成（CustomRecognition）

截图 → 调 OpenAI 兼容 API → 生成标题+正文 → 存模块级变量
"""

import json
import base64
import io

import requests
from PIL import Image

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.define import RectType
from utils.logger import logger

# ---------------------------------------------------------------------------
# 模块级变量：存储 LLM 生成的标题和正文
# ---------------------------------------------------------------------------
_bagel_spam_llm_title = ""
_bagel_spam_llm_body = ""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _image_to_base64(image) -> str | None:  # image: numpy.ndarray, [H,W,3], BGR
    """numpy 图片（BGR）→ base64 PNG 字符串，失败返回 None"""
    try:
        # BGR → RGB（PIL.Image.fromarray 需要 RGB）
        img = Image.fromarray(image[..., ::-1])

        # 写入内存缓冲区（不写磁盘）
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # base64 编码 → UTF-8 字符串
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return b64

    except Exception as e:
        logger.error("image to base64 failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# 硬编码基础提示词（角色定位 + 基本规则 + 输出格式）
# ---------------------------------------------------------------------------
_BASE_PROMPT_PREFIX = (
    "你是一个正在游玩《异环》（Neverness to Everness / NTE）的资深玩家，准备在游戏内的「贝果」社区分享你的游戏瞬间。\n"
    "请仔细观察提供的游戏截图，并严格遵循以下要求生成帖子的标题和正文：\n\n"
    "1. 【抓取核心画面】忽略截图中的常规UI（如“发布帖子”、“请输入正文”、“0/40”等），将注意力完全集中在画面主体上（例如：抽卡出金的特效、特定角色、风景、战斗瞬间等）。\n"
    "2. 【符合玩家真实语境】以“玩家”的第一人称视角发帖，不要写成官方公告或硬核评测。使用真实的玩家社区口吻，例如：\n"
    "   - 如果是抽卡/发光特效，表现出激动、吸欧气或炫耀（如“出金了”、“这谁顶得住”、“终于欧了一次”）。\n"
    "   - 如果是角色，表现出喜爱、吐槽或发癫。\n"
    "   - 如果是风景，表现出赞叹（如“绝景”、“随便一截就是壁纸”）。\n"
    "3. 【禁止语C】除非额外说明，否则不要扮演截图中的游戏角色本人，你要做的是“评价”或“记录”他们。\n"
    "4. 【语言与字数】识别截图中主体的语言环境并保持一致。标题控制在 5~15 个字（要求吸睛）；正文控制在 1~3 句话（随性、口语化）。\n"
    "5. 【不编造】贴合截图实际内容，画面里没有什么就不要瞎编。\n"
)

# 中间插入用户自定义的风格提示词 (例如: "今天心情很差，语气暴躁一点" 或 "假装是个萌新提问")

_BASE_PROMPT_SUFFIX = (
    "请直接输出严格的 JSON 格式，不要包含任何 Markdown 标记、代码块符号（如 ```json ）或分析过程。必须以 '{' 开头，以 '}' 结尾。\n"
    '格式示例：{"title": "标题", "body": "正文"}'
)


def _extract_json(text: str) -> dict | None:
    """从混合内容中尝试提取 JSON 对象"""
    import re

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 { ... } 对象
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _call_llm(
    api_base: str,  # API 端点，如 "https://api.openai.com/v1"
    model: str,  # 模型名，如 "gpt-4o"
    api_key: str,  # API Key
    prompt: str,  # 提示词
    image_b64: str,  # 截图的 base64 字符串
) -> dict | None:
    """调用多模态 LLM（OpenAI 兼容格式），返回 {"title": "...", "body": "..."} 或 None"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }

    resp = None
    try:
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        if not content:
            logger.error("LLM returned empty content, raw: %s", resp.text[:500])
            return None
        result = _extract_json(content)
        if not result:
            logger.error("LLM returned no valid JSON, raw content: %s", content[:500])
            return None

        title = result.get("title", "").strip()
        body = result.get("body", "").strip()

        return {"title": title, "body": body}

    except requests.exceptions.Timeout:
        logger.error("LLM API timeout (60s)")
        return None
    except requests.exceptions.RequestException as e:
        logger.error("LLM API request failed: %s", e)
        return None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raw = resp.text[:500] if resp is not None else "N/A"
        logger.error("LLM API response parse failed: %s, raw: %s", e, raw)
        return None


# ---------------------------------------------------------------------------
# CustomRecognition
# ---------------------------------------------------------------------------


@AgentServer.custom_recognition("bagel_spam_llm_generate")
class BagelSpamLLMGenerate(CustomRecognition):
    """截图 → LLM 生成标题+正文 → 存入模块级变量。识别成功返回 dummy box + detail，失败返回 None"""

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        global _bagel_spam_llm_title, _bagel_spam_llm_body

        # ================================================================
        # 第一步：解析参数
        # ================================================================
        params = {}
        if argv.custom_recognition_param:
            try:
                params = json.loads(argv.custom_recognition_param)
            except json.JSONDecodeError as e:
                logger.error("failed to parse custom_recognition_param: %s", e)
                return None

        api_base = params.get("api_base", "https://api.openai.com/v1")
        model = params.get("model", "gpt-4o")
        api_key = params.get("api_key", "")
        style = params.get("prompt", "随意简短，带点整活或感叹的味道")
        prompt = _BASE_PROMPT_PREFIX + "风格要求：" + style + "\n" + _BASE_PROMPT_SUFFIX

        if not api_key:
            logger.error("LLM api_key is empty")
            return None

        # ================================================================
        # 第二步：截图 → base64
        # ================================================================
        # argv.image 是 MaaFramework 自动截的图，numpy.ndarray，BGR 格式
        image_b64 = _image_to_base64(argv.image)
        if not image_b64:
            return None

        logger.debug("screenshot taken, calling LLM...")

        # ================================================================
        # 第三步：调 LLM API
        # ================================================================
        result = _call_llm(api_base, model, api_key, prompt, image_b64)
        if not result:
            logger.error("LLM generation failed")
            return None

        if not result["title"] or not result["body"]:
            logger.error("LLM returned empty title or body: %s", result)
            return None

        # ================================================================
        # 第四步：存储结果 → 返回识别成功
        # ================================================================
        _bagel_spam_llm_title = result["title"]
        _bagel_spam_llm_body = result["body"]

        logger.info(
            "LLM generated: title=%s body=%s",
            _bagel_spam_llm_title,
            _bagel_spam_llm_body,
        )

        # box=[0,0,1,1] 表示识别命中（但位置不重要）
        # detail 里的内容会被记录到识别结果中，也可以在其他地方查询
        return CustomRecognition.AnalyzeResult(
            box=[0, 0, 1, 1],
            detail={"title": result["title"], "body": result["body"]},
        )
