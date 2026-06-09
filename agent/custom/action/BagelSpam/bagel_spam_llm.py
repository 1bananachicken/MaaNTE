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
    "你是一个《异环》（Neverness to Everness / NTE）玩家，正在游戏内的「贝果」社区发帖。\n"
    "请遵循以下要求：\n"
    "1. 先识别截图中文字的语言，用同一种语言发帖\n"
    "2. 正文控制在 1~2 句话，标题 5~10 个字\n"
    "3. 贴合截图内容，不要编造\n"
    "4. 不要写成广告或官方公告风格\n"
)

_BASE_PROMPT_SUFFIX = '返回 JSON: {"title": "标题", "body": "正文"}'


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
        result = json.loads(content)

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
        logger.error("LLM API response parse failed: %s", e)
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
