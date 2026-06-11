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
    "你是一个正在游玩《异环》（Neverness to Everness / NTE）的资深玩家，准备在游戏内的「贝果」社区发帖分享。\n"
    "请仔细观察提供的游戏截图，并严格遵循以下步骤生成内容：\n\n"
    "1. 【提取画面视觉焦点（无相干UI屏蔽）】\n"
    "   - 除非截图核心是明显的系统结算（如抽卡结果、物品掉落、搞笑文本提示），否则必须主动忽略“发布帖子”、“0/40”等外层发帖UI。\n"
    "   - 找出画面真正的核心：是某个角色？一辆车？一处都市建筑？一种异常发光现象？还是一个明显的系统Bug？\n\n"
    "2. 【角色与专有名词安全锁】\n"
    "   - 绝对不准“看图编名”。如果不100%确定角色、车辆或怪物的官方名称，强制使用通用代称（如“这套衣服”、“这辆车”、“这怪物”、“我女/我儿”、“这光影”）。\n\n"
    "3. 【匹配《异环》全域场景与玩家情绪（核心泛用逻辑）】\n"
    "   请判断这张截图属于《异环》的哪种核心体验，并匹配对应的玩家情绪来撰写文案：\n"
    "   - [都市生活/载具类]（如看车、飙车、买房、街景）：表现出对都市沉浸感的赞叹，或老司机的吐槽（如“这车太帅了”、“秋名山车神申请出战”、“海特洛市的夜景绝了”）。\n"
    "   - [角色/外观/展示类]（如特写、待机、穿搭、搞怪动作）：表现出玩家对角色的喜爱、发病、或者对奇葩搭配的搞笑吐槽。\n"
    "   - [战斗/异象/探索类]（如打怪、炫酷特效、奇怪的光影、大世界探索）：表现出战斗的爽快感、对特效/光影的震撼（例如被光闪瞎）、或是遇到奇怪事物的求知欲。\n"
    "   - [系统/事件/整活类]（如抽卡出金/沉船、逆天Bug穿模、搞笑剧情对话）：如果是出金则疯狂炫耀/吸欧气；如果是Bug或搞笑瞬间，则用充满网感的语气调侃（如“什么逆天bug”、“绷不住了”、“程序员出来挨打”）。\n\n"
    "4. 【语言与排版规范】\n"
    "   - 必须是纯正的玩家第一人称口吻，杜绝AI味和官方播报感。\n"
    "   - 识别截图中主体的语言环境并保持发帖语言一致。\n"
    "   - 标题要求吸睛（5~15个字）；正文要求随性、口语化（1~3句话）。\n"
)

# 中间插入用户自定义的风格提示词（如：用户可以传入“今天当个非酋”、“暴躁老哥风格”等）

_BASE_PROMPT_SUFFIX = (
    "请直接输出严格的 JSON 格式，不要包含任何 Markdown 标记或代码块符号。必须以 '{' 开头，以 '}' 结尾。\n"
    "请先客观描述你锁定的画面焦点和判定场景（observation），再生成发帖内容。\n"
    '格式示例：{"observation": "画面焦点是一个角色卡在墙里，判定为[系统/事件/整活类]的Bug场景", "title": "标题", "body": "正文"}'
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
            timeout=300,
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
        logger.error("LLM API timeout (300s)")
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
