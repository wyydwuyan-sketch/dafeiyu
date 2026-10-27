"""DeepSeek Chat Completions（OpenAI 兼容）调用。"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

import httpx

from .config import LLMConfig

log = logging.getLogger(__name__)


@dataclass
class Usage:
    prompt_tokens: int = 0
    cache_hit: int = 0
    cache_miss: int = 0
    completion_tokens: int = 0

    def __str__(self) -> str:
        rate = self.cache_hit / self.prompt_tokens * 100 if self.prompt_tokens else 0
        return (f"输入 {self.prompt_tokens}（缓存命中 {self.cache_hit}，{rate:.0f}%）"
                f" 输出 {self.completion_tokens}")


@dataclass
class LLMResult:
    text: str
    usage: Usage = field(default_factory=Usage)


class LLM:
    def __init__(self, cfg: LLMConfig, http: httpx.AsyncClient | None = None):
        if not cfg.api_key:
            raise ValueError("缺少 DeepSeek API Key：在 config.toml 的 [llm] api_key 里填，或设置环境变量 DEEPSEEK_API_KEY")
        self.cfg = cfg
        self._http = http or httpx.AsyncClient()

    async def chat(self, messages: list[dict], *, model: str | None = None, temperature: float | None = None,
                   max_tokens: int | None = None, json_mode: bool = False) -> LLMResult:
        body = {
            "model": model or self.cfg.chat_model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            # 聊天不需要思考：关掉省输出 token，也更快
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.monotonic()
        resp = await self._http.post(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            timeout=self.cfg.timeout_s,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"DeepSeek API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            cache_hit=u.get("prompt_cache_hit_tokens", 0),
            cache_miss=u.get("prompt_cache_miss_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
        )
        text = (data["choices"][0]["message"].get("content") or "").strip()
        log.info("LLM %s | %s | 耗时 %.2fs", body["model"], usage, time.monotonic() - t0)
        return LLMResult(text=text, usage=usage)

    async def chat_json(self, messages: list[dict], **kw) -> dict:
        res = await self.chat(messages, json_mode=True, temperature=kw.pop("temperature", 0.3), **kw)
        return parse_json_loose(res.text)


def parse_json_loose(text: str) -> dict:
    """模型偶尔会包一层 ```json```，或者前后带废话，这里尽量把 JSON 抠出来。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    log.warning("无法解析模型返回的 JSON：%s", text[:200])
    return {}
