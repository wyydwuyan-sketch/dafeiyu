"""联网搜索：DeepSeek Search API（POST /v1/tools/search，和聊天共用 api_key）。

按成功请求计费（约 $0.75 / 千次，空结果和失败不收费），结果条数不影响价格。
这里只把结果压成一小段文字交给模型，不贴链接、不入库，所以不影响聊天记录的缓存前缀。
"""
from __future__ import annotations

import logging
import time

import httpx

from .config import LLMConfig, WebConfig

log = logging.getLogger(__name__)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": ("上网搜索。需要实时信息（天气、新闻、游戏版本和卡池、价格、比分、最近发生的事），"
                        "或者对某个事实拿不准时用。闲聊、聊感受不要用。"),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索词，简短具体，128 字以内，比如“金铲铲 新赛季 更新内容”"},
            },
            "required": ["query"],
        },
    },
}


class Web:
    def __init__(self, llm_cfg: LLMConfig, cfg: WebConfig, http: httpx.AsyncClient | None = None):
        self.llm_cfg = llm_cfg
        self.cfg = cfg
        self._http = http or httpx.AsyncClient()

    async def search(self, query: str) -> str:
        """返回给模型看的结果摘要；出错也返回一句话，不抛异常，免得一次搜索失败拖垮整条回复。"""
        query = (query or "").strip()[:128]
        if not query:
            return "搜索词是空的。"
        t0 = time.monotonic()
        try:
            resp = await self._http.post(
                self.llm_cfg.base_url.rstrip("/") + "/v1/tools/search",
                json={"query": query, "count": self.cfg.count},
                headers={"Authorization": f"Bearer {self.llm_cfg.api_key}"},
                timeout=self.cfg.timeout_s,
            )
        except httpx.HTTPError as e:
            log.warning("搜索「%s」出错：%s", query, e)
            return "这次没搜成（网络出错），别编，直接说没查到。"
        if resp.status_code >= 400:
            log.warning("搜索「%s」失败：%s %s", query, resp.status_code, resp.text[:200])
            return "这次没搜成，别编，直接说没查到。"
        results = (resp.json() or {}).get("results") or []
        log.info("搜索「%s」%d 条 | 耗时 %.2fs", query, len(results), time.monotonic() - t0)
        if not results:
            return "没搜到相关结果。"
        return format_results(results)


def format_results(results: list[dict], max_chars: int = 300) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        date = (r.get("date") or "")[:10] or "日期不详"
        text = " ".join((r.get("summary") or r.get("snippet") or "").split())[:max_chars]
        lines.append(f"{i}. {r.get('title') or ''}（{r.get('site_name') or ''}，{date}）\n{text}")
    return "\n".join(lines)
