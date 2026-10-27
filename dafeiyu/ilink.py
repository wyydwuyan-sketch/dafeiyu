"""微信 iLink Bot API 客户端（官方 ClawBot 插件同款协议）。

协议细节对照腾讯官方插件 @tencent-weixin/openclaw-weixin@2.4.10 的 src/api 实现：
- 所有接口都是 POST JSON（扫码状态查询是 GET），域名 ilinkai.weixin.qq.com
- 请求头：iLink-App-Id / iLink-App-ClientVersion / AuthorizationType / X-WECHAT-UIN / Authorization
- 每个请求体带 base_info: {channel_version, bot_agent}
- getupdates 长轮询，靠 get_updates_buf 游标续传
- sendmessage 必须带上用户最近一条消息里的 context_token
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import secrets
import uuid
from dataclasses import dataclass
import time
from typing import Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

# 与官方插件保持一致的协议版本号（用于 iLink-App-ClientVersion / channel_version）
PROTOCOL_VERSION = "2.4.10"
ILINK_APP_ID = "bot"
DEFAULT_BOT_TYPE = "3"
FIXED_LOGIN_BASE = "https://ilinkai.weixin.qq.com"

LONG_POLL_TIMEOUT_S = 35.0
API_TIMEOUT_S = 15.0
LIGHT_TIMEOUT_S = 10.0

# 消息类型
MSG_TYPE_USER = 1
MSG_TYPE_BOT = 2
MSG_STATE_FINISH = 2
ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5

# 错误码
ERR_SESSION_EXPIRED = -14  # bot 登录态失效，需要重新扫码
ERR_REJECTED = -2  # context_token 过期 / 回复条数超限 / 频率超限 / 文本过长


class ILinkError(Exception):
    def __init__(self, ret: int | None, errmsg: str = ""):
        super().__init__(f"iLink ret={ret} errmsg={errmsg}")
        self.ret = ret
        self.errmsg = errmsg


def _client_version(version: str) -> int:
    parts = [int(p) for p in version.split(".")[:3]] + [0, 0, 0]
    return ((parts[0] & 0xFF) << 16) | ((parts[1] & 0xFF) << 8) | (parts[2] & 0xFF)


def _random_wechat_uin() -> str:
    n = int.from_bytes(secrets.token_bytes(4), "big")
    return base64.b64encode(str(n).encode()).decode()


_ID_FIELDS = re.compile(r'("(?:message_id|msg_id|svr_id)"\s*:\s*)(-?\d+)')


def parse_json(text: str) -> dict:
    """消息 ID 是 uint64，先转成字符串再解析，避免精度问题（Python 本身不丢精度，这里只为统一成 str）。"""
    return json.loads(_ID_FIELDS.sub(r'\1"\2"', text)) if text else {}


@dataclass
class Account:
    bot_token: str
    bot_id: str
    base_url: str
    owner_user_id: str  # 扫码的人，也就是你自己

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        return cls(**{k: d[k] for k in ("bot_token", "bot_id", "base_url", "owner_user_id")})


def extract_text(msg: dict) -> str:
    """把一条入站消息转成纯文本。语音用平台自带的转写；图片等给占位符（DeepSeek 是纯文本模型）。"""
    parts: list[str] = []
    for item in msg.get("item_list") or []:
        t = item.get("type")
        if t == ITEM_TEXT and (item.get("text_item") or {}).get("text") is not None:
            parts.append(str(item["text_item"]["text"]))
        elif t == ITEM_VOICE:
            vt = (item.get("voice_item") or {}).get("text")
            parts.append(f"[语音] {vt}" if vt else "[语音，没转出文字]")
        elif t == ITEM_IMAGE:
            parts.append("[图片]")
        elif t == ITEM_FILE:
            name = (item.get("file_item") or {}).get("file_name") or ""
            parts.append(f"[文件 {name}]".strip())
        elif t == ITEM_VIDEO:
            parts.append("[视频]")
        ref = item.get("ref_msg")
        if ref:
            # 各层字段都可能是 null，逐层兜底
            quoted = ((ref.get("message_item") or {}).get("text_item") or {}).get("text") or ref.get("title")
            if quoted:
                parts.insert(0, f"[引用: {quoted[:80]}]")
    return "\n".join(p for p in parts if p).strip()


class ILinkClient:
    def __init__(self, base_url: str, token: str | None = None, bot_agent: str = "DaFeiYu/0.1.0",
                 http: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/") + "/"
        self.token = token
        self.bot_agent = bot_agent
        self._http = http or httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ---------- 底层 ----------
    def _common_headers(self) -> dict[str, str]:
        return {
            "iLink-App-Id": ILINK_APP_ID,
            "iLink-App-ClientVersion": str(_client_version(PROTOCOL_VERSION)),
        }

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": _random_wechat_uin(),
            **self._common_headers(),
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _base_info(self) -> dict[str, str]:
        return {"channel_version": PROTOCOL_VERSION, "bot_agent": self.bot_agent}

    async def _post(self, endpoint: str, body: dict, timeout: float, base: str | None = None) -> dict:
        url = (base.rstrip("/") + "/" if base else self.base_url) + endpoint
        resp = await self._http.post(url, content=json.dumps(body, ensure_ascii=False).encode(),
                                     headers=self._headers(), timeout=timeout)
        if resp.status_code >= 400:
            raise ILinkError(resp.status_code, resp.text[:200])
        return parse_json(resp.text)

    # ---------- 收消息 ----------
    async def get_updates(self, buf: str, timeout_s: float = LONG_POLL_TIMEOUT_S) -> dict:
        try:
            return await self._post("ilink/bot/getupdates",
                                    {"get_updates_buf": buf or "", "base_info": self._base_info()},
                                    timeout=timeout_s + 5)
        except httpx.TimeoutException:
            # 长轮询超时是正常的，返回空结果让上层继续
            return {"ret": 0, "msgs": [], "get_updates_buf": buf}

    # ---------- 发消息 ----------
    async def send_text(self, to_user_id: str, text: str, context_token: str | None) -> str | None:
        body = {
            "msg": {
                "from_user_id": "",
                "to_user_id": to_user_id,
                "client_id": f"dafeiyu-{uuid.uuid4().hex[:16]}",
                "message_type": MSG_TYPE_BOT,
                "message_state": MSG_STATE_FINISH,
                "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
                "context_token": context_token or None,
            },
            "base_info": self._base_info(),
        }
        data = await self._post("ilink/bot/sendmessage", body, timeout=API_TIMEOUT_S)
        ret = data.get("ret")
        if ret not in (None, 0):
            raise ILinkError(ret, data.get("errmsg", ""))
        return data.get("message_id")

    # ---------- "对方正在输入" ----------
    async def get_typing_ticket(self, user_id: str, context_token: str | None) -> str | None:
        data = await self._post("ilink/bot/getconfig", {
            "ilink_user_id": user_id, "context_token": context_token, "base_info": self._base_info(),
        }, timeout=LIGHT_TIMEOUT_S)
        return data.get("typing_ticket")

    async def send_typing(self, user_id: str, ticket: str, typing: bool = True) -> None:
        await self._post("ilink/bot/sendtyping", {
            "ilink_user_id": user_id, "typing_ticket": ticket, "status": 1 if typing else 2,
            "base_info": self._base_info(),
        }, timeout=LIGHT_TIMEOUT_S)

    # ---------- 生命周期通知 ----------
    async def notify(self, start: bool) -> None:
        ep = "ilink/bot/msg/notifystart" if start else "ilink/bot/msg/notifystop"
        try:
            await self._post(ep, {"base_info": self._base_info()}, timeout=LIGHT_TIMEOUT_S)
        except Exception as e:  # 通知失败不影响主流程
            log.debug("notify %s failed: %s", ep, e)

    # ---------- 扫码登录 ----------
    async def login(self, show_qr: Callable[[str], None],
                    ask_code: Callable[[str], Awaitable[str]],
                    timeout_s: float = 480) -> Account:
        """扫码绑定。show_qr 负责把二维码链接显示出来；ask_code 用于手机上显示配对数字时让你输入。"""
        poll_base = FIXED_LOGIN_BASE
        qr = await self._post(f"ilink/bot/get_bot_qrcode?bot_type={DEFAULT_BOT_TYPE}",
                              {"local_token_list": []}, timeout=API_TIMEOUT_S, base=FIXED_LOGIN_BASE)
        qrcode = qr["qrcode"]
        show_qr(qr["qrcode_img_content"])
        verify_code: str | None = None
        refreshes = 1
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            params = {"qrcode": qrcode}
            if verify_code:
                params["verify_code"] = verify_code
            try:
                r = await self._http.get(poll_base.rstrip("/") + "/ilink/bot/get_qrcode_status",
                                         params=params, headers=self._common_headers(), timeout=40)
                st = r.json() if r.status_code < 400 else {"status": "wait"}
            except (httpx.TimeoutException, httpx.TransportError, ValueError):
                st = {"status": "wait"}
            status = st.get("status")
            if status == "scaned":
                verify_code = None
                print("已扫码，请在手机上确认…")
            elif status == "need_verifycode":
                verify_code = await ask_code("输入手机微信上显示的数字：")
                continue
            elif status in ("expired", "verify_code_blocked"):
                refreshes += 1
                if refreshes > 3:
                    raise RuntimeError("二维码多次失效，请稍后再试")
                qr = await self._post(f"ilink/bot/get_bot_qrcode?bot_type={DEFAULT_BOT_TYPE}",
                                      {"local_token_list": []}, timeout=API_TIMEOUT_S, base=FIXED_LOGIN_BASE)
                qrcode = qr["qrcode"]
                verify_code = None
                print("二维码已刷新，请重新扫描：")
                show_qr(qr["qrcode_img_content"])
            elif status == "scaned_but_redirect" and st.get("redirect_host"):
                poll_base = f"https://{st['redirect_host']}"
            elif status == "binded_redirect":
                raise RuntimeError("这个 bot 已经绑定过了。如需重新绑定，请先在微信 ClawBot 插件里解除绑定。")
            elif status == "confirmed":
                if not st.get("ilink_bot_id") or not st.get("bot_token"):
                    raise RuntimeError("登录确认了，但服务器没返回 bot 凭据")
                return Account(bot_token=st["bot_token"], bot_id=st["ilink_bot_id"],
                               base_url=st.get("baseurl") or FIXED_LOGIN_BASE,
                               owner_user_id=st.get("ilink_user_id") or "")
            await asyncio.sleep(1)
        raise TimeoutError("扫码超时")


class RateLimiter:
    """滑动窗口限速：window_s 秒内最多 max_events 次。"""

    def __init__(self, max_events: int, window_s: float, clock: Callable[[], float] | None = None):
        self.max_events = max_events
        self.window_s = window_s
        self._clock = clock or time.monotonic
        self._events: list[float] = []

    def _prune(self) -> None:
        now = self._clock()
        self._events = [t for t in self._events if now - t < self.window_s]

    def wait_time(self) -> float:
        self._prune()
        if len(self._events) < self.max_events:
            return 0.0
        return self.window_s - (self._clock() - self._events[0]) + 0.5

    def record(self) -> None:
        self._events.append(self._clock())

    def remaining(self) -> int:
        self._prune()
        return self.max_events - len(self._events)

