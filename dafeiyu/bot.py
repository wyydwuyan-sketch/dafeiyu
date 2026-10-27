"""主程序：收消息 → 攒一攒 → 生成回复 → 像真人一样分条发出去；后台做记忆整理和主动消息。"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime

from .config import Config
from .ilink import (ERR_REJECTED, ERR_SESSION_EXPIRED, MSG_TYPE_USER, Account, ILinkClient, ILinkError,
                    RateLimiter, extract_text)
from .llm import LLM, Usage
from .memory import BUBBLE_SEP, PASS_TOKEN, Memory, split_bubbles, strip_markdown
from .proactive import ProactiveState, decide
from .store import Store

log = logging.getLogger(__name__)

TOKEN_TTL_S = 24 * 3600


class Bot:
    def __init__(self, cfg: Config, account: Account, ilink: ILinkClient, llm: LLM, store: Store,
                 clock=time.time, sleep=asyncio.sleep):
        self.cfg = cfg
        self.account = account
        self.ilink = ilink
        self.llm = llm
        self.store = store
        self.memory = Memory(cfg, store, llm)
        self.clock = clock
        self.sleep = sleep
        self.limiter = RateLimiter(cfg.ilink.rate_max_sends, cfg.ilink.rate_window_s, clock=clock)
        self._new_msg = asyncio.Event()
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._typing_ticket: tuple[str, str] | None = None  # (context_token, ticket)

    # ================= 状态 =================
    @property
    def owner(self) -> str:
        return self.account.owner_user_id

    def replies_left(self) -> int:
        if not self.store.get("ctx_token"):
            return 0
        if self.clock() - self.store.get("ctx_token_ts", 0) > TOKEN_TTL_S:
            return 0
        return self.cfg.ilink.replies_per_token - self.store.get("replies_on_token", 0)

    def _add_usage(self, u: Usage) -> None:
        day = datetime.now(self.memory.tz).strftime("%Y-%m-%d")
        st = self.store.get("usage", {})
        if st.get("day") != day:
            st = {"day": day, "prompt": 0, "hit": 0, "out": 0, "calls": 0}
        st["prompt"] += u.prompt_tokens
        st["hit"] += u.cache_hit
        st["out"] += u.completion_tokens
        st["calls"] += 1
        self.store.set("usage", st)

    # ================= 收消息 =================
    async def handle_inbound(self, msg: dict) -> None:
        if msg.get("message_type") != MSG_TYPE_USER:
            return
        sender = msg.get("from_user_id") or ""
        if self.owner and sender != self.owner:
            log.warning("忽略非主人的消息 from=%s", sender)
            return
        if not self.owner and sender:
            self.account.owner_user_id = sender
        mid = str(msg.get("message_id") or "")
        seen: list[str] = self.store.get("seen_ids", [])
        if mid and mid in seen:
            return
        if mid:
            self.store.set("seen_ids", (seen + [mid])[-300:])

        now = self.clock()
        if msg.get("context_token"):
            self.store.set("ctx_token", msg["context_token"])
            self.store.set("ctx_token_ts", now)
            self.store.set("replies_on_token", 0)
        self.store.set("last_user_ts", now)
        self.store.set("unanswered", 0)

        text = extract_text(msg)
        if not text:
            return
        log.info("收到：%s", text[:80])
        if await self._command(text):
            return
        self.store.add_message("user", text, ts=now)
        self._new_msg.set()

    async def _command(self, text: str) -> bool:
        t = text.strip()
        if t == "/ping":
            await self.send_bubbles(["pong"], kind="system", store=False)
            return True
        if t in ("/状态", "/status"):
            u = self.store.get("usage", {})
            rate = (u.get("hit", 0) / u["prompt"] * 100) if u.get("prompt") else 0
            notes = len(self.store.active_notes())
            msg = (f"今天调用 {u.get('calls', 0)} 次，输入 {u.get('prompt', 0)} token（缓存命中 {rate:.0f}%），"
                   f"输出 {u.get('out', 0)} token。\n零碎记忆 {notes} 条，当前额度还剩 {self.replies_left() - 1} 条。")
            await self.send_bubbles([msg], kind="system", store=False)
            return True
        if t in ("/记忆", "/memory"):
            prof = self.memory.profile() or "（还没整理出画像）"
            await self.send_bubbles([prof[:1500]], kind="system", store=False)
            return True
        return False

    async def poll_loop(self) -> None:
        buf = self.store.get("get_updates_buf", "")
        timeout = 35.0
        failures = 0
        while not self._stop.is_set():
            try:
                resp = await self.ilink.get_updates(buf, timeout_s=timeout)
            except Exception as e:
                failures += 1
                log.warning("getupdates 出错（%d）：%s", failures, e)
                await self.sleep(30 if failures >= 3 else 2)
                if failures >= 3:
                    failures = 0
                continue
            if resp.get("longpolling_timeout_ms"):
                timeout = resp["longpolling_timeout_ms"] / 1000
            ret, errcode = resp.get("ret"), resp.get("errcode")
            if (ret not in (None, 0)) or (errcode not in (None, 0)):
                if ERR_SESSION_EXPIRED in (ret, errcode):
                    log.error("微信登录态失效了，请重新运行：python -m dafeiyu login 。1 小时后再试。")
                    await self.sleep(3600)
                    continue
                failures += 1
                log.warning("getupdates ret=%s errcode=%s %s", ret, errcode, resp.get("errmsg"))
                await self.sleep(30 if failures >= 3 else 2)
                continue
            failures = 0
            if resp.get("get_updates_buf"):
                buf = resp["get_updates_buf"]
                self.store.set("get_updates_buf", buf)
            for m in resp.get("msgs") or []:
                try:
                    await self.handle_inbound(m)
                except Exception:
                    log.exception("处理消息出错")

    # ================= 回复 =================
    async def reply_loop(self) -> None:
        debounce = self.cfg.chat.debounce_s
        while not self._stop.is_set():
            await self._new_msg.wait()
            # 他可能在连发，等他停下来
            while True:
                self._new_msg.clear()
                try:
                    await asyncio.wait_for(self._new_msg.wait(), timeout=debounce)
                except asyncio.TimeoutError:
                    break
            async with self._lock:
                try:
                    await self.reply_once()
                except Exception:
                    log.exception("生成回复出错")

    async def reply_once(self) -> bool:
        last = self.store.last_message()
        if not last or last.role != "user":
            return False
        snapshot = last.id
        # "正在输入"和生成回复同时进行，不让它拖慢回复；发完再收起
        typing = asyncio.create_task(self._start_typing())
        try:
            res = await self.llm.chat(self.memory.build_reply_messages())
        finally:
            ticket = await typing
        try:
            self._add_usage(res.usage)
            newest = self.store.last_message()
            if newest and newest.id != snapshot:
                # 生成期间他又发了新消息：像真人一样，看完再回
                log.info("生成期间有新消息，重新组织回复")
                self._new_msg.set()
                return False
            bubbles = split_bubbles(strip_markdown(res.text), self.cfg.chat.max_bubbles)
            if not bubbles:
                return False
            sent = await self.send_bubbles(bubbles, kind="chat", first_delay=False)
            log.info("已回复 %d 条", len(sent))
            return True
        finally:
            await self._stop_typing(ticket)

    async def _start_typing(self) -> str | None:
        token = self.store.get("ctx_token")
        if not token:
            return None
        try:
            if not self._typing_ticket or self._typing_ticket[0] != token:
                t = await self.ilink.get_typing_ticket(self.owner, token)
                self._typing_ticket = (token, t) if t else None
            if self._typing_ticket:
                await self.ilink.send_typing(self.owner, self._typing_ticket[1], True)
                return self._typing_ticket[1]
        except Exception as e:
            log.debug("typing 失败：%s", e)
        return None

    async def _stop_typing(self, ticket: str | None) -> None:
        if ticket:
            try:
                await self.ilink.send_typing(self.owner, ticket, False)
            except Exception:
                pass

    async def send_bubbles(self, bubbles: list[str], kind: str, store: bool = True,
                           first_delay: bool = True) -> list[str]:
        """一条条发，中间按字数停顿；额度不够就把剩下的合并成一条。返回实际发出的内容。"""
        sent: list[str] = []
        token = self.store.get("ctx_token")
        start = self.store.last_message()
        start_id = start.id if start else 0
        row_id: int | None = None
        for i, text in enumerate(bubbles):
            left = self.replies_left()
            if left <= 0:
                log.warning("当前 context_token 额度用完或已过期，剩余内容不发了")
                break
            if left == 1 and i < len(bubbles) - 1:
                text = "\n".join(bubbles[i:])  # 最后一个额度，把剩下的合并发
            if i > 0 or first_delay:
                delay = min(8.0, max(1.0, len(text) * self.cfg.chat.typing_s_per_char))
                ticket = await self._start_typing()
                await self.sleep(delay * random.uniform(0.8, 1.2))
                await self._stop_typing(ticket)
            # 回他的消息不预先限速（平台真拒了再退避）；主动消息不硬等：
            # 频率快超了，或者他已经开口了，剩下的就不发了，先听他说
            if kind == "proactive":
                user_last = self.store.last_message("user")
                if user_last and user_last.id > start_id:
                    log.info("他回话了，主动消息剩下的不发了")
                    break
                if self.limiter.wait_time() > 0:
                    log.info("发送太频繁，主动消息剩下的不发了")
                    break
            ok = await self._send_one(text, token)
            if not ok:
                break
            sent.append(text)
            if store:
                # 发一条记一条：他中途插话时，记录里的先后和微信里看到的一致
                content = f"\n{BUBBLE_SEP}\n".join(sent)
                if row_id is None:
                    row_id = self.store.add_message("assistant", content, kind=kind, ts=self.clock())
                else:
                    self.store.update_message(row_id, content)
            if text is not bubbles[i]:
                break
        return sent

    async def _send_one(self, text: str, token: str | None) -> bool:
        for attempt in range(2):
            try:
                await self.ilink.send_text(self.owner, text, token)
                self.limiter.record()
                self.store.set("replies_on_token", self.store.get("replies_on_token", 0) + 1)
                return True
            except ILinkError as e:
                if e.ret == ERR_REJECTED and attempt == 0:
                    log.warning("发送被拒（ret=-2，可能是频率或额度），30 秒后重试一次")
                    await self.sleep(30)
                    continue
                if e.ret == ERR_SESSION_EXPIRED:
                    log.error("微信登录态失效，请重新扫码：python -m dafeiyu login")
                log.error("发送失败：%s", e)
                return False
            except Exception as e:
                log.error("发送失败：%s", e)
                return False
        return False

    # ================= 主动消息 =================
    def proactive_state(self) -> ProactiveState:
        now = self.clock()
        day = datetime.now(self.memory.tz).strftime("%Y-%m-%d")
        pc = self.store.get("proactive_count", {})
        last = self.store.last_message()
        return ProactiveState(
            now=now,
            last_user_ts=self.store.get("last_user_ts"),
            last_activity_ts=last.ts if last else None,
            unanswered=self.store.get("unanswered", 0),
            sent_today=pc.get("n", 0) if pc.get("day") == day else 0,
            replies_left=self.replies_left(),
            followups=self.store.pending_followups(until=now),
        )

    async def try_proactive(self) -> bool:
        st = self.proactive_state()
        d = decide(st, self.cfg.proactive, self.memory.tz)
        if not d.ok:
            log.debug("不主动：%s", d.reason)
            return False
        res = await self.llm.chat(self.memory.build_proactive_messages(d.situation))
        self._add_usage(res.usage)
        due_ids = [f.id for f in st.followups]
        if PASS_TOKEN in res.text or not res.text.strip():
            log.info("模型决定这次不打扰")
            return False
        # 主动消息可以连发多条；额度不够时 send_bubbles 会把剩下的合并
        bubbles = split_bubbles(strip_markdown(res.text), self.cfg.proactive.max_bubbles)
        sent = await self.send_bubbles(bubbles, kind="proactive")
        if sent:
            day = datetime.now(self.memory.tz).strftime("%Y-%m-%d")
            pc = self.store.get("proactive_count", {})
            n = pc.get("n", 0) if pc.get("day") == day else 0
            self.store.set("proactive_count", {"day": day, "n": n + 1})
            self.store.set("unanswered", self.store.get("unanswered", 0) + 1)
            self.store.finish_followups(due_ids)
            log.info("主动发了：%s", " / ".join(sent))
        return bool(sent)

    async def proactive_loop(self) -> None:
        while not self._stop.is_set():
            wait = random.uniform(self.cfg.proactive.check_min_min, self.cfg.proactive.check_max_min) * 60
            await self.sleep(wait)
            async with self._lock:
                try:
                    await self.try_proactive()
                except Exception:
                    log.exception("主动消息出错")

    # ================= 后台维护 =================
    async def maintenance_once(self) -> None:
        now = self.clock()
        last = self.store.last_message()
        idle = now - last.ts if last else 0
        if last and idle >= self.cfg.memory.extract_idle_min * 60 and self.store.get("extract_cursor", 0) < last.id:
            await self.memory.extract()
        if idle >= self.cfg.memory.extract_idle_min * 60:
            # 压缩历史会改变 system prompt，放在空闲时做，避免聊天中途缓存失效
            await self.memory.maybe_compact()
        local = datetime.fromtimestamp(now, self.memory.tz)
        today = local.strftime("%Y-%m-%d")
        if local.hour == self.cfg.memory.consolidate_hour and self.store.get("consolidated_day") != today:
            await self.memory.extract()
            await self.memory.consolidate()
            self.store.set("consolidated_day", today)
        # 过期两天还没问的事就算了
        stale = self.store.pending_followups(until=now - 2 * 86400)
        self.store.finish_followups([f.id for f in stale])

    async def maintenance_loop(self) -> None:
        while not self._stop.is_set():
            await self.sleep(60)
            async with self._lock:
                try:
                    await self.maintenance_once()
                except Exception:
                    log.exception("记忆整理出错")

    # ================= 启动 =================
    async def run(self) -> None:
        await self.ilink.notify(start=True)
        log.info("大肥鱼上线了。给微信里的 ClawBot 发条消息试试。")
        tasks = [asyncio.create_task(c) for c in
                 (self.poll_loop(), self.reply_loop(), self.proactive_loop(), self.maintenance_loop())]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await self.ilink.notify(start=False)
