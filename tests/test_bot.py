import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from dafeiyu.bot import Bot
from dafeiyu.config import Config
from dafeiyu.ilink import (ILinkClient, ILinkError, Account, extract_text, _client_version, RateLimiter)
from dafeiyu.llm import LLMResult, Usage, parse_json_loose
from dafeiyu.memory import Memory, grams, split_bubbles, NOTE_MARK
from dafeiyu.proactive import ProactiveState, decide
from dafeiyu.store import Store, Followup
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")


# ---------------- fakes ----------------
class FakeLLM:
    def __init__(self, replies=None, json_replies=None):
        self.replies = list(replies or [])
        self.json_replies = list(json_replies or [])
        self.calls = []
        self.kwargs = []
        self.on_call = None

    async def chat(self, messages, **kw):
        self.calls.append(messages)
        self.kwargs.append(kw)
        if self.on_call:
            await self.on_call()
        reply = self.replies.pop(0) if self.replies else "嗯"
        if isinstance(reply, LLMResult):  # 想模拟工具调用时直接给 LLMResult
            return reply
        return LLMResult(text=reply, usage=Usage(prompt_tokens=100, cache_hit=80, completion_tokens=10))

    async def chat_json(self, messages, **kw):
        self.calls.append(messages)
        return self.json_replies.pop(0) if self.json_replies else {}


class FakeILink:
    def __init__(self):
        self.sent = []
        self.fail_with = []  # 依次抛出的错误

    async def send_text(self, to, text, token):
        if self.fail_with:
            err = self.fail_with.pop(0)
            if err:
                raise err
        self.sent.append((to, text, token))
        return "1"

    async def get_typing_ticket(self, user, token):
        return "ticket"

    async def send_typing(self, user, ticket, typing=True):
        pass

    async def notify(self, start):
        pass


class FakeWeb:
    def __init__(self, result="1. 金铲铲新赛季上线（官网，2026-10-26）\n新赛季今天开放"):
        self.result = result
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return self.result


def search_call(query, cid="call_1"):
    return {"id": cid, "type": "function",
            "function": {"name": "web_search", "arguments": json.dumps({"query": query}, ensure_ascii=False)}}


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


async def no_sleep(_s):
    return None


def make_bot(tmp_path, llm=None, clock=None, web=None):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.persona_file = Path(__file__).parent.parent / "persona.example.md"
    clock = clock or Clock(time.time())
    store = Store(tmp_path / "bot.db")
    acc = Account(bot_token="tk", bot_id="b@im.bot", base_url="https://x", owner_user_id="me@im.wechat")
    bot = Bot(cfg, acc, FakeILink(), llm or FakeLLM(), store, clock=clock, sleep=no_sleep, web=web)
    return bot


def user_msg(text, mid="1", token="ctx1", sender="me@im.wechat"):
    return {"message_id": mid, "from_user_id": sender, "message_type": 1, "context_token": token,
            "item_list": [{"type": 1, "text_item": {"text": text}}]}


# ---------------- iLink 协议 ----------------
def test_client_version_matches_official_encoding():
    assert _client_version("1.0.11") == 65547
    assert _client_version("2.4.10") == (2 << 16) | (4 << 8) | 10


def test_ilink_request_format():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text='{"ret":0,"message_id":12345678901234567890}')

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    c = ILinkClient("https://ilinkai.weixin.qq.com", token="TOKEN", http=http)
    mid = asyncio.run(c.send_text("u@im.wechat", "你好", "CTX"))
    assert mid == "12345678901234567890"  # uint64 不丢精度，转成字符串
    assert seen["url"] == "https://ilinkai.weixin.qq.com/ilink/bot/sendmessage"
    h = seen["headers"]
    assert h["authorization"] == "Bearer TOKEN"
    assert h["authorizationtype"] == "ilink_bot_token"
    assert h["ilink-app-id"] == "bot"
    assert "x-wechat-uin" in h
    msg = seen["body"]["msg"]
    assert msg["context_token"] == "CTX" and msg["message_type"] == 2 and msg["message_state"] == 2
    assert msg["item_list"] == [{"type": 1, "text_item": {"text": "你好"}}]
    assert seen["body"]["base_info"]["bot_agent"]


def test_ilink_send_error_raises():
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text='{"ret":-2}')))
    c = ILinkClient("https://x", token="T", http=http)
    with pytest.raises(ILinkError) as e:
        asyncio.run(c.send_text("u", "hi", "ctx"))
    assert e.value.ret == -2


def test_extract_text_voice_and_quote():
    msg = {"item_list": [
        {"type": 3, "voice_item": {"text": "明天面试"}},
        {"type": 1, "text_item": {"text": "紧张"}, "ref_msg": {"title": "加油"}},
    ]}
    assert extract_text(msg) == "[引用: 加油]\n[语音] 明天面试\n紧张"


def test_extract_text_quote_with_null_fields():
    msg = {"item_list": [{"type": 1, "text_item": {"text": "看这个"},
                          "ref_msg": {"message_item": {"text_item": None}, "title": "原消息"}}]}
    assert extract_text(msg) == "[引用: 原消息]\n看这个"


def test_rate_limiter():
    clk = Clock(0.0)
    rl = RateLimiter(2, 300, clock=clk)
    rl.record(); rl.record()
    assert rl.wait_time() > 0
    clk.t = 301
    assert rl.wait_time() == 0


# ---------------- 记忆 / prompt ----------------
def test_prompt_prefix_is_cache_friendly(tmp_path):
    bot = make_bot(tmp_path)
    m: Memory = bot.memory
    s = bot.store
    s.add_note("他在杭州做后端开发", 3)
    s.add_message("user", "今天加班好累", ts=1000)
    req1 = m.build_reply_messages()
    s.add_message("assistant", "又加班？几点下的", ts=1010)
    s.add_message("user", "十点，在赶一个需求", ts=1020)
    req2 = m.build_reply_messages()
    # system 完全一样
    assert req1[0] == req2[0]
    # 第一轮的 user 原文（不含备注）是第二轮的前缀
    assert req2[1]["content"].startswith(req1[1]["content"].split("\n\n" + NOTE_MARK)[0])
    assert NOTE_MARK not in req2[1]["content"]  # 历史里不残留备注
    assert NOTE_MARK in req2[-1]["content"]      # 只有最后一条带备注


def test_retrieve_finds_relevant_note(tmp_path):
    bot = make_bot(tmp_path)
    bot.store.add_note("他下周三要去上海面试", 3)
    bot.store.add_note("他喜欢玩金铲铲", 2)
    hits = bot.memory.retrieve("上海那个面试准备得怎么样")
    assert [n.content for n in hits] == ["他下周三要去上海面试"]
    assert "面试" in grams("面试准备")


def test_split_bubbles():
    assert split_bubbles("哈哈\n---\n真的假的\n---\n离谱", 3) == ["哈哈", "真的假的", "离谱"]
    assert split_bubbles("a\n---\nb\n---\nc\n---\nd", 3) == ["a", "b", "c\nd"]
    assert split_bubbles("一句话", 3) == ["一句话"]
    assert split_bubbles("在呢\n\n阴天确实烦", 3) == ["在呢", "阴天确实烦"]  # 空行也算分条
    assert split_bubbles("第一行\n第二行", 3) == ["第一行\n第二行"]  # 单个换行不拆


def test_parse_json_loose():
    assert parse_json_loose('```json\n{"a":1}\n```') == {"a": 1}
    assert parse_json_loose('好的：{"a":2} 完') == {"a": 2}


def test_extract_and_consolidate(tmp_path):
    llm = FakeLLM(replies=["- 在杭州做开发\n- 下周三上海面试"],
                  json_replies=[{"notes": [{"content": "他下周三去上海面试", "importance": 3}],
                                 "obsolete_ids": [], "followups": [{"due": "2030-01-01 20:00", "topic": "问面试结果"}]}])
    bot = make_bot(tmp_path, llm=llm)
    bot.store.add_message("user", "下周三去上海面试")
    n = asyncio.run(bot.memory.extract())
    assert n == 1
    assert bot.store.get("extract_cursor") == 1
    assert bot.store.pending_followups(until=2e9)[0].topic == "问面试结果"
    assert asyncio.run(bot.memory.consolidate())
    assert "上海面试" in bot.cfg.profile_file.read_text(encoding="utf-8")
    assert "下周三上海面试" in bot.memory.system_prompt()


def test_extract_tolerates_odd_importance(tmp_path):
    llm = FakeLLM(json_replies=[{"notes": [{"content": "他养了只猫", "importance": "3分"},
                                           {"content": "他喜欢吃辣", "importance": "高"}]}])
    bot = make_bot(tmp_path, llm=llm)
    bot.store.add_message("user", "我家猫又拆家了，晚上吃火锅")
    assert asyncio.run(bot.memory.extract()) == 2
    assert bot.store.get("extract_cursor") == 1
    assert [n.importance for n in bot.store.active_notes()] == [3, 2]


def test_compact_history(tmp_path):
    llm = FakeLLM(replies=["你们聊了加班的事。"])
    bot = make_bot(tmp_path, llm=llm)
    bot.cfg.chat.history_max_messages = 6
    bot.cfg.chat.history_compact_batch = 3
    for i in range(8):
        bot.store.add_message("user" if i % 2 == 0 else "assistant", f"m{i}", ts=1000 + i)
    assert asyncio.run(bot.memory.maybe_compact())
    assert bot.store.get("summary") == "你们聊了加班的事。"
    hist = bot.memory.history()
    assert hist[0].role == "user"  # 压缩停在 assistant 之后，不把一问一答拆开
    assert "你们聊了加班的事" in bot.memory.system_prompt()


# ---------------- 收发流程 ----------------
def test_reply_flow_splits_and_counts_quota(tmp_path):
    llm = FakeLLM(replies=["哈哈哈\n---\n**几点**下班的"])
    bot = make_bot(tmp_path, llm=llm)

    async def go():
        await bot.handle_inbound(user_msg("今天又加班"))
        return await bot.reply_once()

    assert asyncio.run(go())
    assert [t for _, t, _ in bot.ilink.sent] == ["哈哈哈", "几点下班的"]  # 去掉了 markdown
    assert all(tok == "ctx1" for _, _, tok in bot.ilink.sent)
    assert bot.store.get("replies_on_token") == 2
    last = bot.store.last_message()
    assert last.role == "assistant" and "---" in last.content


def test_quota_merges_remaining_bubbles(tmp_path):
    llm = FakeLLM(replies=["a\n---\nb\n---\nc"])
    bot = make_bot(tmp_path, llm=llm)

    async def go():
        await bot.handle_inbound(user_msg("hi"))
        bot.store.set("replies_on_token", bot.cfg.ilink.replies_per_token - 2)  # 只剩 2 条额度
        await bot.reply_once()

    asyncio.run(go())
    assert [t for _, t, _ in bot.ilink.sent] == ["a", "b\nc"]


# ---------------- 联网 ----------------
def test_reply_searches_then_answers(tmp_path):
    web = FakeWeb()
    llm = FakeLLM(replies=[LLMResult(text="", tool_calls=[search_call("金铲铲 新赛季")]), "新赛季今天开了 冲不冲"])
    bot = make_bot(tmp_path, llm=llm, web=web)

    async def go():
        await bot.handle_inbound(user_msg("金铲铲是不是更新了"))
        return await bot.reply_once()

    assert asyncio.run(go())
    assert web.queries == ["金铲铲 新赛季"]
    assert [t for _, t, _ in bot.ilink.sent] == ["新赛季今天开了 冲不冲"]
    second = llm.calls[1]
    assert second[-2]["tool_calls"][0]["id"] == "call_1"  # 工具调用和结果都接回了对话
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": web.result}
    assert llm.kwargs[0]["tools"] == llm.kwargs[1]["tools"]  # 工具定义每轮一样，缓存前缀不变
    # 搜索结果不进聊天记录
    assert [m.role for m in bot.store.messages_after(0)] == ["user", "assistant"]
    assert bot.store.get("usage")["searches"] == 1


def test_search_rounds_are_capped(tmp_path):
    web = FakeWeb()
    keeps_searching = [LLMResult(text="", tool_calls=[search_call(f"q{i}", f"c{i}")]) for i in range(5)]
    llm = FakeLLM(replies=keeps_searching)
    bot = make_bot(tmp_path, llm=llm, web=web)
    res = asyncio.run(bot._chat([{"role": "user", "content": "查查"}]))
    assert len(web.queries) == bot.cfg.web.max_rounds
    assert llm.kwargs[-1]["tool_choice"] == "none"  # 最后一轮不许再查
    assert len(llm.calls) == bot.cfg.web.max_rounds + 1


def test_force_search_requires_tool_on_first_round_only(tmp_path):
    web = FakeWeb()
    llm = FakeLLM(replies=[LLMResult(text="", tool_calls=[search_call("明日方舟 新卡池")]), "新池子是个狙"])
    bot = make_bot(tmp_path, llm=llm, web=web)
    asyncio.run(bot._chat([{"role": "user", "content": "主动"}], force_search=True))
    assert [kw["tool_choice"] for kw in llm.kwargs] == ["required", None]


def test_proactive_without_search_cannot_claim_it_just_saw_something(tmp_path):
    clock = Clock(_ts(10))
    llm = FakeLLM(replies=["刚刷到一个冷知识：海豚半脑睡觉"])
    bot = make_bot(tmp_path, llm=llm, clock=clock, web=FakeWeb())
    bot.cfg.web.proactive_search_chance = 0  # 这次不强制搜

    async def go():
        await bot.handle_inbound(user_msg("忙"))
        clock.t = _ts(15)
        return await bot.try_proactive()

    assert asyncio.run(go())
    assert bot.ilink.sent[-1][1] == "突然想到一个冷知识：海豚半脑睡觉"


def test_no_web_means_no_tools(tmp_path):
    llm = FakeLLM(replies=["嗯"])
    bot = make_bot(tmp_path, llm=llm)

    async def go():
        await bot.handle_inbound(user_msg("在吗"))
        return await bot.reply_once()

    assert asyncio.run(go())
    assert "tools" not in llm.kwargs[0]


def test_search_failure_does_not_break_reply(tmp_path):
    from dafeiyu.web import Web

    def boom(request):
        raise httpx.ConnectError("down")

    web = Web(Config().llm, Config().web, http=httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    out = asyncio.run(web.search("杭州天气"))
    assert "没搜成" in out


def test_search_results_are_compact():
    from dafeiyu.web import format_results
    out = format_results([{"title": "杭州天气", "site_name": "天气网", "date": "2026-10-27T00:00:00+00:00",
                           "summary": "多云转晴 " * 100}])
    assert out.startswith("1. 杭州天气（天气网，2026-10-27）")
    assert len(out) < 400


def test_chat_replies_not_held_by_rate_limiter(tmp_path):
    slept = []

    async def record_sleep(s):
        slept.append(s)

    bot = make_bot(tmp_path)
    bot.sleep = record_sleep
    for _ in range(bot.cfg.ilink.rate_max_sends):
        bot.limiter.record()  # 5 分钟内已经发满

    async def go():
        await bot.handle_inbound(user_msg("还在吗"))
        return await bot.reply_once()

    assert asyncio.run(go())
    assert len(bot.ilink.sent) == 1
    assert not any(s > 60 for s in slept)  # 聊天回复不因预先限速而干等


def test_ignores_strangers_and_duplicates(tmp_path):
    bot = make_bot(tmp_path)

    async def go():
        await bot.handle_inbound(user_msg("hi", sender="other@im.wechat"))
        await bot.handle_inbound(user_msg("hi", mid="9"))
        await bot.handle_inbound(user_msg("hi", mid="9"))

    asyncio.run(go())
    assert len(bot.store.messages_after(0)) == 1


def test_new_message_during_generation_regenerates(tmp_path):
    llm = FakeLLM(replies=["旧回复", "新回复"])
    bot = make_bot(tmp_path, llm=llm)

    async def arrive():
        if len(llm.calls) == 1:
            await bot.handle_inbound(user_msg("等等还有", mid="2", token="ctx2"))

    llm.on_call = arrive

    async def go():
        await bot.handle_inbound(user_msg("第一句"))
        r1 = await bot.reply_once()
        r2 = await bot.reply_once()
        return r1, r2

    r1, r2 = asyncio.run(go())
    assert (r1, r2) == (False, True)
    assert [t for _, t, _ in bot.ilink.sent] == ["新回复"]
    assert bot.ilink.sent[0][2] == "ctx2"  # 用最新的 context_token


def test_send_retries_once_on_rejected(tmp_path):
    llm = FakeLLM(replies=["好"])
    bot = make_bot(tmp_path, llm=llm)
    bot.ilink.fail_with = [ILinkError(-2, "busy"), None]

    async def go():
        await bot.handle_inbound(user_msg("hi"))
        await bot.reply_once()

    asyncio.run(go())
    assert [t for _, t, _ in bot.ilink.sent] == ["好"]


def test_commands_do_not_touch_history(tmp_path):
    bot = make_bot(tmp_path)
    asyncio.run(bot.handle_inbound(user_msg("/状态")))
    assert bot.store.messages_after(0) == []
    assert "今天调用" in bot.ilink.sent[0][1]


# ---------------- 主动消息 ----------------
def _ts(h, m=0):
    from datetime import datetime
    return datetime(2026, 10, 27, h, m, tzinfo=TZ).timestamp()


def _state(**kw):
    base = dict(now=_ts(15), last_user_ts=_ts(10), last_activity_ts=_ts(10, 5), unanswered=0,
                sent_today=0, replies_left=9, followups=[])
    base.update(kw)
    return ProactiveState(**base)


def test_proactive_rules():
    cfg = Config().proactive
    assert decide(_state(), cfg, TZ).ok
    assert decide(_state(now=_ts(23, 30)), cfg, TZ).reason == "不在允许时段"
    assert not decide(_state(last_user_ts=_ts(15) - 23 * 3600), cfg, TZ).ok  # 超窗口
    assert not decide(_state(unanswered=1), cfg, TZ).ok
    assert not decide(_state(sent_today=3), cfg, TZ).ok
    assert not decide(_state(replies_left=2), cfg, TZ).ok
    assert not decide(_state(last_activity_ts=_ts(14)), cfg, TZ).ok  # 刚聊过
    # 有到期的惦记事项时，安静半小时就可以
    d = decide(_state(last_activity_ts=_ts(14), followups=[Followup(1, _ts(14), "问面试结果")]), cfg, TZ)
    assert d.ok and "问面试结果" in d.situation


def test_proactive_flow(tmp_path):
    clock = Clock(_ts(10))
    llm = FakeLLM(replies=["哎，面试咋样了"])
    bot = make_bot(tmp_path, llm=llm, clock=clock)

    async def go():
        await bot.handle_inbound(user_msg("下午面试去了"))
        bot.store.add_message("assistant", "加油", ts=_ts(10))
        bot.store.add_followup(_ts(14), "问面试结果")
        clock.t = _ts(15)
        sent = await bot.try_proactive()
        again = await bot.try_proactive()  # 他没回，不会再发
        return sent, again

    sent, again = asyncio.run(go())
    assert (sent, again) == (True, False)
    assert bot.ilink.sent[-1][1] == "哎，面试咋样了"
    trigger = llm.calls[0][-1]["content"]
    assert "问面试结果" in trigger and "<PASS>" in trigger
    assert bot.store.pending_followups(until=_ts(16)) == []
    assert bot.store.last_message().kind == "proactive"


def test_proactive_pass(tmp_path):
    clock = Clock(_ts(10))
    llm = FakeLLM(replies=["<PASS>"])
    bot = make_bot(tmp_path, llm=llm, clock=clock)

    async def go():
        await bot.handle_inbound(user_msg("睡了"))
        clock.t = _ts(15)
        return await bot.try_proactive()

    assert asyncio.run(go()) is False
    assert bot.ilink.sent == []


def test_proactive_can_send_a_burst(tmp_path):
    clock = Clock(_ts(10))
    llm = FakeLLM(replies=["突然想到一个事\n---\n深海鱼大多不怕冷\n---\n因为那边一直就是 4 度\n---\n跟我待的机房差不多"])
    bot = make_bot(tmp_path, llm=llm, clock=clock)

    async def go():
        await bot.handle_inbound(user_msg("忙去了"))
        clock.t = _ts(15)
        return await bot.try_proactive()

    assert asyncio.run(go())
    assert len(bot.ilink.sent) == 4
    assert bot.store.last_message().content.count("---") == 3


def test_proactive_burst_stops_when_he_replies(tmp_path):
    clock = Clock(_ts(10))
    llm = FakeLLM(replies=["一\n---\n二\n---\n三", "嗯你说"])
    bot = make_bot(tmp_path, llm=llm, clock=clock)
    calls = 0

    async def he_cuts_in(_s):
        nonlocal calls
        calls += 1
        if calls == 2:  # 第一条发出后、第二条"打字"时他回话了
            await bot.handle_inbound(user_msg("等下，先说个事", mid="2", token="ctx2"))

    async def go():
        await bot.handle_inbound(user_msg("忙去了"))
        clock.t = _ts(15)
        bot.sleep = he_cuts_in
        await bot.try_proactive()
        return await bot.reply_once()

    assert asyncio.run(go())
    assert [t for _, t, _ in bot.ilink.sent] == ["一", "嗯你说"]
    roles = [m.role for m in bot.store.messages_after(0)]
    assert roles == ["user", "assistant", "user", "assistant"]  # 记录顺序和微信里一致


def test_proactive_trigger_lists_previous_shares(tmp_path):
    clock = Clock(_ts(10))
    llm = FakeLLM(replies=["<PASS>"])
    bot = make_bot(tmp_path, llm=llm, clock=clock)

    async def go():
        await bot.handle_inbound(user_msg("嗯"))
        bot.store.add_message("assistant", "章鱼有三颗心脏", kind="proactive", ts=_ts(9))
        bot.store.set("unanswered", 0)
        clock.t = _ts(15)
        await bot.try_proactive()

    asyncio.run(go())
    assert "章鱼有三颗心脏" in llm.calls[0][-1]["content"]
