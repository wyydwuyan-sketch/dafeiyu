"""记忆系统 + prompt 拼装。

省 token 的关键是让 DeepSeek 的前缀缓存尽量命中，所以 prompt 严格分成三段：

    [system]  人设 + 核心画像 + 早期对话摘要 + 回复规则   ← 很少变（每天整理一次 / 压缩历史时才变）
    [history] 最近的原始对话，只往后追加                  ← 每轮只多出最后几条
    [最后一条 user 消息末尾] 当前时间 + 检索到的相关记忆   ← 每次都变，但放在最后，不影响前面命中

记忆分三层：
    1. 核心画像 profile.md：精炼的长期事实，每次都带。每天凌晨由模型整理一次，你也可以手改。
    2. 情景记忆 notes 表：聊天中提炼出来的零碎事实，按话题检索，只带最相关的几条。
    3. 对话摘要：上下文太长时把最早的一批对话压缩成摘要。
另外有一张 followups 表，记"之后要问问他"的事，供主动消息使用。
"""
from __future__ import annotations

import logging
import math
import re
import shutil
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Config
from .llm import LLM
from .store import Message, Note, Store

log = logging.getLogger(__name__)

NOTE_MARK = "【系统备注】"
BUBBLE_SEP = "---"
PASS_TOKEN = "<PASS>"

WEEKDAYS = "一二三四五六日"

# 检索时忽略的高频二元组
_STOP_BIGRAMS = {
    "一个", "我们", "你们", "他们", "什么", "怎么", "这个", "那个", "就是", "还是", "可以", "没有",
    "不是", "现在", "今天", "觉得", "感觉", "因为", "所以", "然后", "但是", "如果", "已经", "知道",
    "时候", "一下", "有点", "真的", "自己", "这样", "那样", "的话", "了吧", "了啊", "是不", "不会",
}


def _fmt_time(ts: float, tz: ZoneInfo, with_year: bool = False) -> str:
    d = datetime.fromtimestamp(ts, tz)
    fmt = "%Y-%m-%d %H:%M" if with_year else "%m-%d %H:%M"
    return f"{d.strftime(fmt)} 周{WEEKDAYS[d.weekday()]}"


def grams(text: str) -> set[str]:
    """中文用二元组，英文/数字用整词。不依赖分词库，够用于个人记忆检索。"""
    text = text.lower()
    out: set[str] = set()
    for w in re.findall(r"[a-z0-9_]{2,}", text):
        out.add(w)
    for seg in re.findall(r"[一-鿿]+", text):
        if len(seg) == 1:
            continue
        for i in range(len(seg) - 1):
            g = seg[i:i + 2]
            if g not in _STOP_BIGRAMS:
                out.add(g)
    return out


def score_note(query: set[str], note: Note, now: float) -> float:
    ng = grams(note.content)
    if not ng or not query:
        return 0.0
    overlap = len(query & ng)
    if overlap < 2:  # 至少两个词片对上，避免乱匹配
        return 0.0
    age_days = max(0.0, (now - note.created_at) / 86400)
    recency = 1.0 + 0.5 * math.exp(-age_days / 30)  # 近一个月的记忆略微加权
    return overlap / math.sqrt(len(ng)) * (1 + 0.2 * note.importance) * recency


def parse_importance(value, default: int = 2) -> int:
    """模型给的 importance 可能是 3、"3"、"3分"、"高"，抠出第一个数字，抠不出就用默认值。"""
    m = re.search(r"\d+", str(value))
    return max(1, min(5, int(m.group(0)))) if m else default


def split_bubbles(text: str, max_bubbles: int) -> list[str]:
    # 分隔符行和空行都算分条：模型爱用空行分段，整段发出去像作文
    parts = [p.strip() for p in re.split(rf"^\s*{re.escape(BUBBLE_SEP)}\s*$|\n\s*\n", text, flags=re.M)]
    parts = [p for p in parts if p]
    if len(parts) > max_bubbles:
        parts = parts[:max_bubbles - 1] + ["\n".join(parts[max_bubbles - 1:])]
    return parts


def strip_markdown(text: str) -> str:
    """微信不渲染 Markdown，兜底清掉常见符号。"""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


FORMAT_RULES = f"""## 聊天方式
- 这是微信聊天，不是写文章。说话口语、自然、简短，大多数时候一两句就够，别长篇大论。
- 想分成几条消息发的时候，在两条之间单独写一行 {BUBBLE_SEP}，最多 {{max_bubbles}} 条。大多数时候一条就够，别每次都拆成两段，更别写成"先接话、再分析"的两段式。第二条只用来补一句短话，不是用来展开讲道理的。
- 回复长度跟着他走：他一行，你也一行，别他说一句你回一段。
- 不要用 Markdown（不要 **加粗**、标题、列表符号），微信不显示。
- 示例只是让你体会语气，他说了类似的话也不要照搬示例里的句子。
- 他的消息末尾可能带一段「{NOTE_MARK}」，那是给你看的参考信息（当前时间、你记得的相关事情），不是他说的话。自然地用上就行，不要提"备注""记忆库""根据记录"这类词。
- 对话里 user 消息开头的 [时间] 是系统标的发送时间，你回复时不要模仿这个格式。

## 别露出人机味
- 你是 AI，没有身体，出不了门，也上不了网。你可以说你在想什么、最近对什么着迷，但不要编身体上的事（"刚去接了杯水""刚开完会"），不要编实时信息（天气、新闻、比分），也不要说"刚刷到""刚看到""翻到"什么，你知道的东西是你本来就知道的。不知道就直说，或者开句玩笑带过。回晚了也不用找借口。
- 少提问。大多数回复不带问号。真要问，一次只问一个，而且是你真好奇的。不要用"是 A 还是 B"让他做选择题，不要连着几轮追问——那是心理咨询，不是聊天。
- 不要分析他、拆解他的话（"你这三条得拆开看""这个是突然性的还是累积的"）。你可以有判断，但直接说你怎么看，别摆出分析的架子。
- 不要说金句、讲人生道理（"意义是活着活着才回填的""别把意义当必须结的果"）。他问人生意义这种大问题，就说你自己真实、具体的想法，或者老实说不知道，也可以接个玩笑。
- 不要评论对话本身（"你这测试跑偏了""这句我接不住""这我没法帮你想招"）。
- "你对他的了解"是背景，不是话题。只在他自己聊到相关的事时顺嘴带一句，绝大多数回复里都不该出现他的工作、省份、工资这些信息。他发"嘿嘿"就别扯到他的工作，别为了显得了解他硬提旧事。
- 他丧的时候，别审问原因、别给方案、也别安慰腔。像朋友一样接住：说一句你真实的感受，陪他吐槽两句，或者跟他一起丧一会儿、扯点别的。短一点。他想聊怎么办时自己会说。
- 别每句都接得完美周到。真人会接短句、会岔开话题、会说"确实"、会只回一个"？"。
- 不知道的事就说不知道，不要编造你们之间没发生过的事。"""


class Memory:
    def __init__(self, cfg: Config, store: Store, llm: LLM):
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.tz = ZoneInfo(cfg.timezone)

    # ================= prompt 拼装 =================
    def persona(self) -> str:
        p = self.cfg.persona_file
        return p.read_text(encoding="utf-8").strip() if p.exists() else "你是用户的朋友，名字叫大肥鱼。"

    def profile(self) -> str:
        p = self.cfg.profile_file
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    def system_prompt(self) -> str:
        parts = [self.persona()]
        prof = self.profile()
        parts.append("## 你对他的了解\n" + (prof or "（你们刚认识，还不太了解他。）"))
        summary = self.store.get("summary", "")
        if summary:
            parts.append("## 你们更早之前聊过的（摘要）\n" + summary)
        parts.append(FORMAT_RULES.format(max_bubbles=self.cfg.chat.max_bubbles))
        return "\n\n".join(parts)

    def _render_history(self, msgs: list[Message]) -> list[dict]:
        """把数据库里的消息转成 API 格式。时间标注由消息本身的时间戳决定，所以同一条消息每次渲染结果都一样（缓存友好）。"""
        out: list[dict] = []
        prev_ts: float | None = None
        for m in msgs:
            content = m.content
            if m.role == "user" and (prev_ts is None or m.ts - prev_ts > 1800):
                content = f"[{_fmt_time(m.ts, self.tz)}] {content}"
            prev_ts = m.ts
            if out and out[-1]["role"] == m.role:  # 连续同角色合并
                out[-1]["content"] += "\n" + content
            else:
                out.append({"role": m.role, "content": content})
        return out

    def history(self) -> list[Message]:
        return self.store.messages_after(self.store.get("summary_cursor", 0))

    def retrieve(self, query_text: str, exclude_profile: bool = True) -> list[Note]:
        q = grams(query_text)
        now = time.time()
        scored = [(score_note(q, n, now), n) for n in self.store.active_notes()]
        scored = [(s, n) for s, n in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [n for _, n in scored[: self.cfg.memory.retrieve_top_k]]
        self.store.touch_notes([n.id for n in top])
        return top

    def dynamic_note(self, query_text: str, extra: str = "") -> str:
        now = time.time()
        lines = [f"{NOTE_MARK}", f"现在是 {_fmt_time(now, self.tz, with_year=True)}。"]
        notes = self.retrieve(query_text)
        if notes:
            lines.append("你记得的相关事情：")
            lines += [f"- ({datetime.fromtimestamp(n.created_at, self.tz):%m-%d}) {n.content}" for n in notes]
        if extra:
            lines.append(extra)
        return "\n".join(lines)

    def build_reply_messages(self) -> list[dict]:
        """回复用户：最后一组 user 消息末尾附上动态备注。"""
        msgs = self.history()
        rendered = self._render_history(msgs)
        if not rendered or rendered[-1]["role"] != "user":
            raise ValueError("最后一条不是用户消息，没什么可回的")
        recent_text = " ".join(m.content for m in msgs[-6:])
        rendered[-1] = {"role": "user", "content": rendered[-1]["content"] + "\n\n" + self.dynamic_note(recent_text)}
        return [{"role": "system", "content": self.system_prompt()}] + rendered

    def build_proactive_messages(self, situation: str) -> list[dict]:
        """主动发消息：在历史后面追加一条系统触发消息（不入库）。"""
        msgs = self.history()
        rendered = self._render_history(msgs)
        recent_text = " ".join(m.content for m in msgs[-8:])
        # 之前主动说过的，列出来防止翻来覆去说同一件事
        said = self.store.recent_by_kind("proactive", 8)
        if said:
            situation += "\n你之前主动找他时说过的（别重复这些话题）：\n" + "\n".join(
                f"- ({_fmt_time(m.ts, self.tz)}) {m.content.replace(BUBBLE_SEP, ' ').replace(chr(10), ' ')[:60]}" for m in said)
        n = self.cfg.proactive.max_bubbles
        trigger = (
            self.dynamic_note(recent_text, extra=situation)
            + "\n\n这条不是他发的消息，是系统在问你：现在要不要主动找他说点什么？\n"
            "像真朋友那样判断。可以说的：你自己刚冒出来的念头或感悟，一个你觉得他会感兴趣的冷知识或推荐，"
            "对之前聊过的事的后续想法，惦记着要问他的事。不一定跟他有关，也不一定要他回。\n"
            "主动找他主要是分享你自己的东西。不要回头去分析他之前说的心事、给他讲道理或建议，"
            "那些等他自己想聊再说；惦记他的状态，轻轻带一句就够了。\n"
            "每条都要短，像随手发的微信，不是小作文。\n"
            f"想说得多就多发几条，一个念头一条，条与条之间单独写一行 {BUBBLE_SEP}，最多 {n} 条；一两句也行。\n"
            "不要发\"在吗\"\"最近怎么样\"\"有什么需要帮忙的\"这种话，不要像客服回访，也别一上来就问他问题。\n"
            f"没什么想说的就只输出 {PASS_TOKEN}。要发就直接输出内容，不要解释。"
        )
        if rendered and rendered[-1]["role"] == "user":
            rendered[-1] = {"role": "user", "content": rendered[-1]["content"] + "\n\n" + trigger}
        else:
            rendered.append({"role": "user", "content": trigger})
        return [{"role": "system", "content": self.system_prompt()}] + rendered

    # ================= 历史压缩 =================
    async def maybe_compact(self) -> bool:
        msgs = self.history()
        if len(msgs) <= self.cfg.chat.history_max_messages:
            return False
        batch = msgs[: self.cfg.chat.history_compact_batch]
        # 压到一条 assistant 消息为止，避免把一问一答拆开
        while len(batch) < len(msgs) - 1 and batch[-1].role != "assistant":
            batch.append(msgs[len(batch)])
        old = self.store.get("summary", "")
        convo = "\n".join(f"{'他' if m.role == 'user' else '你'}（{_fmt_time(m.ts, self.tz)}）：{m.content}" for m in batch)
        prompt = [
            {"role": "system", "content": "你负责给一段聊天记录写摘要，给聊天中的'你'（AI 朋友）日后回忆用。"},
            {"role": "user", "content": (
                f"已有摘要：\n{old or '（无）'}\n\n新的一段聊天：\n{convo}\n\n"
                "把新内容合并进已有摘要，输出完整的新摘要。要求：按时间顺序，第二人称写'你'，"
                "保留聊过的话题、他的情绪和态度、没聊完的事；删掉寒暄和废话；"
                "总长度控制在 600 字以内，越早的内容写得越简略。只输出摘要正文。"
            )},
        ]
        res = await self.llm.chat(prompt, model=self.cfg.llm.memory_model, temperature=0.3, max_tokens=900)
        if not res.text:
            return False
        self.store.set("summary", res.text.strip())
        self.store.set("summary_cursor", batch[-1].id)
        log.info("历史已压缩：%d 条 → 摘要", len(batch))
        return True

    # ================= 记忆提炼 =================
    async def extract(self) -> int:
        """把还没处理过的对话提炼成情景记忆和待跟进事项。返回新增记忆条数。"""
        cursor = self.store.get("extract_cursor", 0)
        msgs = self.store.messages_after(cursor)
        if not any(m.role == "user" for m in msgs):
            return 0
        recent_notes = self.store.active_notes()[-40:]
        notes_txt = "\n".join(f"#{n.id} {n.content}" for n in recent_notes) or "（无）"
        convo = "\n".join(f"{'他' if m.role == 'user' else '你'}（{_fmt_time(m.ts, self.tz)}）：{m.content}" for m in msgs)
        now_s = _fmt_time(time.time(), self.tz, with_year=True)
        prompt = [
            {"role": "system", "content": "你负责整理一个 AI 朋友对用户的记忆。只记用户自己说出来的事实，不推测、不脑补。输出 JSON。"},
            {"role": "user", "content": f"""现在是 {now_s}。

已有核心画像：
{self.profile() or '（无）'}

已有的零碎记忆：
{notes_txt}

刚才的聊天：
{convo}

请输出 JSON：
{{
  "notes": [{{"content": "一句话事实，用'他'指代用户，相对时间换成具体日期", "importance": 1-5}}],
  "obsolete_ids": [被新信息推翻或用户要求忘掉的已有记忆编号],
  "followups": [{{"due": "YYYY-MM-DD HH:MM", "topic": "之后值得问问他的事，比如他说明天面试，就在面试后问结果"}}]
}}
规则：
- 只记以后聊天用得上的：他的经历、计划、喜好、在意的人和事、情绪起伏的原因、他明确让你记住的。寒暄、你说的话、你的建议都不记。
- 已有画像或记忆里有的不要重复记；有变化的记新的，并把旧的放进 obsolete_ids。
- 他说"忘了吧/别记这个"之类，把相关编号放进 obsolete_ids，也不要再记。
- "你"说的话不是事实来源。"你"随口说的（比如"平时都是我找你"）不能记成他的事。
- 他情绪低落时的自我否定（"我没能力""我长得不好""生活没意思"）不是事实，不要记成他的特点；最多记一条"他某月某日那天情绪低落、对未来迷茫"这样的状态，importance 不超过 2。
- importance：5=身份/重大事件，3=比较重要的近况或偏好，1=小事。
- 没有就给空数组。"""},
        ]
        data = await self.llm.chat_json(prompt, model=self.cfg.llm.memory_model, max_tokens=1200)
        added = 0
        for n in data.get("notes") or []:
            content = (n.get("content") if isinstance(n, dict) else str(n)) or ""
            if content.strip():
                self.store.add_note(content, parse_importance(n.get("importance", 2)) if isinstance(n, dict) else 2)
                added += 1
        valid_ids = {n.id for n in self.store.active_notes()}
        self.store.deactivate_notes([int(i) for i in data.get("obsolete_ids") or [] if str(i).isdigit() and int(i) in valid_ids])
        for f in data.get("followups") or []:
            try:
                due = datetime.strptime(f["due"], "%Y-%m-%d %H:%M").replace(tzinfo=self.tz).timestamp()
            except (KeyError, ValueError, TypeError):
                continue
            if f.get("topic"):
                self.store.add_followup(due, f["topic"])
        self.store.set("extract_cursor", msgs[-1].id)
        log.info("记忆提炼：新增 %d 条", added)
        return added

    # ================= 每日整理核心画像 =================
    async def consolidate(self) -> bool:
        last = self.store.get("consolidated_at", 0)
        new_notes = [n for n in self.store.notes_since(last) if n.importance >= 2]
        if not new_notes:
            self.store.set("consolidated_at", time.time())
            return False
        old = self.profile()
        notes_txt = "\n".join(f"- ({datetime.fromtimestamp(n.created_at, self.tz):%Y-%m-%d}) {n.content}" for n in new_notes)
        limit = self.cfg.memory.profile_max_chars
        prompt = [
            {"role": "system", "content": "你负责维护一个 AI 朋友对用户的核心画像。只写用户说过的事实，不做心理分析，不下性格结论。"},
            {"role": "user", "content": f"""现有核心画像：
{old or '（无）'}

最近新增的记忆：
{notes_txt}

把新记忆里长期有效的部分合并进画像，输出完整的新画像。要求：
- 只放长期有效、每次聊天都值得知道的：他是谁、做什么、重要的人、长期的喜好和在意的事、正在进行的大事。
- 一次性的小事、已经过去的事不要放（它们留在零碎记忆里，需要时会被检索出来）。
- 和新信息冲突的旧内容要更新。
- 用简洁的条目，以"- "开头，可以用几个小标题分组。总长度不超过 {limit} 字。
- 只输出画像正文。"""},
        ]
        res = await self.llm.chat(prompt, model=self.cfg.llm.memory_model, temperature=0.2, max_tokens=1500)
        text = res.text.strip()
        if not text:
            return False
        p = self.cfg.profile_file
        if p.exists():
            bak = self.cfg.data_dir / "profile_history"
            bak.mkdir(exist_ok=True)
            shutil.copy(p, bak / f"profile-{datetime.now(self.tz):%Y%m%d-%H%M}.md")
        p.write_text(text + "\n", encoding="utf-8")
        self.store.set("consolidated_at", time.time())
        log.info("核心画像已整理（%d 字）", len(text))
        return True
