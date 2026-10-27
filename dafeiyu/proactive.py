"""主动发消息的判断逻辑（纯函数，方便测试）。

iLink 平台限制：
- 只有他在 24 小时内给 bot 发过消息，bot 发的消息才会送达（超时会被静默丢弃）
- 每个 context_token 能回复的条数有限（约 10 条）
所以"主动找你"只能发生在他最近说过话的窗口里，且要给后续回复留够额度。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from .config import ProactiveConfig
from .store import Followup


@dataclass
class ProactiveState:
    now: float
    last_user_ts: float | None
    last_activity_ts: float | None  # 任何一方最后一条消息
    unanswered: int  # 他上次说话后，你已经主动发了几次
    sent_today: int
    replies_left: int  # 当前 context_token 还能发几条
    followups: list[Followup]


@dataclass
class Decision:
    ok: bool
    reason: str
    situation: str = ""


def _in_active_hours(now: float, cfg: ProactiveConfig, tz: ZoneInfo) -> bool:
    t = datetime.fromtimestamp(now, tz).time()
    start = dtime.fromisoformat(cfg.active_start)
    end = dtime.fromisoformat(cfg.active_end)
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end  # 跨午夜的时段


def _ago(seconds: float) -> str:
    h = seconds / 3600
    if h < 1:
        return f"{int(seconds // 60)} 分钟"
    if h < 24:
        return f"{h:.0f} 小时"
    return f"{h / 24:.1f} 天"


def decide(state: ProactiveState, cfg: ProactiveConfig, tz: ZoneInfo, min_replies_left: int = 3) -> Decision:
    if not cfg.enabled:
        return Decision(False, "未开启")
    if not _in_active_hours(state.now, cfg, tz):
        return Decision(False, "不在允许时段")
    if state.last_user_ts is None:
        return Decision(False, "他还没发过消息")
    since_user = state.now - state.last_user_ts
    if since_user > cfg.window_h * 3600:
        return Decision(False, "超出 24 小时窗口，发了也收不到")
    if state.unanswered >= cfg.max_unanswered:
        return Decision(False, "上次主动找他还没回")
    if state.sent_today >= cfg.daily_max:
        return Decision(False, "今天主动次数用完了")
    if state.replies_left < min_replies_left:
        return Decision(False, "回复额度不够，留给他回话后用")
    silence = state.now - (state.last_activity_ts or state.last_user_ts)
    due = [f for f in state.followups if f.due_at <= state.now]
    need_silence = 0.5 * 3600 if due else cfg.min_silence_h * 3600
    if silence < need_silence:
        return Decision(False, "刚聊过不久")

    lines = [f"距离你们上一条消息已经过了 {_ago(silence)}，他上次说话是 {_ago(since_user)} 前。"]
    if due:
        lines.append("你之前惦记着要问问他的事：")
        lines += [f"- {f.topic}" for f in due]
    return Decision(True, "可以", "\n".join(lines))
