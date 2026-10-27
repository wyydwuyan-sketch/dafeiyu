"""配置加载：config.toml + 环境变量覆盖。"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-v4-flash"
    # 记忆提炼 / 整理用的模型，默认同一个；想更准可以换 pro
    memory_model: str = "deepseek-v4-flash"
    temperature: float = 1.1
    max_tokens: int = 600
    timeout_s: float = 60.0


@dataclass
class ChatConfig:
    # 用户连发几条时，等他停下来再回（秒）
    debounce_s: float = 4.0
    # 一次回复最多拆成几条气泡
    max_bubbles: int = 3
    # 每个字的"打字"耗时（秒），用来模拟真人间隔
    typing_s_per_char: float = 0.12
    # 上下文里保留的原始消息条数，超过就把最早的一批压缩成摘要
    history_max_messages: int = 40
    history_compact_batch: int = 20


@dataclass
class MemoryConfig:
    # 核心画像上限（字符）。这部分每次都会带上，越精炼越好
    profile_max_chars: int = 1500
    # 每次检索带上的情景记忆条数
    retrieve_top_k: int = 6
    # 对话空闲多久后提炼记忆（分钟）
    extract_idle_min: int = 10
    # 每天几点整理核心画像（本地时间，小时）
    consolidate_hour: int = 4


@dataclass
class ProactiveConfig:
    enabled: bool = True
    # 允许主动发消息的时段（本地时间，HH:MM）
    active_start: str = "09:00"
    active_end: str = "23:00"
    # 他多久没说话，才考虑主动找他（小时）
    min_silence_h: float = 3.0
    # 只在他最近一次说话后的这么多小时内主动发（iLink 24 小时窗口，留余量）
    window_h: float = 22.0
    # 他没回之前，最多连续主动发几次
    max_unanswered: int = 1
    # 每天最多主动几次
    daily_max: int = 3
    # 检查间隔（分钟，随机取区间内的值）
    check_min_min: int = 25
    check_max_min: int = 70
    # 一次主动最多连发几条（受平台每条消息约 10 条回复额度限制）
    max_bubbles: int = 5


@dataclass
class ILinkConfig:
    base_url: str = "https://ilinkai.weixin.qq.com"
    # 每个 context_token 最多回复几条（平台约 10 条，留余量）
    replies_per_token: int = 9
    # 发送频率限制：window_s 秒内最多 max_sends 条（平台约 7 条 / 5 分钟）
    rate_window_s: int = 300
    rate_max_sends: int = 6
    bot_agent: str = "DaFeiYu/0.1.0"


@dataclass
class Config:
    data_dir: Path = Path("data")
    persona_file: Path = Path("persona.md")
    timezone: str = "Asia/Shanghai"
    llm: LLMConfig = field(default_factory=LLMConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    ilink: ILinkConfig = field(default_factory=ILinkConfig)

    @property
    def profile_file(self) -> Path:
        return self.data_dir / "profile.md"

    @property
    def db_file(self) -> Path:
        return self.data_dir / "bot.db"

    @property
    def account_file(self) -> Path:
        return self.data_dir / "account.json"


def _apply(obj, values: dict) -> None:
    for key, val in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"未知配置项: {key}")
        cur = getattr(obj, key)
        if isinstance(val, dict):
            _apply(cur, val)
        elif isinstance(cur, Path):
            setattr(obj, key, Path(val))
        else:
            setattr(obj, key, val)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else Path("config.toml")
    if p.exists():
        with open(p, "rb") as f:
            _apply(cfg, tomllib.load(f))
        base = p.resolve().parent
        if not cfg.data_dir.is_absolute():
            cfg.data_dir = base / cfg.data_dir
        if not cfg.persona_file.is_absolute():
            cfg.persona_file = base / cfg.persona_file
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        cfg.llm.api_key = env_key
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
