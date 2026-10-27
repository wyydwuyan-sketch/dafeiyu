"""命令行入口。

    python -m dafeiyu login      扫码绑定微信 ClawBot
    python -m dafeiyu run        启动 bot
    python -m dafeiyu prompt     打印当前会发给模型的 system prompt（调人设用）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import load_config


def _show_qr(url: str) -> None:
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        pass
    print(f"\n用手机微信扫上面的二维码（扫不了就用微信打开这个链接）：\n{url}\n")


async def _ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip()


async def cmd_login(cfg) -> None:
    from .ilink import ILinkClient
    client = ILinkClient(cfg.ilink.base_url, bot_agent=cfg.ilink.bot_agent)
    try:
        acc = await client.login(_show_qr, _ask)
    finally:
        await client.aclose()
    cfg.account_file.write_text(json.dumps(acc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    cfg.account_file.chmod(0o600)
    print(f"绑定成功，凭据已保存到 {cfg.account_file}（别泄露这个文件）。现在可以运行：python -m dafeiyu run")


async def cmd_run(cfg) -> None:
    from .bot import Bot
    from .ilink import Account, ILinkClient
    from .llm import LLM
    from .store import Store
    if not cfg.account_file.exists():
        sys.exit("还没绑定微信，先运行：python -m dafeiyu login")
    acc = Account.from_dict(json.loads(cfg.account_file.read_text(encoding="utf-8")))
    ilink = ILinkClient(acc.base_url, token=acc.bot_token, bot_agent=cfg.ilink.bot_agent)
    llm = LLM(cfg.llm)
    bot = Bot(cfg, acc, ilink, llm, Store(cfg.db_file))
    try:
        await bot.run()
    finally:
        await ilink.aclose()


def cmd_prompt(cfg) -> None:
    from .memory import Memory
    from .store import Store

    class _NoLLM:  # 只拼 prompt，不调模型
        pass
    print(Memory(cfg, Store(cfg.db_file), _NoLLM()).system_prompt())


def main() -> None:
    ap = argparse.ArgumentParser(prog="dafeiyu", description="微信里的 DeepSeek 大肥鱼")
    ap.add_argument("command", choices=["login", "run", "prompt"])
    ap.add_argument("-c", "--config", default="config.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.command == "login":
        asyncio.run(cmd_login(cfg))
    elif args.command == "run":
        try:
            asyncio.run(cmd_run(cfg))
        except KeyboardInterrupt:
            print("\n下线了。")
    else:
        cmd_prompt(cfg)


if __name__ == "__main__":
    main()
