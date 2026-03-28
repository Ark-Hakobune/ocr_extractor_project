from __future__ import annotations


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u3000", " ").replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()


def sanitize_dialogue(text: str) -> str:
    text = normalize_text(text)
    for token in ["▼", "▽", "◆", "◇", "… …", "···"]:
        text = text.replace(token, "")
    text = text.strip("|[](){}<>“”\"'")
    return normalize_text(text)


def sanitize_speaker(text: str) -> str:
    text = normalize_text(text)
    text = text.strip("[](){}<>【】「」『』 ")
    if len(text) > 12:
        return ""
    return text


def is_ui_noise(text: str) -> bool:
    noise_keywords = {
        "自动",
        "快进",
        "存档",
        "读档",
        "设置",
        "返回标题",
        "系统",
        "回想",
        "跳过",
        "菜单",
        "config",
        "save",
        "load",
        "auto",
        "skip",
        "system",
    }
    lowered = text.lower()
    if lowered in noise_keywords:
        return True
    return any(kw in text or kw in lowered for kw in noise_keywords)


def format_record(record) -> str:
    prefix = f"[{record.timestamp_sec:>8.3f}s]"
    if record.speaker:
        return f"{prefix} [{record.speaker}] {record.text}"
    return f"{prefix} {record.text}"
