# -*- coding: utf-8 -*-
"""
フェーズ2: 分身の教科書づくり(v2: 文脈つき学習例)
================================
raw/ フォルダに入れた素材を、AIが学習できる形式に変換する。

  raw/line/  にLINEのトーク履歴(.txt)        → 「会話の流れ+自分の返事」の学習例
  raw/ai/    にAIとの会話エクスポート(.json) → 「AI→自分」の学習例 + 文体
  raw/style/ に自分の文章(.txt)              → 文体学習用のテキスト

実行方法:
    cd bunshin_data
    python make_dataset.py

v2の変更点(分身1号の反省から):
  ・LINEは1往復だけでなく「直前6ターンまでの流れ」を文脈として付ける
    → 会話の流れをふまえた返事ができるようになる
  ・AI由来の学習例を800組までに間引く(全体の77%を占めていて、
    分身の口調が「AIに話すときの丁寧さ」に引っぱられていたため)
"""
import json
import random
import re
from datetime import date, datetime, timedelta
from pathlib import Path

# ★★★ 自分のLINEの表示名 ★★★
MY_NAME = "あなたのLINE表示名"

# これ以上時間が空いたら「別の会話」とみなす(分)
SESSION_GAP_MINUTES = 60
# 自分の返事の前に、何ターンぶんの会話の流れを文脈として付けるか
CONTEXT_TURNS = 6
# AIとの会話から採用する学習例の上限(口調バランス調整)
AI_CHAT_MAX = 800

HERE = Path(__file__).parent

SKIP_PATTERNS = [
    r"^\[(スタンプ|写真|動画|アルバム|ファイル|ボイスメッセージ|位置情報|連絡先)\]$",
    r"^☎",
    r"メッセージの送信を取り消しました",
    r"^https?://\S+$",
]

DATE_RE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})")
MSG_RE = re.compile(r"^(午前|午後)?(\d{1,2}):(\d{2})\t([^\t]+)\t(.*)$")


def data_files(folder: Path, pattern: str):
    """フォルダ内のデータファイル一覧。本物のデータが入ったら sample_ は自動で無視"""
    files = sorted(folder.glob(pattern))
    real = [p for p in files if not p.name.startswith("sample_")]
    return real or files


# ============================================================
# LINEのトーク履歴
# ============================================================

def parse_line_export(path: Path):
    """LINEのトーク履歴(.txt)を「会話のまとまり(セッション)」のリストにする"""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    sessions, current = [], []
    current_date, last_time = None, None
    i = 0
    while i < len(lines):
        d = DATE_RE.match(lines[i])
        if d:
            current_date = date(int(d.group(1)), int(d.group(2)), int(d.group(3)))
            i += 1
            continue
        m = MSG_RE.match(lines[i])
        if not m:
            i += 1
            continue
        ampm, hour, minute, name, text = m.groups()
        hour = int(hour) % 12 + (12 if ampm == "午後" else 0)
        if text.startswith('"') and not (len(text) > 1 and text.endswith('"')):
            parts = [text[1:]]
            i += 1
            while i < len(lines) and not lines[i].endswith('"'):
                parts.append(lines[i])
                i += 1
            if i < len(lines):
                parts.append(lines[i][:-1])
            text = "\n".join(parts)
        elif text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if current_date is not None:
            t = datetime(current_date.year, current_date.month, current_date.day,
                         hour, int(minute))
            if (last_time is not None and current
                    and t - last_time > timedelta(minutes=SESSION_GAP_MINUTES)):
                sessions.append(current)
                current = []
            last_time = t
        text = text.strip()
        if text and not any(re.search(p, text) for p in SKIP_PATTERNS):
            current.append((name, text))
        i += 1
    if current:
        sessions.append(current)
    return sessions


def merge_consecutive(messages):
    """同じ人の連続メッセージを1つにまとめる(これで発言が交互になる)"""
    merged = []
    for name, text in messages:
        if merged and merged[-1][0] == name:
            merged[-1] = (name, merged[-1][1] + "\n" + text)
        else:
            merged.append((name, text))
    return merged


def build_examples(turns):
    """「会話の流れ(最大CONTEXT_TURNS) + 自分の返事」の学習例を作る。

    自分の返事1つごとに1例。返事の直前までの流れを文脈として含めるので、
    分身は「単発の反応」ではなく「流れをふまえた返事」を学べる。
    """
    examples = []
    for i, (name, _) in enumerate(turns):
        if name != MY_NAME or i == 0 or turns[i - 1][0] == MY_NAME:
            continue
        window = turns[max(0, i - CONTEXT_TURNS): i + 1]
        # 会話は必ず相手の発言から始める(先頭が自分なら削る)
        while window and window[0][0] == MY_NAME:
            window = window[1:]
        msgs = [{"role": "assistant" if n == MY_NAME else "user", "content": t}
                for n, t in window]
        if len(msgs) >= 2:
            examples.append(msgs)
    return examples


# ============================================================
# AIとの会話(ChatGPT / Claude / Gemini)
# ============================================================

def parse_chatgpt_export(data):
    conversations = []
    for conv in data:
        msgs = []
        for node in conv.get("mapping", {}).values():
            msg = node.get("message")
            if not msg:
                continue
            role = msg.get("author", {}).get("role")
            content = msg.get("content", {})
            if role not in ("user", "assistant"):
                continue
            if content.get("content_type") != "text":
                continue
            text = "\n".join(p for p in content.get("parts", [])
                             if isinstance(p, str)).strip()
            if text:
                msgs.append((msg.get("create_time") or 0, role, text))
        msgs.sort(key=lambda x: x[0])
        if msgs:
            conversations.append([(role, text) for _, role, text in msgs])
    return conversations


def parse_claude_export(data):
    conversations = []
    for conv in data:
        msgs = []
        for m in conv.get("chat_messages", []):
            role = "user" if m.get("sender") == "human" else "assistant"
            text = m.get("text") or "\n".join(
                b.get("text", "") for b in m.get("content", [])
                if b.get("type") == "text")
            text = (text or "").strip()
            if text:
                msgs.append((role, text))
        if msgs:
            conversations.append(msgs)
    return conversations


def parse_gemini_takeout(data):
    conversations = []
    for d in data:
        title = d.get("title", "")
        if title.startswith("送信したメッセージ: "):
            text = title[len("送信したメッセージ: "):].strip()
            if text:
                conversations.append([("user", text)])
    return conversations


def load_ai_chats(path: Path):
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, list) and data:
        if "mapping" in data[0]:
            return "ChatGPT", parse_chatgpt_export(data)
        if "chat_messages" in data[0]:
            return "Claude", parse_claude_export(data)
        if "header" in data[0] and "title" in data[0]:
            return "Gemini", parse_gemini_takeout(data)
    return "不明", []


def build_ai_pairs(conversations):
    """「AIの発言 → 自分の返事」ペアと、自分の発言(文体用)を取り出す"""
    pairs = []
    my_texts = []
    for msgs in conversations:
        for (prev_role, prev_text), (cur_role, cur_text) in zip(msgs, msgs[1:]):
            if prev_role == "assistant" and cur_role == "user":
                if "```" in cur_text or len(cur_text) < 8:
                    continue
                pairs.append({"prompt": prev_text[-400:], "response": cur_text})
        for role, text in msgs:
            if role == "user" and len(text) >= 20 and "```" not in text:
                my_texts.append(text)
    return pairs, my_texts


# ============================================================
# メイン処理
# ============================================================

def main():
    out_dir = HERE / "output"
    out_dir.mkdir(exist_ok=True)
    all_examples = []   # {"messages": [...], "source", "partner", "relation"}
    style_texts = []

    relations = {}
    partners_path = HERE / "raw" / "partners.json"
    if partners_path.exists():
        raw_map = json.loads(partners_path.read_text(encoding="utf-8"))
        relations = {re.sub(r"\s+", "", k): v for k, v in raw_map.items()}

    # ---- 1. LINE → 文脈つき学習例 ----
    all_names = set()
    for path in data_files(HERE / "raw" / "line", "*.txt"):
        partner = re.sub(r"^\[LINE\]\s*", "", path.stem).replace("とのトーク", "").strip()
        relation = relations.get(re.sub(r"\s+", "", partner), "不明")
        sessions = parse_line_export(path)
        n_msgs = sum(len(s) for s in sessions)
        all_names.update(name for s in sessions for name, _ in s)
        n_before = len(all_examples)
        for session in sessions:
            for msgs in build_examples(merge_consecutive(session)):
                all_examples.append({"messages": msgs, "source": "line",
                                     "partner": partner, "relation": relation})
        print(f"[LINE] {path.name}: メッセージ{n_msgs}件 "
              f"→ 学習例{len(all_examples) - n_before}件({relation})")

    if all_names and MY_NAME not in all_names:
        print(f"\n⚠ 「{MY_NAME}」が見つからない。見つかった名前: {sorted(all_names)}")

    # ---- 2. AIとの会話 → 学習例 + 文体 ----
    ai_examples = []
    for path in data_files(HERE / "raw" / "ai", "*.json"):
        service, convs = load_ai_chats(path)
        if service == "不明":
            print(f"[AI] {path.name}: 形式不明(Claudeに相談を)")
            continue
        pairs, my_texts = build_ai_pairs(convs)
        for p in pairs:
            ai_examples.append({
                "messages": [{"role": "user", "content": p["prompt"]},
                             {"role": "assistant", "content": p["response"]}],
                "source": "ai_chat", "relation": "AIアシスタント"})
        style_texts.extend(my_texts)
        print(f"[AI] {path.name}: {service}形式 / 会話{len(convs)}件 "
              f"→ ペア{len(pairs)}組 + 文体{len(my_texts)}件")

    # AI由来が多すぎると分身の口調が硬くなるので、上限まで間引く
    if len(ai_examples) > AI_CHAT_MAX:
        random.seed(1225)
        ai_examples = random.sample(ai_examples, AI_CHAT_MAX)
        print(f"→ AI由来はバランス調整のため {AI_CHAT_MAX} 件に間引いた")
    all_examples.extend(ai_examples)

    # ---- 3. 自分の文章 → 文体コーパス ----
    for path in data_files(HERE / "raw" / "style", "*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        style_texts.append(text)
        print(f"[文体] {path.name}: {len(text):,} 文字")

    # ---- 保存 ----
    dataset_path = out_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        for e in all_examples:
            row = {"messages": e["messages"], "source": e["source"]}
            if "partner" in e:
                row["partner"] = e["partner"]
            if "relation" in e:
                row["relation"] = e["relation"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    style_corpus = "\n\n".join(style_texts)
    (out_dir / "style_corpus.txt").write_text(style_corpus, encoding="utf-8")

    # ---- 結果まとめ ----
    n_line = sum(1 for e in all_examples if e["source"] == "line")
    n_ai = sum(1 for e in all_examples if e["source"] == "ai_chat")
    avg_turns = (sum(len(e["messages"]) for e in all_examples if e["source"] == "line")
                 / max(n_line, 1))
    print("-" * 40)
    print(f"学習例: {len(all_examples):,} 件(LINE {n_line} / AI {n_ai})→ {dataset_path}")
    print(f"LINE学習例の平均ターン数: {avg_turns:.1f}(文脈つき)")
    rel_counts = {}
    for e in all_examples:
        rel = e.get("relation", "不明")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    print("関係ごとの内訳:")
    for rel, n in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {n}件")
    print(f"文体コーパス: {len(style_corpus):,} 文字 → {out_dir / 'style_corpus.txt'}")


if __name__ == "__main__":
    main()
