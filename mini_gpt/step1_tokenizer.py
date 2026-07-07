# -*- coding: utf-8 -*-
"""
ステップ1: トークナイザ 〜 LLMは文字を「数字」として読む
==========================================================
LLMは文字をそのまま扱えない。すべてを数字(トークンID)に変換してから計算する。
この変換器を「トークナイザ」と呼ぶ。

ここでは一番シンプルな「1文字 = 1トークン」方式(文字レベル)を自作する。
ChatGPTなどの本物は「よく出る文字のかたまり = 1トークン」(BPE)だが、原理は同じ。

実行方法:
    cd mini_gpt
    python step1_tokenizer.py
"""
from pathlib import Path

# ---- 1. テキストを読み込む ----
text = (Path(__file__).parent / "data" / "input.txt").read_text(encoding="utf-8")
print(f"テキスト全体: {len(text):,} 文字")
print("--- 最初の120文字 ---")
print(text[:120])
print()

# ---- 2. 語彙(ボキャブラリ)を作る ----
# テキストに登場する文字を全部集めて、五十音・記号順に並べる。
# この「登場する文字の一覧」が、このLLMが知っている全世界になる。
chars = sorted(set(text))
vocab_size = len(chars)
print(f"文字の種類(語彙サイズ): {vocab_size}")
print("語彙の一部:", "".join(chars[100:140]))
print()

# ---- 3. 文字 <-> 数字 の対応表を作る ----
stoi = {ch: i for i, ch in enumerate(chars)}  # string to int: 「あ」-> 123
itos = {i: ch for ch, i in stoi.items()}      # int to string: 123 -> 「あ」


def encode(s: str) -> list[int]:
    """文字列 -> トークンIDのリスト"""
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    """トークンIDのリスト -> 文字列"""
    return "".join(itos[i] for i in ids)


# ---- 4. 動作確認 ----
sample = "吾輩は猫である"
ids = encode(sample)
print(f"「{sample}」をエンコード -> {ids}")
print(f"デコードして元に戻す   -> 「{decode(ids)}」")

# まとめ:
#   LLMに入っていくのは文字ではなく、この数字の列。
#   LLMから出てくるのも数字の列で、最後にdecodeして人間が読める文字に戻す。
#   次のステップ2では、この数字の列から「次の文字を予測する」モデルを作る。
