# -*- coding: utf-8 -*-
"""
フェーズ3準備: 実名の伏せ字(仮名)処理
========================================
Colab(Google)にアップロードする前に、データの中の実名を自然な仮名に置き換える。
対応表は raw/name_map.json(自由に編集してOK)。

実行方法:
    cd bunshin_data
    python anonymize.py

出力(output/anon/ フォルダ):
    dataset.jsonl     … 仮名化された会話ペア(Colabにはこちらを上げる)
    style_corpus.txt  … 仮名化された文体コーパス

※本人の名前(息吹)はあえて残している。分身が自分の名前を知っているべきだから。
  隠したい場合は name_map.json に "息吹": "○○" を追加すればよい。
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    name_map = json.loads((HERE / "raw" / "name_map.json").read_text(encoding="utf-8"))
    # 「山田 太郎」を「山田」より先に置換するため、長い名前から順に処理する
    names = sorted(name_map.keys(), key=len, reverse=True)

    out_dir = HERE / "output" / "anon"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {n: 0 for n in names}

    def mask(text: str) -> str:
        for n in names:
            if n in text:
                counts[n] += text.count(n)
                text = text.replace(n, name_map[n])
        return text

    # ---- 1. 会話ペア ----
    src = (HERE / "output" / "dataset.jsonl").read_text(encoding="utf-8")
    with (out_dir / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for line in src.splitlines():
            d = json.loads(line)
            for m in d["messages"]:
                m["content"] = mask(m["content"])
            if "partner" in d:
                d["partner"] = mask(d["partner"])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- 2. 文体コーパス ----
    corpus = (HERE / "output" / "style_corpus.txt").read_text(encoding="utf-8")
    (out_dir / "style_corpus.txt").write_text(mask(corpus), encoding="utf-8")

    # ---- 結果と安全確認 ----
    print("置き換えた回数:")
    for n in names:
        if counts[n]:
            print(f"  {n} → {name_map[n]}: {counts[n]}回")

    # 元の名前が残っていないか最終チェック
    check = (out_dir / "dataset.jsonl").read_text(encoding="utf-8") + \
            (out_dir / "style_corpus.txt").read_text(encoding="utf-8")
    leftover = [n for n in names if n in check and name_map[n] != n
                and n not in name_map.values()]
    if leftover:
        print(f"⚠ まだ残っている名前: {leftover}")
    else:
        print("✅ 元の名前はすべて仮名になりました")
    print(f"出力先: {out_dir}")


if __name__ == "__main__":
    main()
