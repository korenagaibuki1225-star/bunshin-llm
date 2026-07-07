# -*- coding: utf-8 -*-
"""
ステップ0: 学習用テキストの準備
================================
青空文庫(著作権切れの小説サイト)から小説をダウンロードして、
LLMの学習に使える「きれいなテキスト」に整形し、input.txt に保存する。

実行方法:
    cd mini_gpt
    python data/prepare_data.py

※あとで自分の日記やLINEのトーク履歴に差し替えれば「自分の分身」の素材になる。
  その場合は data/input.txt を自分のテキスト(UTF-8)で上書きするだけでOK。
"""
import io
import re
import urllib.request
import zipfile
from pathlib import Path

# ダウンロードする作品リスト(すべて著作権切れ・パブリックドメイン)
WORKS = [
    ("走れメロス(太宰治)", "https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip"),
    ("吾輩は猫である(夏目漱石)", "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip"),
    ("こころ(夏目漱石)", "https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip"),
    ("銀河鉄道の夜(宮沢賢治)", "https://www.aozora.gr.jp/cards/000081/files/43737_ruby_19028.zip"),
]


def download_zip_text(url: str) -> str:
    """青空文庫のzipをダウンロードして、中のテキストを文字列で返す"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as res:
        zip_bytes = res.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # zipの中の .txt ファイルを探す
        txt_name = next(n for n in zf.namelist() if n.endswith(".txt"))
        raw = zf.read(txt_name)
    # 青空文庫はShift_JIS(cp932)で保存されている
    return raw.decode("cp932", errors="ignore")


def clean_aozora(text: str) -> str:
    """青空文庫特有の記号(ルビや注釈)を取り除いて本文だけにする"""
    # 冒頭の説明部分は「----------」の線で囲まれているので、その後ろを本文とする
    parts = re.split(r"-{20,}\r?\n", text)
    if len(parts) >= 3:
        text = parts[2]
    # 末尾の「底本:」以降は書誌情報なので削除
    text = re.split(r"\r?\n底本[::]", text)[0]
    text = re.sub(r"《[^》]*》", "", text)      # ルビ(ふりがな)を削除
    text = re.sub(r"[#[^]]*]", "", text)      # 入力者注 [#...] を削除
    text = text.replace("|", "")               # ルビの開始位置記号を削除
    text = text.replace("\r\n", "\n")
    return text.strip()


def main():
    out_path = Path(__file__).parent / "input.txt"
    chunks = []
    for title, url in WORKS:
        try:
            body = clean_aozora(download_zip_text(url))
            chunks.append(body)
            print(f"OK   {title}: {len(body):,} 文字")
        except Exception as e:
            print(f"SKIP {title}: 取得失敗 ({e})")

    full_text = "\n\n".join(chunks)
    out_path.write_text(full_text, encoding="utf-8")

    vocab = sorted(set(full_text))
    print("-" * 40)
    print(f"保存先: {out_path}")
    print(f"合計文字数: {len(full_text):,}")
    print(f"文字の種類(語彙サイズ): {len(vocab):,}")


if __name__ == "__main__":
    main()
