# -*- coding: utf-8 -*-
"""
ステップ2: 一番単純な言語モデル(バイグラム)を学習させる
==========================================================
LLMの本質は「直前までの文章を見て、次の文字を予測する」こと。それだけ。

ここでは最弱のモデル「バイグラム」を作る。
バイグラム = 「直前の1文字だけ」を見て次の文字を当てるモデル。
(例:「吾」の次は「輩」が来やすい、を表で覚えるだけ)

弱いモデルだが、「学習ループ」「損失」「文章生成」という
LLM学習の全部品がここに登場する。ステップ3ではモデル部分だけを賢くする。

実行方法:
    cd mini_gpt
    python step2_bigram.py   (1〜2分で終わる)
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1225)

# ---- ハイパーパラメータ(人間が決める設定値) ----
batch_size = 32      # 一度に並行して学習する文章の本数
block_size = 8       # 一度に見る文脈の長さ(文字数)
max_iters = 3000     # 学習の繰り返し回数
lr = 1e-2            # 学習率(1回の学習でどれくらい大胆に修正するか)

# ---- ステップ1と同じ:テキスト読み込みとトークナイザ ----
text = (Path(__file__).parent / "data" / "input.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# テキスト全体を1本の巨大な数字の列(テンソル)にする
data = torch.tensor(encode(text), dtype=torch.long)

# 前半90%を訓練用、後半10%を検証用に分ける。
# 検証用 = 「丸暗記ではなく本当に言葉の規則を学べたか」をテストするための未見データ。
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split: str):
    """テキストのランダムな場所から、学習用の問題(x)と答え(y)のセットを取り出す。

    x が「ここまでの文章」、y が「その次に来る正解の文字」。
    y は x を1文字ずらしただけ。つまり訓練データは無限に作れる!
    """
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):
    """「直前の1文字」から「次の文字の出やすさ」を引くだけの表。

    nn.Embedding は (語彙サイズ × 語彙サイズ) の巨大な表で、
    行 = 今の文字、列 = 次に来る各文字のスコア(ロジット)。
    学習でこの表の数値がどんどん調整されていく。
    """

    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # (バッチ, 文脈長, 語彙) のスコア表
        if targets is None:
            return logits, None
        # 損失(loss) = 「正解の文字にどれだけ低いスコアを付けてしまったか」。
        # これが小さくなるように表を修正するのが「学習」。
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """文章生成 = 予測した確率でサイコロを振って1文字選ぶ、を繰り返すだけ"""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # 最後の文字の予測だけ使う
            next_id = torch.multinomial(probs, num_samples=1)  # 確率でサイコロ
            idx = torch.cat([idx, next_id], dim=1)  # 選んだ文字を文脈に追加して繰り返す
        return idx


model = BigramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ---- 学習前の生成(でたらめなはず) ----
start = torch.zeros((1, 1), dtype=torch.long)  # 0番の文字からスタート
print("=== 学習前の生成(完全にランダム) ===")
print(decode(model.generate(start, 100)[0].tolist()))
print()

# ---- 学習ループ:LLM学習の心臓部 ----
for it in range(max_iters):
    x, y = get_batch("train")
    logits, loss = model(x, y)          # 1. 予測して、間違い度(損失)を測る
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                     # 2. どのパラメータをどう直すべきか計算(誤差逆伝播)
    optimizer.step()                    # 3. パラメータをちょっとだけ修正
    if it % 500 == 0 or it == max_iters - 1:
        with torch.no_grad():
            vx, vy = get_batch("val")
            _, vloss = model(vx, vy)
        print(f"iter {it:4d}: 訓練損失 {loss.item():.3f} / 検証損失 {vloss.item():.3f}")

# ---- 学習後の生成 ----
print()
print("=== 学習後の生成(日本語の雰囲気が少し出るはず) ===")
print(decode(model.generate(start, 200)[0].tolist()))

# まとめ:
#   ・損失が下がる = 次の文字の予測が上手くなっている
#   ・でも1文字前しか見ないので、文章としては支離滅裂
#   ・ステップ3では「文脈全体を見る仕組み = 自己注意機構」を追加して本物のGPTにする
