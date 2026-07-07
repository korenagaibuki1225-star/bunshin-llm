# -*- coding: utf-8 -*-
"""
ステップ3: 本物のGPTを作る 〜 自己注意機構(Self-Attention)
==========================================================
ステップ2のバイグラムは「直前の1文字」しか見られなかった。
GPTの発明は「文脈の全文字を同時に見て、どの文字に注目すべきか自分で決める」仕組み
= 自己注意機構(Self-Attention)。ChatGPTもClaudeも中身はこれの巨大版。

構造:  文字埋め込み + 位置埋め込み
        → [自己注意 → 全結合] × 層の数
        → 次の文字の予測

実行方法:
    cd mini_gpt
    python step3_gpt.py            # 学習して生成(CPUで20〜40分くらい)
    python step3_gpt.py generate   # 学習済みモデルで生成だけ(数秒)
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1225)

# ---- ハイパーパラメータ ----
batch_size = 32    # 一度に並行して学習する文章の本数
block_size = 128   # 文脈の長さ:128文字前まで見て次を予測できる
max_iters = 2000   # 学習回数(増やすほど賢くなるが時間がかかる)
eval_interval = 200
lr = 3e-4
n_embd = 128       # 1文字を表すベクトルの次元数
n_head = 4         # 注意機構のヘッド数(4つの視点で同時に文脈を見る)
n_layer = 4        # ブロックを何段重ねるか
dropout = 0.1      # 過学習(丸暗記)を防ぐため、学習中わざと一部の計算を間引く

HERE = Path(__file__).parent
CKPT_PATH = HERE / "gpt_model.pt"

# ---- トークナイザ(ステップ1と同じ) ----
text = (HERE / "data" / "input.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class Head(nn.Module):
    """自己注意機構の1ヘッド。GPTの心臓部はこのクラス。

    各文字が3つのベクトルを出す:
      Q(クエリ)  「私はこういう情報を探しています」
      K(キー)    「私はこういう情報を持っています」
      V(バリュー)「実際に渡す情報はこれです」
    QとKの相性が良い文字ほど注目され、そのVを多く受け取る。
    """

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 未来の文字はカンニング禁止(まだ書かれていないので)。そのためのマスク。
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        # 全文字ペアの「注目度スコア」を一気に計算(√で割るのはスコアの暴走防止)
        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # (B, T, T)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # 未来を隠す
        wei = F.softmax(wei, dim=-1)  # スコアを「合計1の注目度」に変換
        wei = self.dropout(wei)
        return wei @ v  # 注目度に応じて各文字の情報をブレンドして受け取る


class MultiHeadAttention(nn.Module):
    """ヘッドを複数並べる。各ヘッドが別の観点(文法・話題・人名…)を担当できる"""

    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """各文字が(注意機構で集めた情報を)一人でじっくり考えるパート"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformerブロック = 「みんなで情報交換(注意)」→「各自で熟考(全結合)」

    x + ... の足し算は「残差接続」:元の情報に修正を足していく形にすると
    深く積んでも学習が安定する、というTransformerの重要な工夫。
    """

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)  # 数値のスケールを整えて学習を安定させる
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # 文字埋め込み:各文字を「意味を持つn_embd次元のベクトル」に変換する表
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # 位置埋め込み:「文の何文字目か」の情報。これがないと語順が分からない
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # ベクトル→次の文字のスコア

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                       # 文字の意味ベクトル
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # 位置ベクトル
        x = self.blocks(tok + pos)                      # Transformer本体を通す
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # 文脈は直近block_size文字まで
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model):
    """損失はバッチごとにブレるので、50回の平均でならして測る"""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(50)
        for i in range(50):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(model, prompt="今日", tokens=300):
    # 語彙にない文字が書き出しに入っていても落ちないようにする
    ids = [stoi[c] for c in prompt if c in stoi] or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    print(decode(model.generate(idx, tokens)[0].tolist()))


def main():
    model = MiniGPT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {n_params / 1e6:.2f}M(ChatGPT級の約10万分の1)")

    # 「python step3_gpt.py generate」なら学習済みモデルを読み込んで生成だけ
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        model.load_state_dict(torch.load(CKPT_PATH))
        model.eval()
        sample(model)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = float("inf")
    for it in range(max_iters):
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            marker = ""
            # 検証損失が過去最高に良いときだけ保存する。
            # 学習が進みすぎると訓練データの「丸暗記」が始まり(過学習)、
            # 訓練損失は下がるのに検証損失が上がりだす。その前の状態を残すため。
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(model.state_dict(), CKPT_PATH)
                marker = " ★保存"
            print(
                f"iter {it:4d}: 訓練損失 {losses['train']:.3f} / "
                f"検証損失 {losses['val']:.3f} ({elapsed / 60:.1f}分経過){marker}",
                flush=True,
            )

    print(f"\nモデルを保存: {CKPT_PATH}(検証損失が最良だった時点)")
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()
    print("\n=== 生成デモ(「今日」の続きを書かせる) ===")
    sample(model)


if __name__ == "__main__":
    main()
