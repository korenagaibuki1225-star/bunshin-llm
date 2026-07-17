# 分身3号のコード全解説 — 一行ずつ、初学者向けに

学習ノートブック(`bunshin_data/colab/bunshin_lora_kaggle.ipynb`)の中身を、
上から順に一行ずつ解説します。「なぜその行が必要なのか」まで書きます。

---

## セル1: ライブラリのインストール

```python
!pip install unsloth
```
- `!` … ノートブックで「Pythonではなくパソコンへの命令」を実行する記号
- `pip install` … Pythonの道具(ライブラリ)をネットから取り寄せてインストールする命令
- `unsloth` … LoRA学習を高速・省メモリでやってくれる道具。これのおかげで無料GPUで9Bモデルが学習できる

---

## セル2: データの確認

```python
from pathlib import Path
```
- 「`pathlib`という道具箱から`Path`(ファイルの住所を扱う道具)だけ取り出して使う」という宣言

```python
hits = list(Path("/kaggle/input").rglob("dataset.jsonl"))
```
- `/kaggle/input` … Kaggleでは、接続したデータセットが必ずこのフォルダに現れる
- `.rglob("dataset.jsonl")` … そのフォルダの中を**総なめ検索**して`dataset.jsonl`を探す(rはrecursive=フォルダの奥まで、の意味)
- `list(...)` … 検索結果を一覧表にする

```python
assert hits, "dataset.jsonl が見つからない!"
```
- `assert` … 「これが成り立たなければ、ここで止まってメッセージを出せ」という**安全装置**。
  後の工程で意味不明なエラーになる前に、分かりやすい言葉で早めに死ぬのが良いプログラムの作法

---

## セル3: モデルの読み込みとLoRA化(いちばん大事)

```python
from unsloth import FastLanguageModel
```
- Unslothの道具箱から、モデルを扱う`FastLanguageModel`を取り出す

```python
max_seq_length = 1024
```
- 一度に読み書きできる文章の長さの上限(単位はトークン≒単語のかけら)。
  長くするほど賢く文脈を読めるが、GPUメモリを食う。1024はLINE会話には十分

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-9B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
```
- `from_pretrained` … 「**学習済み**のモデルをネット(Hugging Face)から取り寄せる」。
  ゼロから作らず、日本語がすでに話せる90億パラメータのモデルを借りるのがファインチューニングの本質
- 受け取るものが2つあるので変数も2つ:
  - `model` … 脳本体(90億個の数値のかたまり)
  - `tokenizer` … 文字⇄数字の翻訳辞書(AIは数字しか読めない)
- `load_in_4bit=True` … 各数値を4bit(16段階)に**圧縮**して読み込む。
  精度をすこし犠牲にして、90億パラメータを16GBのGPUに押し込む技(QLoRAの「Q」)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=1225,
)
```
- ここが**LoRA**。90億個のダイヤルは全部凍結して、横に小さな「差分メモ帳」を貼り付け、
  学習ではメモ帳だけを書き換える
- `r=16` … メモ帳の厚さ。大きいほど表現力が上がるがデータが少ないと丸暗記しやすくなる。16は定番
- `lora_alpha=16` … メモ帳の内容をどれくらい強く本体に効かせるかの倍率
- `lora_dropout=0` … 学習中にわざと一部を無効化する保険(今回は0=使わない)
- `target_modules=[...]` … メモ帳を貼る場所のリスト。`q/k/v/o_proj`は**自己注意機構**
  (文のどこに注目するか決める部品)、`gate/up/down_proj`は**全結合層**(考える部品)。
  つまり「注目の仕方」と「考え方」の両方に差分を仕込む
- `use_gradient_checkpointing="unsloth"` … 計算の途中結果を全部覚えず、必要なとき再計算してメモリを節約する技
- `random_state=1225` … 乱数のサイコロの目を固定する「種」。これを同じにすれば同じ実験を**再現**できる

---

## セル4: 教科書(データ)の準備

```python
import json
from datasets import Dataset
```
- `json` … JSONという形式(データの書き方)を読むための標準道具
- `Dataset` … Hugging Face製の「学習データ専用の容れ物」

```python
PROFILE = "あなたは「息吹(いぶき)」、21歳の日本人大学生。"
```
- 分身の自己紹介文。あとで全学習例の頭に貼り付ける

```python
def to_text(messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
```
- `def` … 関数(何度も使う手順に名前を付けたもの)を定義する
- `apply_chat_template` … 会話のリスト(user/assistantの往復)を、モデルが学習した
  **正式な書式**の1本のテキストに変換する。裏では `<|im_start|>user` のような目印が挿入される。
  この目印がズレるとモデルは混乱するので、手作りせず必ずこの関数を使う

```python
rows = []
with DATA_PATH.open(encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
```
- `rows = []` … 空のリスト(学習例を貯めていく箱)
- `with ... as f:` … ファイルを開き、使い終わったら自動で閉じる安全な書き方
- `for line in f:` … ファイルを1行ずつ取り出して繰り返す(JSONLは1行=1データの形式)
- `json.loads(line)` … 1行の文字列を、Pythonで扱える辞書(データのかたまり)に変換

```python
        rel = d.get("relation", "知り合い")
```
- `.get(キー, 予備)` … 辞書から`relation`(相手との関係)を取り出す。無ければ「知り合い」を使う。
  `d["relation"]`と書くと無いときにエラーで死ぬが、`.get`なら死なない

```python
        system = PROFILE + f"相手は{rel}。息吹本人として、息吹のいつもの口調・文体で返信する。"
```
- `f"...{rel}..."` … **f文字列**。文字列の中に変数の中身を埋め込む書き方。
  これで各学習例に「相手は友達」「相手は母」という指示が付き、
  **相手によって話し方を切り替えること**まで学習される(このプロジェクトの発明ポイント)

```python
        msgs = [{"role": "system", "content": system}] + d["messages"]
        rows.append({"text": to_text(msgs)})
```
- 指示(system)を会話の先頭にくっつけて、正式な書式に変換し、箱に追加

```python
dataset = Dataset.from_list(rows).shuffle(seed=1225)
```
- 箱を学習用の容れ物に変換し、**シャッフル**する。データが「LINE→AI」の順に並んだまま学習すると
  変な偏りが出るため。`seed=1225`で混ぜ方も再現可能にする

```python
split = dataset.train_test_split(test_size=0.05, seed=1225)
train_ds, eval_ds = split["train"], split["test"]
```
- データの5%を**検証用**に取り分ける。学習には使わず「初見の問題を解けるか」のテスト専用。
  これが**過学習の見張り役**(3号ではこの見張りが「3周目が頂点」と教えてくれた)

---

## セル5: 学習

```python
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
```
- `SFTTrainer` … SFT(教師ありファインチューニング)を回してくれる訓練係
- `SFTConfig` … 訓練係に渡す設定用紙

```python
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_ds, eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
```
- `per_device_train_batch_size=2` … 1回の練習で2例ずつ解く(多いほど安定するがメモリを食う)
- `gradient_accumulation_steps=4` … 4回分の採点結果を貯めてからまとめて1回修正する。
  2×4=**実質8例ずつ**学習しているのと同じ効果を、少ないメモリで実現する技

```python
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
```
- `num_train_epochs` … 教科書を何周するか。**3号の実験で「このデータは3周が適量」と判明**(5周は過学習した)
- `learning_rate=2e-4` … 1回の修正の歩幅(0.0002)。大きすぎると暴れ、小さすぎると進まない
- `cosine` … 歩幅を最初は大きく、終盤はそっと小さくしていく計画表
- `warmup_steps=20` … 最初の20歩はさらに慎重に小さく(いきなり大股だと転ぶ)

```python
        eval_strategy="steps", eval_steps=50,
        seed=1225, report_to="none",
    ),
)
```
- 50歩ごとに検証テストを実施 → あの`eval_loss`の表が出る仕組み

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```
- **損失マスキング**。相手の発言部分は採点対象から外し、「息吹の返事」の部分だけを学習させる。
  これをしないと「相手の発言の予測」まで練習してしまい、相手の口調が混ざる

```python
trainer.train()
```
- 発射ボタン。ここから数時間、「予測→採点→修正」が885回繰り返される

---

## セル6: 保存

```python
model.save_pretrained(str(SAVE_DIR / "bunshin_lora"))
tokenizer.save_pretrained(str(SAVE_DIR / "bunshin_lora"))
```
- 保存されるのは差分メモ帳(LoRAアダプタ、約111MB)だけ。本体90億は保存不要
  (誰でも同じものをネットから取り寄せられるから)。「ベースモデル+差分=分身」という設計

---

## セル7: 会話(推論)

```python
FastLanguageModel.for_inference(model)
```
- モデルを「学習モード」から「会話モード」に切り替える(採点の仕組みを止めて高速化)

```python
def chat(message):
    history.append({"role": "user", "content": message})
    msgs = [{"role": "system", "content": system}] + history
```
- `history` … 会話履歴のリスト。ここに毎回追記していくことで**流れを覚えた会話**になる。
  LLM自体に記憶は無く、「毎回、会話全文を最初から読ませ直している」のが真実

```python
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
```
- `add_generation_prompt=True` … 文末に「次はassistantの番ですよ」という目印を付ける。
  これが無いとモデルは自分が話す番だと気づかない

```python
    enc = tokenizer(text=prompt, return_tensors="pt")
```
- 文章を数字の列に変換。`text=`と明示するのは、Qwen3.5が画像も読めるモデルなので
  「これは文章です」とはっきり伝えるため(ここを曖昧にしてエラーになったのが2号時代の事件)
- `return_tensors="pt"` … PyTorch形式の数字の束で返して、という指定

```python
    out = model.generate(input_ids=ids, attention_mask=mask,
                         max_new_tokens=128,
                         temperature=0.7, top_p=0.9,
                         repetition_penalty=1.1, do_sample=True)
```
- `generate` … いよいよ生成。「次の1トークン予測」を最大128回繰り返す
- `temperature=0.7` … サイコロの偏り具合。0だと毎回同じ優等生回答、高いほど冒険的。0.7は「人間らしい揺らぎ」
- `top_p=0.9` … 確率の上位90%の候補からだけ選ぶ(変すぎる単語の足切り)
- `repetition_penalty=1.1` … 同じ言葉を繰り返したら減点(「www www www」防止)
- `do_sample=True` … サイコロを振る宣言(Falseだと常に1位の単語だけ選ぶ)

```python
    reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
```
- `decode` … 数字の列を文字に戻す(翻訳辞書の逆引き)
- `out[0][ids.shape[1]:]` … 出力には「入力した文章+新しい返事」が丸ごと入っているので、
  入力の長さ分(`ids.shape[1]`)を切り捨てて**新しい返事だけ**を取り出すスライス
- `skip_special_tokens=True` … `<|im_end|>`などの内部記号を消して人間用の文章にする

```python
    history.append({"role": "assistant", "content": reply})
```
- 分身の返事も履歴に追加 → 次の発言で「さっき自分が言ったこと」を踏まえられる

---

## まとめ: 3号は結局なにをしているのか

1. 日本語ペラペラの借り物の脳(9B)を4bitに圧縮して積む
2. その脳に「差分メモ帳」を貼る(LoRA)
3. 「相手は◯◯」という指示付きの会話1,490件で、**自分の返事の部分だけ**を885回練習
4. 検証テストの成績が一番良かった瞬間のメモ帳(checkpoint-500)を採用
5. 会話時は、履歴全文+「次はあなたの番」を毎回読ませて、サイコロ付きで1トークンずつ生成

コードにすると約100行。この100行の裏に、データ整形(フェーズ2)と評価(フェーズ5)の
試行錯誤があって、初めて「ちょいまち」が生まれます。
