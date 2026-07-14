# 分身3号のコード全解説 — 1行ずつ、初学者向け

学習ノートブック(`bunshin_data/colab/bunshin_lora_kaggle.ipynb`)のコードを上から順に、
1行ずつ解説する。これを読み終わると「自分が何を動かしたのか」が全部わかる。

---

## セル1: 道具のインストール

```python
%%capture
!pip install unsloth
```

- `%%capture` … このセルの出力(大量のインストールログ)を画面に出さない、Jupyterの魔法コマンド
- `!` … 「ここからはPythonではなくパソコンへの命令(シェルコマンド)」の印
- `pip install unsloth` … LoRA学習を高速・省メモリでやってくれるライブラリを入れる。料理で言えば圧力鍋の購入

---

## セル2: データの確認

```python
from pathlib import Path
hits = list(Path("/kaggle/input").rglob("dataset.jsonl"))
assert hits, "右側の Input → Upload から dataset.jsonl をアップロードしてね"
DATA_PATH = hits[0]
SAVE_DIR = Path("/kaggle/working")
```

- `from pathlib import Path` … ファイルの場所を扱う道具を取り出す
- `rglob("dataset.jsonl")` … `/kaggle/input`フォルダの中を**再帰的に**(サブフォルダの奥まで)探して、その名前のファイルを全部見つける。`r`はrecursive(再帰)のr
- `list(...)` … 見つけた結果をリスト(配列)に変換
- `assert hits, "メッセージ"` … 「hitsが空っぽなら、このメッセージを出して即停止しろ」という安全装置。**早く・分かりやすく失敗させる**のが良いコードの作法
- `hits[0]` … リストの0番目(=最初に見つかったファイル)。プログラミングは0から数える
- `SAVE_DIR` … 成果物の保存先。Kaggleでは`/kaggle/working`に置いたものだけが持ち帰れる

---

## セル3: モデルの読み込みとLoRAの取り付け(心臓部その1)

```python
from unsloth import FastLanguageModel

max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-9B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
```

- `max_seq_length = 1024` … 一度に読める文章の長さ(トークン数)。分身の「短期記憶の容量」
- `from_pretrained(...)` … 「事前学習済み(pretrained)のモデルをネットから取ってくる」。90億個のダイヤルがすでに日本語向けに調整済みの脳をダウンロードしている
- `model, tokenizer = ...` … 戻り値が2つ。`model`=脳本体、`tokenizer`=文字と数字の翻訳辞書(mini_gptで自作したアレの本格版)。**脳と辞書は必ずペアで使う**
- `load_in_4bit=True` … 通常32bitで持つ数値を4bitに圧縮して持つ(QLoRAの「Q」)。精度をわずかに犠牲に、メモリを約8分の1へ。これで無料GPU(16GB)に9Bが載る

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

- `get_peft_model(...)` … 脳本体を**凍結**し(90億個のダイヤルは触らない)、横に小さな「差分メモ帳」を取り付ける。学習で書き換わるのはこのメモ帳だけ。これが**LoRA**
- `r=16` … メモ帳の厚さ(ランク)。大きいほど表現力が上がるが、過学習もしやすくなる。16は定番の値
- `lora_alpha=16` … メモ帳の内容を本体にどれくらい強く効かせるかの倍率
- `lora_dropout=0` … 学習中にわざと一部を無効にする保険(mini_gptで学んだdropout)。今回はデータが少ないので0
- `target_modules=[...]` … メモ帳をどの部品に貼るか。`q/k/v/o_proj`は**自己注意機構**(mini_gptで自作したQ・K・V!)、`gate/up/down_proj`は全結合層。つまり「考える部品ぜんぶ」
- `use_gradient_checkpointing="unsloth"` … 計算の途中メモを保存せず、必要になったら計算し直す省メモリ術。時間と引き換えにメモリを節約
- `random_state=1225` … 乱数のタネ。同じタネなら同じ結果になり、実験がやり直せる(再現性)

---

## セル4: 教科書づくり(心臓部その2)

```python
import json
from datasets import Dataset

PROFILE = "あなたは「息吹(いぶき)」、21歳の日本人大学生。"
```

- `json` … データ交換の世界共通フォーマットを読む道具
- `Dataset` … 学習データを効率よく扱うHugging Face製の入れ物
- `PROFILE` … 分身の自己紹介。全学習例の冒頭に付く「役作りの台本」

```python
def to_text(messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
```

- `def to_text(messages):` … 「会話リスト→学習用テキスト」への変換関数を定義
- `apply_chat_template(...)` … モデルごとに決まっている**台本の書式**に変換する。Qwenの場合、内部では `<|im_start|>user\nこんにちは<|im_end|>` のような特殊記号付きテキストになる。この書式を間違えるとモデルは誰の発言か分からなくなる
- `tokenize=False` … まだ数字にはせず、テキストのままもらう
- `enable_thinking=False` … Qwen3.5の「考えてから答える」モードをオフ(分身は反射で返事するタイプなので)
- `try/except TypeError` … 古いバージョンのライブラリだと`enable_thinking`引数が存在せずエラーになるので、その場合は引数なしでやり直す保険

```python
rows = []
with DATA_PATH.open(encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        rel = d.get("relation", "知り合い")
        system = PROFILE + f"相手は{rel}。息吹本人として、息吹のいつもの口調・文体で返信する。"
        msgs = [{"role": "system", "content": system}] + d["messages"]
        rows.append({"text": to_text(msgs)})
```

- `with ... as f:` … ファイルを開き、ブロックを抜けたら自動で閉じる安全な書き方
- `for line in f:` … JSONLは「1行=1つの学習例」なので、1行ずつ処理
- `json.loads(line)` … 文字列をPythonの辞書(dict)に変換
- `d.get("relation", "知り合い")` … relationラベルを取り出す。**無かった場合は"知り合い"を使う**(`.get`の第2引数は「デフォルト値」)。エラーで止まらない書き方
- `f"相手は{rel}。..."` … f-stringという文字列合成。`{rel}`の部分に変数の中身が埋まる
- `[{"role": "system", ...}] + d["messages"]` … システム指示(役作り)を会話の先頭に足す。**これが「相手によって話し方を切り替える」仕掛けの正体** — 学習中ずっと「相手は母」「相手は友達」を見せられるので、モデルがラベルと口調の対応を学ぶ
- `rows.append(...)` … できあがった1例をリストに追加

```python
dataset = Dataset.from_list(rows).shuffle(seed=1225)
split = dataset.train_test_split(test_size=0.05, seed=1225)
train_ds, eval_ds = split["train"], split["test"]
```

- `.shuffle(seed=1225)` … 順番をシャッフル。父との会話が固まって並んでいると学習が偏るため
- `train_test_split(test_size=0.05)` … 5%を**検証用**に取り分ける。検証用は学習に使わず「初見の問題での実力テスト」に使う。過学習・学習不足はこのテストの点数(eval_loss)で見抜く

---

## セル5: 学習(心臓部その3)

```python
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        output_dir="outputs",
        seed=1225,
        report_to="none",
    ),
)
```

- `SFTTrainer` … 教師ありファインチューニング(SFT)の学習ループを全部やってくれる管理人。mini_gptで手書きした「予測→損失→逆伝播→更新」のプロ版
- `per_device_train_batch_size=2` … 一度に2例ずつ解かせる
- `gradient_accumulation_steps=4` … 4回ぶんの採点結果をためてからダイヤルを回す。2×4=**実質8例ずつ**学習するのと同じ。メモリが足りないときの定番テクニック
- `num_train_epochs=5` … 教科書を5周(→結果的に3周が最適と判明したのは記録の通り)
- `learning_rate=2e-4` … 1回の修正でダイヤルをどれくらい大胆に回すか。2e-4は0.0002のこと。大きすぎると暴走、小さすぎると進まない
- `lr_scheduler_type="cosine"` … 学習率を後半ゆるやかに下げていく(最初は大胆に、仕上げは慎重に)
- `warmup_steps=20` … 最初の20ステップは学習率を徐々に上げる準備運動。いきなり全力だと壊れやすい
- `eval_strategy="steps", eval_steps=50` … 50ステップごとに検証データで抜き打ちテスト。あの「eval_lossの表」はこれが出していた
- `output_dir="outputs"` … 途中経過(checkpoint)の保存先。**5周して過学習しても3周時点を取り出せた**のはこのおかげ
- `report_to="none"` … 外部の記録サービスに送信しない

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
trainer.train()
```

- `train_on_responses_only(...)` … 「**相手の発言部分は採点しない。息吹の返事部分だけを学習しろ**」という指示(損失マスキング)。これが無いと、モデルは友達のセリフの真似まで学習してしまう
- `instruction_part / response_part` … 台本のどこからが相手で、どこからが自分かの目印(Qwenの特殊記号)
- `trainer.train()` … 発射ボタン。ここから数時間、885ステップの学習が走る

---

## セル7: 分身と話す(推論)

```python
FastLanguageModel.for_inference(model)
```

- 学習モード(採点の準備をしながら動く)から**会話専用モード**(予測だけを高速に)へ切り替え

```python
def talk(message, relation="友達(高校からの友達)"):
    system = PROFILE + f"相手は{relation}。息吹本人として、息吹のいつもの口調・文体で返信する。"
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
```

- 学習時とまったく同じ書式で台本を作る(**学習と推論で書式を揃えるのが鉄則**)
- `add_generation_prompt=True` … 台本の最後に「次はassistant(あなた)の番ですよ」という合図を付ける。学習時はFalse(正解の返事も台本に入っていた)、会話時はTrue(続きを書かせたい)

```python
    enc = tokenizer(text=prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda")
    attention_mask = enc["attention_mask"].to("cuda")
```

- `tokenizer(text=prompt, ...)` … 台本を数字の列(トークンID)に変換。`text=`を明示するのは、Qwen3.5が画像も読めるモデルで、渡し方が曖昧だと画像と勘違いするから(実際にやらかしたバグ)
- `return_tensors="pt"` … PyTorch形式の行列で返してもらう
- `.to("cuda")` … データをCPUからGPUへ引っ越し。モデルがGPUにいるので、データも同じ場所にないと計算できない

```python
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                         max_new_tokens=128,
                         temperature=0.7, top_p=0.9, do_sample=True)
```

- `generate(...)` … mini_gptで作った「1文字予測→追加→また予測」のループのプロ版
- `max_new_tokens=128` … 返事の長さの上限
- `temperature=0.7` … 冒険度。0だと毎回同じ優等生回答、高いと支離滅裂。0.7は「その人らしい揺らぎ」ゾーン
- `top_p=0.9` … 確率の高い候補で合計90%になるまでの単語だけからサイコロを振る(変な単語の混入防止)
- `do_sample=True` … サイコロを振る(=毎回違う返事になる)。Falseだと常に最有力候補のみ

```python
    reply = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    reply = reply.split("</think>")[-1].strip()
```

- `out[0][input_ids.shape[1]:]` … 出力には「こちらが渡した台本+新しい返事」が全部入っているので、台本の長さぶんをスキップして**新しい部分だけ**切り出す
- `decode(...)` … 数字の列を文字に戻す(encodeの逆)
- `skip_special_tokens=True` … `<|im_end|>`などの内部記号を消す
- `.split("</think>")[-1].strip()` … 思考タグが混ざったら本文だけ取り出し、前後の空白を除去

---

## 全体を一言でまとめると

```
辞書と脳を借りる → 差分メモ帳を貼る → 自分の会話を台本形式に整える
→ 自分の返事だけを採点対象に、実質8例ずつ・ダイヤルを少しずつ回す
→ 50歩ごとに抜き打ちテストで実力を測る → ベストの瞬間を保存
→ 同じ台本形式で話しかけて、続きを書かせる
```

mini_gptで手作りした部品(トークナイザ、損失、学習ループ、生成ループ)が、
そのまま業務用ライブラリに置き換わっているだけ — 原理は全部、最初に自分で作ったものだ。
