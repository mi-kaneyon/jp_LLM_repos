import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import random
import pandas as pd
from datetime import datetime
import gc
import re

# モデルの読み込み
def load_model(model_name):
    print(f"モデル '{model_name}' を読み込んでいます...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"モデル '{model_name}' の読み込みが完了しました。")
    return tokenizer, model, streamer

# GPUメモリを解放する関数
def free_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 小学1,2年生で習う漢字のリスト
grade_1_2_kanji = [
    "一", "右", "雨", "円", "王", "音", "下", "火", "花", "貝", "学", "気", "休", "金", "空", "月", "犬", "見",
    "五", "口", "校", "左", "山", "子", "四", "糸", "字", "耳", "車", "手", "十", "出", "女", "小", "上", "森",
    "人", "水", "正", "生", "青", "石", "赤", "千", "川", "先", "早", "足", "村", "大", "男", "竹", "中", "虫",
    "町", "天", "田", "土", "二", "日", "入", "年", "白", "八", "百", "文", "本", "名", "目", "立", "力", "林",
    "六", "学", "何", "作", "社", "家", "万", "海", "絵", "音", "場", "馬", "思", "姉", "島", "寺", "森", "鳥",
    "羽", "画", "工", "公", "広", "交", "光", "考", "行", "高", "黄", "合", "黒", "今", "近", "語", "午", "後",
    "行", "語", "米", "来", "楽", "間", "男", "里", "思", "会", "食", "首", "星", "雪", "黄", "森", "姉", "思",
]

# テキスト内の漢字を小学1,2年生で習う漢字のみに置き換える関数
def replace_kanji_with_hiragana(text):
    def is_valid_kanji(char):
        return char in grade_1_2_kanji

    new_text = ""
    for char in text:
        if char.isdigit() or char.isalpha():
            new_text += char
        elif re.match(r'\W', char):
            new_text += char
        elif not is_valid_kanji(char):
            new_text += f"({char})"
        else:
            new_text += char
    return new_text

# チャンクごとにテキストを生成する関数
def generate_text_in_chunks(tokenizer, model, prompt, streamer, max_length=800, chunk_size=256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"プロンプト: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    generated_text = ""

    while len(generated_text) < max_length:
        try:
            new_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=chunk_size,
                min_new_tokens=20,  # 生成が途中で止まらないようにする
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                temperature=1.0  # デフォルトの安定値
            )
            new_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)

            if new_text in generated_text:
                print("これ以上新しい生成結果がありません。終了します。")
                break

            generated_text += new_text
            input_ids = tokenizer.encode(generated_text[-chunk_size:], return_tensors="pt").to(device)
        except RuntimeError as e:
            print(f"エラーが発生しました: {e}")
            free_memory()
            break

    return generated_text[:max_length]

# CSVファイルをUTF-16エンコーディングで読み込む
csv_file = "sentences.csv"
df = pd.read_csv(csv_file, encoding='utf-16')

# 問題生成メイン関数
def generate_problems(tokenizer, model, streamer, num_problems=10):
    problems = []
    for i in range(num_problems):
        try:
            theme = df.sample(n=1).iloc[0]['sentence']
            prompt = f"{theme} このテーマに基づいて、800文字程度の物語を作成してください。"
            story = generate_text_in_chunks(tokenizer, model, prompt, streamer, max_length=800)
            story_with_hiragana = replace_kanji_with_hiragana(story)
            problems.append(story_with_hiragana)
        except Exception as e:
            print(f"問題生成中にエラーが発生しました: {e}")
            continue
    return problems

# テキストファイル出力関数
def save_problems_to_txt(problems):
    today = datetime.today().strftime('%Y%m%d')
    filename = f"problems_{today}.txt"
    with open(filename, mode='w', encoding='utf-8') as file:
        for i, problem in enumerate(problems):
            file.write(f"問題 {i + 1}\n")
            file.write(problem)
            file.write("\n\n")
    print(f"問題生成とテキストファイルへの保存が完了しました。ファイル名: {filename}")

if __name__ == "__main__":
    model_name = "cyberagent/calm2-7b-chat-dpo-experimental"
    tokenizer, model, streamer = load_model(model_name)
    free_memory()
    problems = generate_problems(tokenizer, model, streamer, 10)
    save_problems_to_txt(problems)
    free_memory()
