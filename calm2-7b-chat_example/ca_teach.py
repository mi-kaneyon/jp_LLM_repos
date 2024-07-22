import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from tqdm import tqdm

# 環境変数の設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

model_name = "cyberagent/calm2-7b-chat-dpo-experimental"
tokenizer = AutoTokenizer.from_pretrained(model_name)
language_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
language_model.to(device)

# 小学1年生向けサンプル文章
sample_texts = [
    "ある日、たろうくんはこうえんであそんでいました。そこへいぬがやってきて、いっしょにあそびました。",
    "あさ、はなちゃんはままといっしょにやさいをかいにいきました。にんじんとじゃがいもをかって、うちにかえりました。",
    "ひこうきがそらをとびました。たくさんのひとがひこうきをみあげていました。",
    "にわにさくらのきがありました。さくらのはながさいて、とてもきれいでした。",
    "くるまでうみへいきました。うみでおよいで、とてもたのしかったです。",
    "まいにちのようにあめがふっていました。そとはびしょぬれでした。",
    "やまのぼりをしていたら、ふしぎなけしきをみつけました。",
    "きょうはたんじょうびです。みんなでおいわいしました。",
    "どうぶつえんへいきました。たくさんのどうぶつにあいました。",
    "えんそくでやまにいきました。そこでたくさんのむしをみつけました。",
    "ともだちといっしょにあそびました。いっしょにえをかきました。",
    "いえでしゅくだいをしました。あとであそびにいきました。",
    "こうえんでサッカーをしました。みんなでおおさわぎしました。",
    "やさいをうえました。みんなでみずをあげました。",
    "どうぶつのなかまとあそびました。いっしょにたのしいじかんをすごしました。",
    "はるになって、はなみをしました。さくらがきれいにさいていました。",
    "おまつりにいきました。たくさんのひとがいました。",
    "やまにキャンプにいきました。たきびをしました。",
    "ともだちのいえにいきました。いっしょにゲームをしました。",
    "こうえんでブランコにのりました。たのしかったです。",
    "えほんをよみました。たくさんのおはなしがありました。",
    "えいがをみました。おもしろかったです。",
    "うちでパーティーをしました。みんながきました。",
    "いっしょにおかしをつくりました。おいしかったです。",
    "どうぶつえんでぱんだをみました。かわいかったです。",
    "うみでさかなをつりました。おおきなさかながつれました。",
    "もりのなかをさんぽしました。いろいろなはなをみつけました。",
    "えんそくでたべたおべんとうがおいしかったです。",
    "がっこうでおんがくをならいました。たのしかったです。",
    "まちでまいごになりました。でも、おまわりさんがたすけてくれました。"
]

# 問題の質問
summary_questions = [
    "この文章を一言でまとめるとどうなりますか？",
    "この物語の主な出来事は何ですか？",
    "このお話の要点を教えてください。"
]

character_questions = [
    "この物語に登場する人物を教えてください。",
    "物語の中で誰が何をしましたか？",
    "登場人物の中で一番重要なのは誰ですか？"
]

main_point_questions = [
    "この文章で一番大切なことは何ですか？",
    "この話の結論は何ですか？",
    "この物語で作者が伝えたいことは何ですか？"
]

kanji_questions = [
    "次の言葉を漢字で書いてください：",
    "この文章の中から漢字を書きましょう：",
    "次の文の中のひらがなを漢字に変えてください："
]

creation_questions = [
    "この物語の続きを書いてください。",
    "次の部分を読んで物語の続きがどうなるか考えてください。",
    "この話の次の展開を想像して書いてください。"
]

# アウトライン生成関数
def generate_outline(situation):
    prompt = f"{situation} このシチュエーションに基づいて、起承転結がしっかりとした小学1年生向けの物語のアウトラインを作成してください。"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    output_ids = language_model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_length=300, 
        pad_token_id=tokenizer.eos_token_id, 
        num_return_sequences=1, 
        temperature=0.7, 
        repetition_penalty=2.0, 
        do_sample=True,
        top_k=50, 
        top_p=0.95
    )
    outline = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return outline

# 具体的な物語生成関数
def generate_story_from_outline(outline):
    prompt = f"以下のアウトラインに基づいて、小学1年生向けの物語を作成してください：\n{outline}\n物語："
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    output_ids = language_model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id, 
        num_return_sequences=1, 
        temperature=0.7, 
        repetition_penalty=2.0, 
        do_sample=True,
        top_k=50, 
        top_p=0.95
    )
    story = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 指示文やアウトラインの痕跡を削除して物語のみを抽出
    story = story.split("物語：")[1].strip()
    return story

# 一時ファイルを使用してアウトラインから物語を生成する関数
def generate_story_with_temp_file(situation):
    outline = generate_outline(situation)
    
    # アウトラインを一時ファイルに保存
    with open("outline.txt", "w", encoding="utf-8") as f:
        f.write(outline)
    
    # 一時ファイルからアウトラインを読み込んで物語を生成
    with open("outline.txt", "r", encoding="utf-8") as f:
        outline = f.read()
    
    # 初期生成された物語
    story = generate_story_from_outline(outline)
    
    # 1000文字前後まで繰り返し修正
    for _ in tqdm(range(5)):
        if len(story) >= 900:
            break
        outline += " " + story  # 生成された物語を元にアウトラインを拡張
        story = generate_story_from_outline(outline)

    # 不要な表現を削除して、指示文が含まれないようにする
    story = story.split("以下のシチュエーション")[0]
    
    return story[:1000]  # 1000文字に切り詰める

# 問題生成メイン関数
def generate_problems(num_problems=10):
    problems = []
    for i in tqdm(range(num_problems)):
        situation = random.choice(sample_texts)
        story = generate_story_with_temp_file(situation)

        summary_question = random.choice(summary_questions)
        character_question = random.choice(character_questions)
        main_point_question = random.choice(main_point_questions)
        kanji_question = random.choice(kanji_questions)
        creation_question = random.choice(creation_questions)
        
        problems.append(f"物語:\n{story}\n\n"
                        f"1. {summary_question}\n"
                        f"2. {character_question}\n"
                        f"3. {main_point_question}\n"
                        f"4. {kanji_question}\n"
                        f"5. {creation_question}\n")
    
    return problems

# テキストファイル出力関数
def save_problems_to_txt(problems, filename="problems.txt"):
    with open(filename, mode='w', encoding='utf-8') as file:
        for i, problem in enumerate(problems):
            file.write(f"問題 {i + 1}\n")
            file.write(problem)
            file.write("\n\n")

if __name__ == "__main__":
    problems = generate_problems(1)  # 必要な問題数を指定して生成
    save_problems_to_txt(problems)
    print("問題生成とテキストファイルへの保存が完了しました。")
