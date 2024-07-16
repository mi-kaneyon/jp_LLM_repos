import requests
from bs4 import BeautifulSoup
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import logging
from torch.cuda.amp import autocast
import csv
import os
from datetime import datetime
import subprocess

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# バージョン確認
assert transformers.__version__ >= "4.34.1"

# モデルとトークナイザーの読み込み
model_name = "cyberagent/calm2-7b-chat-dpo-experimental"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 会話履歴の保存ファイル
history_file = "chat_history.csv"

def save_history(user_input, bot_response):
    with open(history_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_input, bot_response])

def load_history():
    if not os.path.exists(history_file):
        return []
    with open(history_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return list(reader)

def search_history(user_input, history):
    for user_query, response in history:
        if user_input.lower() in user_query.lower():
            return response
    return None

def google_search_summary(query):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    summaries = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')
    return [summary.get_text() for summary in summaries[:5]]  # 上位5件のサマリーを返す

def generate_response(prompt, max_tokens=500, temperature=0.8):
    try:
        token_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        # Mixed Precisionモードの利用
        with autocast():
            output_ids = model.generate(
                input_ids=token_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                streamer=streamer,
            )
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error generating response."
    finally:
        # 未使用メモリの解放
        torch.cuda.empty_cache()

def fact_check(response, summaries):
    for summary in summaries:
        if summary.lower() in response.lower():
            return True
    return False

def integrate_search_results(response, search_results):
    integrated_response = f"{response}\n\nAccording to the information found online:\n"
    for result in search_results:
        integrated_response += f"- {result}\n"
    return integrated_response

def generate_contextual_response(history, user_input):
    context = ""
    for h in history[-5:]:  # 直近5件の履歴を文脈として使用
        context += f"USER: {h[0]}\nASSISTANT: {h[1]}\n"
    context += f"USER: {user_input}\nASSISTANT: "
    response = generate_response(context)
    return response

def handle_special_questions(user_input, current_date):
    if "今日の日付" in user_input or "今日の日時" in user_input:
        return f"今日は{current_date}です。"
    if "今日の天気" in user_input:
        return get_weather()
    return None

def get_weather():
    # OpenWeatherMap APIを使用する例。APIキーを取得して置き換えてください。
    api_key = "YOUR_OPENWEATHERMAP_API_KEY"
    location = "Tokyo,JP"  # 必要に応じて場所を変更
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&lang=ja&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"今日の{location}の天気は{weather}で、気温は{temp}度です。"
    else:
        return "天気情報を取得できませんでした。"

def get_current_date():
    result = subprocess.run(["date", "+%Y年%m月%d日"], capture_output=True, text=True)
    return result.stdout.strip()

def chat():
    logger.info("Chatbot is ready to receive input.")
    history = load_history()
    current_date = get_current_date()
    while True:
        try:
            user_input = input("USER: ")
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting chat.")
                break

            # 特定の質問に対する事前処理
            special_response = handle_special_questions(user_input, current_date)
            if special_response:
                print(f"ASSISTANT: {special_response}")
                save_history(user_input, special_response)
                history.append([user_input, special_response])
                continue
            
            cached_response = search_history(user_input, history)
            if cached_response:
                print(f"ASSISTANT (from history): {cached_response}")
            else:
                response = generate_contextual_response(history, user_input)
                search_results = google_search_summary(user_input)
                
                if search_results:
                    fact_checked = fact_check(response, search_results)
                    if fact_checked:
                        print(f"ASSISTANT (fact-checked): {response}")
                    else:
                        integrated_response = integrate_search_results(response, search_results)
                        print(f"ASSISTANT: {integrated_response}")
                else:
                    print(f"ASSISTANT: {response}")

                save_history(user_input, response)
                history.append([user_input, response])
            
        except KeyboardInterrupt:
            logger.info("Chat interrupted by user.")
            break

if __name__ == "__main__":
    chat()
