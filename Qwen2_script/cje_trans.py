from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import os

# 環境変数の設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# モデルとトークナイザーの読み込み
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(model, tokenizer, prompt, device, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_new_tokens=max_tokens,  # 最大トークン数を設定
                num_beams=2,  # ビームサーチを使用してメモリを節約
                early_stopping=True
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            # メモリクリア
            torch.cuda.empty_cache()
            gc.collect()
            # 再試行
            outputs = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_new_tokens=max_tokens // 2,  # トークン数を減らして再試行
                num_beams=2, 
                early_stopping=True
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def simulate_conversation(topic):
    conversation_history = []
    
    if is_japanese(topic):
        agent_a_prompt = f"議題: {topic}\n意見を中国語で述べてください。"
        agent_b_prompt = f"議題: {topic}\n意見を英語で述べてください。"
    else:
        agent_a_prompt = f"Topic: {topic}\nProvide your opinion in Japanese."
        agent_b_prompt = f"Topic: {topic}\nProvide your opinion in Japanese."

    # 初回のプロンプト
    conversation_history.append(f"User: {topic}")
    response_a = generate_response(model, tokenizer, agent_a_prompt, device)
    conversation_history.append(f"Agent A: {response_a}")

    for i in range(3):  # 会話回数を3回に減らす
        agent_b_input = f"Agent A: {response_a}\nAgent B: "
        response_b = generate_response(model, tokenizer, agent_b_input, device)
        conversation_history.append(f"Agent B: {response_b}")

        agent_a_input = f"Agent B: {response_b}\nAgent A: "
        response_a = generate_response(model, tokenizer, agent_a_input, device)
        conversation_history.append(f"Agent A: {response_a}")

        # 各ラウンド後にメモリをクリア
        torch.cuda.empty_cache()
        gc.collect()

    return conversation_history

def is_japanese(text):
    return any('\u3000' <= char <= '\u303F' or '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' for char in text)

def create_report(conversation_history):
    title = "議事録"
    topic = conversation_history[0].replace("User: ", "")
    proceedings = "\n".join(conversation_history[1:])
    summary_prompt = f"以下の会話を基に、AIエージェントAとBの議論をまとめてください。\n\n{proceedings}\n\nレポートを次の形式で作成してください：\n1. タイトル\n2. 議題内容\n3. 決定内容\n4. まとめ"
    summary = generate_response(model, tokenizer, summary_prompt, device, max_tokens=300)

    report = f"1. タイトル: {title}\n2. 議題内容: {topic}\n3. 決定内容:\n{proceedings}\n4. まとめ:\n{summary}"
    return report

# ユーザーから議題を入力してもらう
topic = input("議題を入力してください: ")

# 会話のシミュレーション
conversation_history = simulate_conversation(topic)

# 議事録の作成
report = create_report(conversation_history)

# 議事録の出力
print(report)
