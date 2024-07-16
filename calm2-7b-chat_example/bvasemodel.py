import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# バージョン確認
assert transformers.__version__ >= "4.34.1"

# モデルとトークナイザーの読み込み
model_name = "cyberagent/calm2-7b-chat-dpo-experimental"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def generate_response(prompt, max_tokens=1000, temperature=0.8):
    try:
        token_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(
            input_ids=token_ids.to(model.device),
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

def chat():
    logger.info("Chatbot is ready to receive input.")
    while True:
        try:
            user_input = input("USER: ")
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting chat.")
                break
            prompt = f"USER: {user_input}\nASSISTANT: "
            response = generate_response(prompt)
            print(f"ASSISTANT: {response}")
        except KeyboardInterrupt:
            logger.info("Chat interrupted by user.")
            break

if __name__ == "__main__":
    chat()
