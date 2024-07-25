import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True).cuda()

model = model.eval()

def chat_with_bot():
    history = []  # ここで初期化
    with torch.no_grad():  # 勾配計算を無効にする
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'q':
                print("Conversation ended.")
                break
            response, history = model.chat(tokenizer, user_input, history=history)
            print(f"Bot: {response}")

def stream_chat_with_bot():
    history = []  # ここで初期化
    with torch.no_grad():  # 勾配計算を無効にする
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'q':
                print("Conversation ended.")
                break
            length = 0
            for response, history in model.stream_chat(tokenizer, user_input, history=history):
                print(response[length:], flush=True, end="")
                length = len(response)
            print("\n")  # 行を改行

if __name__ == "__main__":
    print("Choose chat mode: [1] Normal Chat [2] Stream Chat")
    mode = input("Enter mode (1 or 2): ")
    
    if mode == '1':
        chat_with_bot()
    elif mode == '2':
        stream_chat_with_bot()
    else:
        print("Invalid mode selected.")
