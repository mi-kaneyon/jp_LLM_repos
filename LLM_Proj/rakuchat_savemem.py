import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    # 環境変数の設定
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

    # モデルパスの設定
    model_path = "Rakuten/RakutenAI-7B-chat"

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # BitsAndBytesConfig の設定（4ビット量子化）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # モデルのロード
    print("モデルをロード中...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            # max_memory を設定してメモリ制限を指定（必要に応じて調整）
            max_memory={0: "5GB", "cpu": "15GB"},
        )
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return

    model.eval()
    print("モデルのロードが完了しました。")

    print("チャットボットを開始します。何か質問はありますか？(終了するには 'exit' と入力してください)")

    try:
        while True:
            user_input = input("USER: ")
            if user_input.lower() == "exit":
                print("チャットボットを終了します。")
                break

            # 入力テキストをトークナイズし、モデルのデバイスに移動
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(model.device)

            # 推論の実行
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=128,  # 必要に応じて調整
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,  # 必要に応じて調整
                    top_p=0.9,        # 必要に応じて調整
                    do_sample=True
                )

            # 応答のデコード
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"ASSISTANT: {response}")

            # メモリのクリーンアップ
            del input_ids, output_ids, response
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"エラーが発生しました: {e}")

    finally:
        # VRAMのクリーンアップ処理
        if torch.cuda.is_available():
            model.to('cpu')
            torch.cuda.empty_cache()

        gc.collect()
        print("リソースのクリーンアップが完了しました。")

if __name__ == "__main__":
    main()
