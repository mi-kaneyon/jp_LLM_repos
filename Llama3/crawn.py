#!/usr/bin/env python3
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration, AutoTokenizer
from PIL import Image
import cv2
import gc

# =============================================
# 1. ローカルモデルのパス設定
# =============================================
local_model_path = "your path and model/Llama-3.2-11B-Vision"

# =============================================
# 2. Llamaモデルのロード（オフロード付き）
# =============================================
print("Loading Llama model with offload...")
model = MllamaForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",          # GPUとCPUを自動で振り分け
    offload_folder="offload",   # オフロード用の一時保存先
    torch_dtype=torch.float16   # FP16で動作（VRAM節約）
)
model.tie_weights()  # 内部重みの結合

# =============================================
# 3. 画像プロセッサーとトークナイザーのロード
# =============================================
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# =============================================
# 4. カメラから画像をキャプチャする関数
# =============================================
def capture_image():
    cap = cv2.VideoCapture(2)  # カメラIDは環境に合わせて調整
    if not cap.isOpened():
        raise Exception("カメラが開けません。")
    print("カメラが起動しました。スペースキーで撮影、qキーでキャンセルしてください。")
    captured = False
    image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # スペースキーでキャプチャ
            captured = True
            # BGR -> RGB に変換し、PIL画像に変換（明示的にRGBモードに）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame).convert("RGB")
            break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return image if captured and image is not None else None

# =============================================
# 5. 画像説明生成関数（processorとmodelを直接使用）
# =============================================
def generate_image_description(input_image):
    """
    撮影した画像と定型プロンプトを用いて、その内容を説明するテキストを生成します。
    """
    # 人物に触れずにシーン全体の特徴を具体的に描写するプロンプト
    prompt = "<|image|><|begin_of_text|>この画像に写っている背景、光の加減、物の配置、色彩など、人物に関する記述を一切せず、シーンの全体的な特徴を具体的に描写してください。"
    # 画像とプロンプトを位置引数として processor に渡す
    inputs = processor(input_image, prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    # temperature を 0.8 に設定。必要に応じて max_new_tokens やビーム数も調整してください。
    output_ids = model.generate(**inputs, max_new_tokens=200, num_beams=2, num_return_sequences=2, temperature=0.8)
    descriptions = [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in output_ids]
    return descriptions



# =============================================
# 6. メインループ（対話型）
# =============================================
def main():
    print("=== 画像キャプチャと内容説明生成サンプル ===")
    while True:
        command = input("Enterキーで撮影、qで終了: ")
        if command.lower().strip() == "q":
            print("終了します。")
            break
        try:
            print("→ カメラを起動します...")
            captured_image = capture_image()
            if captured_image is None:
                print("画像がキャプチャされませんでした。")
                continue
            captured_image.save("captured_image.png")
            print("画像を 'captured_image.png' に保存しました。")
            print("→ 画像の内容を評価中...")
            descriptions = generate_image_description(captured_image)
            print("生成された説明：")
            for i, desc in enumerate(descriptions, 1):
                print(f"[{i}] {desc}")
        except Exception as e:
            print("エラーが発生しました:", e)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()
