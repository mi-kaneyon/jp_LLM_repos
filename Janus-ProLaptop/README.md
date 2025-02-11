
# Janus-ProLaptop
- It is used for low VRAM PC or laptop PC to generate instruction
- If your PC is insufficient power to generate sentences, you can use cpu generating version

# Base model

https://huggingface.co/deepseek-ai/Janus-Pro-7B

@article{chen2025janus,
  title={Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling},
  author={Chen, Xiaokang and Wu, Zhiyu and Liu, Xingchao and Pan, Zizheng and Liu, Wen and Xie, Zhenda and Yu, Xingkai and Ruan, Chong},
  journal={arXiv preprint arXiv:2501.17811},
  year={2025}
}


# Requirement 

Transformers
pytorch

# Usage
1. GPU version
 
```
python picchk.py

```
2. CPU version 

```
python cpujanus_inst.py

```

# Example

```
[INFO] Model loaded successfully (CPU mode).
User prompt (type 'exit' to quit): 芋を食べる

---- Janus-Pro Assistant (CPU) ----
: 芋を食べる

:芋を食べることは、一般的に芋の皮を削除して、芋の本体を食べることです。芋の本体は、芋のように甘いものです。

芋の食べ方


```






