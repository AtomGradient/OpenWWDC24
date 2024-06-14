from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 加载保存的模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned/checkpoint-55000")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned/checkpoint-55000")

# 移动模型到合适的设备（如CPU或MPS）
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
model.to(device)

# 推理函数
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = model.generate(
        inputs.input_ids, 
        attention_mask=attention_mask, 
        max_length=max_length, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试推理
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)

