from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch

# 加载自定义文本数据集
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', download_mode="force_redownload")

# 加载预训练的GPT-2分词器和配置
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')

# 设置填充 token 为结束 token
tokenizer.pad_token = tokenizer.eos_token

# 初始化模型并移动到MPS设备
device = torch.device("mps")
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device)

# 数据处理函数
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 使用动态掩码的数据收集器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_total_limit=2,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 评估模型
eval_results = trainer.evaluate()

print(f"Perplexity: {eval_results['eval_loss']}")

model.save_pretrained("./gpt2-finetuned-2")
tokenizer.save_pretrained("./gpt2-finetuned-2")

