import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
glue_zip_path = r"G:\软件开发\glue.zip"
glue_extract_dir = "./glue_data"

if not os.path.exists(glue_extract_dir):
    print("📦 解压 glue.zip 中...")
    with zipfile.ZipFile(glue_zip_path, 'r') as zip_ref:
        zip_ref.extractall(glue_extract_dir)
    print("✅ 解压完成！")

sst2_path = os.path.join(glue_extract_dir, "SST-2", "train.tsv")
df = pd.read_csv(sst2_path, sep="\t")
df = df.dropna()
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
)
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def preprocess(examples):
    return tokenizer(examples['text'], truncation=True)

train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
print("✅ 模型训练完成，正在保存...")
trainer.save_model("./output_model")
tokenizer.save_pretrained("./output_model")
print("✅ 模型和 tokenizer 已保存至 ./output_model")
