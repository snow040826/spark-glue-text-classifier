import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np

# 配置替换为你的任务
TASK = "glue_data/SST-2"
MODEL_DIR = "output_model"
GLUE_DIR = "glue_data/SST-2"
MAX_LENGTH = 128

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# 加载 test.tsv；GLUE 原始结构 test 集无标签
test_df = pd.read_csv(os.path.join(GLUE_DIR, "test.tsv"), sep="\t")

# 将句子列改名为统一字段 text
test_texts = test_df["sentence"].tolist()

# 转化为 HuggingFace Dataset
ds = Dataset.from_dict({"text": test_texts})

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized = ds.map(tokenize, batched=True)

# 推理
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
outputs = model(**{k: tokenized[k] for k in ("input_ids", "attention_mask")})
logits = outputs.logits.detach().cpu().numpy()
preds = np.argmax(logits, axis=1)

# 保存提交格式
submission = pd.DataFrame({
    "index": list(range(len(preds))),
    "prediction": preds
})
output_file = f"submission_{TASK}.tsv"
submission.to_csv(output_file, sep="\t", index=False)
print(f"✅ 已生成预测文件：{output_file}")
