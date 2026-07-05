import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# 라벨: 0=정상, 1=주의, 2=위험
"""
빅데이터 수집 및 라벨링 검증 가이드

본 데이터셋은 주택임대차보호법 제10조(강행규정)에 의거하여 작성되었습니다.

정상(0) 조항은 법무부 표준 임대차 계약서 가이드라인을 준수합니다.

주의(1) 조항은 사적 자치의 원칙에 따라 합의는 가능하나 임차인에게 일방적으로 불리한 '독소 조항' 유의어 패턴을 학습시키기에 적합합니다.

위험(2) 조항은 주택임대차보호법의 강행규정을 위반하여 법적 무효에 해당하거나, 전세 사기(신탁 사기, 당일 대출 등) 및 불법 추심(단전·단수, 무단 침입)의 명백한 범죄 징후를 담고 있어, 위험도 예측 분류 모델(Classification Model)을 고도화하는 데 탁월한 학습 데이터로 기능할 것입니다.
"""
LABELS = {0: "정상", 1: "주의", 2: "위험"}
MODEL_NAME = "klue/roberta-base"
SAVE_PATH = "../model/clause_classifier"


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["clause", "label"])
    df["label"] = df["label"].astype(int)
    return df


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["clause"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, target_names=list(LABELS.values()), output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
    }


def main():
    # 1. 데이터 로드
    df = load_data("../data/clauses.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    # 2. 토크나이저 & 모델
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # MPS (Apple Silicon) 가속
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # 3. 토크나이징
    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 4. 학습 설정
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir="../model/logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 5. 학습
    trainer.train()

    # 6. 모델 저장
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"모델 저장 완료: {SAVE_PATH}")


if __name__ == "__main__":
    main()
