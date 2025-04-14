

from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
import evaluate
import torch
import os
import numpy as np

try:
    from peft import get_peft_model, LoraConfig, TaskType  
    peft_available = True
except ImportError:
    peft_available = False


def prepare_dataset(df, tokenizer, source_col="text", target_col="summary", max_input_length=512, max_target_length=128):
    """Transforme un DataFrame en dataset Hugging Face tokenisé."""
    dataset = Dataset.from_pandas(df)

    def preprocess(example):
        model_inputs = tokenizer(
            example[source_col], max_length=max_input_length, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example[target_col], max_length=max_target_length, truncation=True, padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess, batched=True)


def compute_metrics(eval_preds):
    """Évalue les prédictions avec ROUGE et BLEU."""
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, 0)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "rouge1": rouge_result["rouge1"],
        "rougeL": rouge_result["rougeL"],
        "bleu": bleu_result["bleu"]
    }


def fine_tune_model(
    df,
    model_name="facebook/bart-large-cnn",
    output_dir="./models/fine_tuned_bart",
    epochs=3,
    batch_size=4,
    eval_split_ratio=0.2,
    use_peft=False,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1
):
    """Entraîne un modèle de résumé avec support pour évaluation et LoRA (PEFT)."""
    global tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    split_index = int(len(df) * (1 - eval_split_ratio))
    train_df = df[:split_index]
    val_df = df[split_index:]

    
    tokenized_datasets = DatasetDict({
        "train": prepare_dataset(train_df, tokenizer),
        "eval": prepare_dataset(val_df, tokenizer)
    })

    if use_peft and peft_available:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        predict_with_generate=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, "logs"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer

