# %%
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

neo_train_set = "NeelNanda/pile-10k", "train"
neo_val_set = "mit-han-lab/pile-val-backup", "validation"

neo_models = ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6B"]


@torch.no_grad()
def preprocess_and_compute_metrics(model_name, ds, split):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    dataset = load_dataset(ds, split=split)

    texts = [dataset[i]["text"] for i in range(1024)]

    perplexities = []
    accuracies = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            # Calculate perplexity
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
            # Calculate top-1 accuracy
            predictions = torch.argmax(shift_logits, dim=-1)
            accuracy = (predictions == shift_labels).float().mean().item()
            accuracies.append(accuracy)
    print(f"Model: {model_name} on {ds} {split}")
    print(f"Average Perplexity: {np.mean(perplexities)} +- {np.std(perplexities, ddof=1) / np.sqrt(len(perplexities))}")
    print(f"Top-1 Accuracy by token: {np.mean(accuracies)} +- {np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))}")

    return {
        "model": model_name,
        "dataset": ds,
        "split": split,
        "perplexity": np.mean(perplexities),
        "accuracy": np.mean(accuracies),
    }


res = []

for model in neo_models:
    res.append(preprocess_and_compute_metrics(model, *neo_val_set))
    res.append(preprocess_and_compute_metrics(model, *neo_train_set))
