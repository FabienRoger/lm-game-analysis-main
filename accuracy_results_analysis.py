#%%

from tqdm import tqdm
import os
import json

make_anonym = False
if make_anonym:
    data_ = json.load(open("results_raw/res_accuracy.json", "r"))
    anonym_data = []
    for d in data_:
        d[1] = hash(d[1])
        anonym_data.append(d)
    json.dump(anonym_data, open("results_raw/res_accuracy_anonym.json", "w"))

data = json.load(open("results_raw/res_accuracy_anonym.json", "r"))


# %%


def load_doc(idx):
    d = json.load(open(f"docs/doc{idx}.json"))
    return d


docs = []
N = 10_000
for i in tqdm(range(N)):
    docs.append(load_doc(i))
#%%

import pandas as pd

data_ = [x for x in data if x[5] < N]
df = pd.DataFrame(
    data={
        "id": [x[0] for x in data_],
        "username": [x[1] for x in data_],
        "guess": [x[2] for x in data_],
        "correct_answer": [x[3] for x in data_],
        "guessed_token_idx": [x[4] for x in data_],
        "doc_id": [x[5] for x in data_],
        "doc_tokens": [docs[x[5]] for x in data_],
    }
)
df["doc_str"] = df["doc_tokens"].apply(lambda l: "".join(l))
#%%

# Remove users that tested the app and the "anon" username
cleaned_df = df[
    (df["username"] != hash("anon"))
    & (df["username"] != -6247052403041399844)
    & (df["username"] != 7513247084907933005)
    & (df["username"] != 3056584949900737159)
]


def f(row):
    if row["guessed_token_idx"] > len(row["doc_tokens"]):
        print("invalid doc")
        return ""
    else:
        return row["doc_tokens"][row["guessed_token_idx"] - 1]


cleaned_df["correct_guess"] = cleaned_df.apply(func=f, axis=1)
cleaned_df = cleaned_df[cleaned_df["correct_guess"].str.contains("\n") == False]


def f(row):
    return "".join(row["doc_tokens"][: row["guessed_token_idx"] - 1])


cleaned_df["prompt"] = cleaned_df.apply(func=f, axis=1)
# %%
good_answers = cleaned_df[cleaned_df["correct_guess"] == cleaned_df["guess"]]
bad_answers = cleaned_df[cleaned_df["correct_guess"] == cleaned_df["guess"]]
accuracy = good_answers.shape[0] / cleaned_df.shape[0]
print(accuracy)
#%%
users_counts = cleaned_df.groupby(["username"])["username"].count()
good_users = list(users_counts[users_counts > 50].index.values)
large_users_df = cleaned_df[cleaned_df["username"].isin(good_users)]
large_users_good_answers = cleaned_df[cleaned_df["correct_guess"] == cleaned_df["guess"]]

print("global acc = ", large_users_good_answers.shape[0] / large_users_df.shape[0])

accuracies = []
for user in large_users_df["username"].unique():
    good = large_users_good_answers[large_users_good_answers["username"] == user].shape[0]
    total = large_users_df[large_users_df["username"] == user].shape[0]
    print(user, good, total, f"{good/total:.2f}")
    accuracies.append(good / total)
#%%
run_accuracy = False
if run_accuracy:
    import openai
    from functools import lru_cache

    OAI = (os.getenv("OPENAI_API_KEY"), "https://api.openai.com/v1")
    # GAI = (os.getenv("GOOSE_API_KEY"), "https://api.goose.ai/v1")
    # engines = {"davinci": OAI, "fairseq-13b": GAI, "fairseq-1-3b": GAI, "fairseq-125m": GAI}
    engines = {"davinci": OAI, "babbage": OAI, "curie": OAI, "ada": OAI}

    def get_prediction_on(prompt, engine):
        openai.api_key, openai.api_base = engines[engine]
        completion = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            stream=False,
        )
        return completion.choices[0].text

    @lru_cache(maxsize=None)
    def get_prediction(prompt):
        return {engine: get_prediction_on(prompt, engine) for engine in engines.keys()}

    from random import randint

    c = 0

    def get_model_preds(row):
        global c
        c += 1
        if c % 10 == 0:
            print(c, end=" ")
        r = get_prediction(row["prompt"])
        return pd.Series(list(r.values()))

    small_cleaned_df = cleaned_df.sample(1000, random_state=0)
    small_cleaned_df[list(engines.keys())] = small_cleaned_df.apply(func=get_model_preds, axis=1)

    models_perf = {}
    for engine in engines.keys():
        good_answers = small_cleaned_df[small_cleaned_df["correct_guess"] == small_cleaned_df[engine]]
        bad_answers = small_cleaned_df[small_cleaned_df["correct_guess"] == small_cleaned_df[engine]]
        accuracy = good_answers.shape[0] / small_cleaned_df.shape[0]
        print(engine, accuracy)
        models_perf[engine] = accuracy

    json.dump(models_perf, open("results_raw/model_perfs.json", "w"))
else:
    models_perf = json.load(open("results_raw/model_perfs.json", "r"))
#%%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5))

plt.xlabel("top-1 accuracy")
plt.ylabel("number of players")
plt.title("players vs LM accuracy")
plt.hist(accuracies, bins=np.arange(0, 0.60 + 1e-10, 0.02), label="human performance distribution")

displayed_names = {
    "davinci": "GPT-3",
    "curie": "curie",
    "babbage": "babbage",
    "ada": "ada",
}
models_perf = {displayed_names[name]: perf for name, perf in models_perf.items() if name in displayed_names}

for name, acc in models_perf.items():
    plt.axvline(acc, c="red", alpha=0.5)
    plt.text(acc - 0.012, 5, name, rotation=90, va="top")

plt.xticks(list(np.arange(0, 0.60 + 1e-10, 0.1)))
plt.minorticks_on()

plt.legend()
#%%
# Accuracy per text, ordered by difficulty
results = []
for doc_id in cleaned_df["doc_id"].unique():
    if cleaned_df[cleaned_df["doc_id"] == doc_id].shape[0] > 10:
        good = good_answers[good_answers["doc_id"] == doc_id].shape[0]
        total = cleaned_df[cleaned_df["doc_id"] == doc_id].shape[0]
        r = (doc_id, f"{good / total:.2f}", good, total, repr("".join(docs[doc_id]))[:100], good / total)
        results.append(r)
results = sorted(results, key=lambda t: (t[-1]))
for r in results:
    print(*r)
# %%
