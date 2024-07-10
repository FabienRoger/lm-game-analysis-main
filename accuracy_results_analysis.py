# %%

from math import e
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
# %%

import pandas as pd

data_ = [x for x in data if x[5] < N]
df = pd.DataFrame(
    data={
        "id": [x[0] for x in data_],
        "username": [x[1] for x in data_],
        "guess": [x[2] for x in data_],
        "guessed_token_idx": [x[4] for x in data_],
        "doc_id": [x[5] for x in data_],
        "doc_tokens": [docs[x[5]] for x in data_],
    }
)
df["doc_str"] = df["doc_tokens"].apply(lambda l: "".join(l))
# %%

# Remove users that tested the app and the "anon" username
cleaned_df = df[
    (df["username"] != hash("anon"))
    & (df["username"] != -6247052403041399844)
    & (df["username"] != 7513247084907933005)
    & (df["username"] != 3056584949900737159)
]


def get_correct(row):
    if row["guessed_token_idx"] > len(row["doc_tokens"]):
        print("invalid doc")
        return ""
    else:
        return row["doc_tokens"][row["guessed_token_idx"] - 1]


def get_next_correct(row):
    if row["guessed_token_idx"] >= len(row["doc_tokens"]):
        return ""
    else:
        return row["doc_tokens"][row["guessed_token_idx"]]


cleaned_df["correct_guess"] = cleaned_df.apply(func=get_correct, axis=1)
cleaned_df["next_correct_guess"] = cleaned_df.apply(func=get_next_correct, axis=1)

print(len(cleaned_df))  # 24175

initial_filter = True
if initial_filter:
    def filter_guess(row):
        return "\n" not in row["correct_guess"] and "�" not in row["correct_guess"] and row["correct_guess"] != ""
    
    cleaned_df = cleaned_df[cleaned_df.apply(func=filter_guess, axis=1)]
else:
    # new filter is only keeps full words: space then letters (no digit), and next tok start with space
    def filter_guess(row):
        this_correct = row["correct_guess"]
        next_correct = row["next_correct_guess"]

        if len(this_correct) < 2 or len(next_correct) < 1:
            return False

        return this_correct[0] == " " and this_correct[1:].isalpha() and next_correct[0] == " "

    cleaned_df = cleaned_df[cleaned_df.apply(func=filter_guess, axis=1)]


def f(row):
    return "".join(row["doc_tokens"][: row["guessed_token_idx"] - 1])


cleaned_df["prompt"] = cleaned_df.apply(func=f, axis=1)

print(len(cleaned_df))  # 22642 with initial filter, 14442 with new filter
# %%
good_answers = cleaned_df[cleaned_df["correct_guess"] == cleaned_df["guess"]]
bad_answers = cleaned_df[cleaned_df["correct_guess"] == cleaned_df["guess"]]
accuracy = good_answers.shape[0] / cleaned_df.shape[0]
print(accuracy)
# %%
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
# %%
import asyncio
import os
import openai
import pandas as pd
import json
from functools import lru_cache
import nest_asyncio

nest_asyncio.apply()

suffix = "_orig" if initial_filter else ""

run_accuracy = False
if run_accuracy:
    # OAI = (os.getenv("OPENAI_API_KEY"), "https://api.openai.com/v1")
    GAI = (os.getenv("GOOSE_API_KEY"), "https://api.goose.ai/v1")
    # engines = {"fairseq-13b": GAI, "fairseq-1-3b": GAI, "fairseq-125m": GAI}
    # use gpt-neo models
    engines = {
        "gpt-neo-125m": GAI,
        "gpt-neo-1-3b": GAI,
        "gpt-neo-2-7b": GAI,
        "gpt-j-6b": GAI,
    }

    async def get_prediction_on(prompt, engine):
        openai.api_key, openai.api_base = engines[engine]
        try:
            completion = await openai.Completion.acreate(
                engine=engine,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                stream=False,
            )
        except Exception as e:
            await asyncio.sleep(5)
            print(e)
            return get_prediction_on(prompt, engine)
        return completion.choices[0].text

    async def get_prediction(prompt):
        return {engine: await get_prediction_on(prompt, engine) for engine in engines.keys()}

    c = 0

    async def get_model_preds(row):
        global c
        c += 1
        if c % 10 == 0:
            print(c, end=" ")
        r = await get_prediction(row["prompt"])
        return pd.Series(list(r.values()))

    async def main():
        small_cleaned_df = cleaned_df.sample(1000, random_state=0)
        tasks = [get_model_preds(row) for _, row in small_cleaned_df.iterrows()]
        results = await asyncio.gather(*tasks)
        small_cleaned_df[list(engines.keys())] = results

        models_perf = {}
        for engine in engines.keys():
            good_answers = small_cleaned_df[small_cleaned_df["correct_guess"] == small_cleaned_df[engine]]
            bad_answers = small_cleaned_df[small_cleaned_df["correct_guess"] != small_cleaned_df[engine]]
            accuracy = good_answers.shape[0] / small_cleaned_df.shape[0]
            print(engine, accuracy)
            models_perf[engine] = accuracy

        json.dump(models_perf, open(f"results_raw/model_perfs_new_new_new{suffix}.json", "w"))

    asyncio.run(main())

models_perf = json.load(open(f"results_raw/model_perfs_new_new_new{suffix}.json", "r"))
del models_perf["gpt-neo-2-7b"]
# %%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 4))

plt.xlabel("Top-1 accuracy (%)")
plt.ylabel("Number of human players")
plt.hist(np.array(accuracies) * 100, bins=np.arange(0, 60 + 1e-10, 2), label="human performance")

displayed_names = {
    # "davinci": "GPT-3",
    # "curie": "curie",
    # "babbage": "babbage",
    # "ada": "ada",
    # "davinci-002": "Davinci v2",
    # "babbage-002": "Babbage v2",
    # "fairseq-13b": "fairseq-13b",
    # "fairseq-1-3b": "fairseq-1.3b",
    # "fairseq-125m": "fairseq-125m",
    "gpt-neo-125m": "GPT-Neo 125M",
    "gpt-neo-1-3b": "GPT-Neo 1.3B",
    "gpt-neo-2-7b": "GPT-Neo 2.7B",
    "gpt-j-6b": "GPT-J (6B)",
    "davinci": "GPT-3 (175B)",
}
models_perf_d = {displayed_names[name]: perf for name, perf in models_perf.items() if name in displayed_names}

for name, acc in models_perf_d.items():
    plt.axvline(acc * 100, c="red", alpha=0.9)
    plt.text(acc * 100 + 0.5, 4 if initial_filter else 4.5, name, rotation=90, va="bottom")

# minor ticks
ax.set_xticks(np.arange(0, 60 + 1e-10, 2), minor=True)

plt.legend()
plt.tight_layout()
plt.savefig(f"figures/accuracy_vs_model_perfs{suffix}.pdf")
plt.show()
# %%
