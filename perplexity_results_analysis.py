# %%
from tqdm import tqdm
import os
import json


data = json.load(open("results_raw/res_loss_anonym.json", "r"))


# %%
def load_doc(idx):
    d = json.load(open(f"multi_comparisons/doc{idx}.json", "r"))
    return d


# %%
# First set of questions
QUESTIONS_START_INDEX = 10_000
N_QUESTIONS = 40
questions_idxs_1 = list(range(QUESTIONS_START_INDEX, QUESTIONS_START_INDEX + N_QUESTIONS))

# Questions that weren't asked to human, to evaluate the bias of the method
more_questions_idxs_1 = list(range(QUESTIONS_START_INDEX, QUESTIONS_START_INDEX + 1000))

word_tokens_idxs = []
filter_word_tokens = True
suffix = "_fil" if filter_word_tokens else ""

docs = {}
for i in tqdm(more_questions_idxs_1):
    docs[i] = load_doc(i)
    if docs[i]["correct_token_str"][0] == " " and docs[i]["correct_token_str"][1:].isalpha():
        word_tokens_idxs.append(i)

# Second set of questions
QUESTIONS_START_INDEX_2 = 20_010
N_QUESTIONS_2 = 80
questions_idxs_2 = list(range(QUESTIONS_START_INDEX_2, QUESTIONS_START_INDEX_2 + N_QUESTIONS_2))


for i in tqdm(questions_idxs_2):
    docs[i] = load_doc(i)
    if docs[i]["correct_token_str"][0] == " " and docs[i]["correct_token_str"][1:].isalpha():
        word_tokens_idxs.append(i)

if filter_word_tokens:
    questions_idxs_1 = [x for x in questions_idxs_1 if x in word_tokens_idxs]
    questions_idxs_2 = [x for x in questions_idxs_2 if x in word_tokens_idxs]
    print("keep", len(questions_idxs_1 + questions_idxs_2), "questions out of", N_QUESTIONS + N_QUESTIONS_2)
    N_QUESTIONS = len(questions_idxs_1)
    N_QUESTIONS_2 = len(questions_idxs_2)

all_q_idxs = questions_idxs_1 + questions_idxs_2
# %%
# Load the data

import pandas as pd

data = [x for x in data if x[3] in all_q_idxs]

df = pd.DataFrame(
    data={
        "username": [x[1] for x in data],
        "guess": [x[2] for x in data],
        "comparison_id": [x[3] for x in data],
        "comparison_number": [x[4] for x in data],
        "comparison": [docs[x[3]] for x in data],
    }
)


def get_generation_low_prob(row):
    d = row["comparison"]
    generation_idx = d["generator_index"]
    return d["generated_logprobss"][generation_idx][row["comparison_number"]]


df["generation_logprob"] = df.apply(get_generation_low_prob, axis=1)

df1 = df[df["comparison_id"] < 20_000]
df2 = df[df["comparison_id"] > 20_000]

# Find users which exactly answered all questions once for each set of question
users_counts1 = df1.groupby(["username"])["username"].count()
good_users1 = list(users_counts1[users_counts1 == N_QUESTIONS].index.values)
users_counts2 = df2.groupby(["username"])["username"].count()
good_users2 = list(users_counts2[users_counts2 == N_QUESTIONS_2].index.values)
good_users = list(set(good_users1 + good_users2))


users1 = list(users_counts1.index.values)
users2 = list(users_counts2.index.values)
assert len([u for u in good_users1 if u in users2 and u not in good_users2]) == 0
assert len([u for u in good_users2 if u in users1 and u not in good_users1]) == 0

# And only keep those
df = df[df["username"].isin(good_users)]
# %%
# Different rounding functions
import numpy as np


def round_to_neirest(r, l):
    distances = [abs(r - x) for x in l]
    return l[np.argmin(distances)]


def round_basic(r):
    r = round(r * 10) / 10
    if r < 0.01:
        return 0.01
    if r > 0.99:
        return 0.99
    return r


percentages_1 = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
percentages_2 = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
percentage_quite_extreme = [0.1, 0.2, 0.5, 0.8, 0.9]
percentages_extreme = [0.2, 0.5, 0.8]
percentage_very_extreme = [0.25, 0.75]


def round_to_dec_or_cent_p1(r):
    return round_to_neirest(r, percentages_1)


def round_to_dec_or_cent_p2(r):
    return round_to_neirest(r, percentages_2)


def round_to_dec_or_cent_p_quite_extreme(r):
    return round_to_neirest(r, percentage_quite_extreme)


def round_to_dec_or_cent_p_extreme(r):
    return round_to_neirest(r, percentages_extreme)


def round_to_dec_or_cent_p_very_extreme(r):
    return round_to_neirest(r, percentage_very_extreme)


def no_round(r):
    return r


def to_log_space(x):
    return np.log(1 / (1 / x - 1))


def from_log_space(x):
    return 1 / (1 + np.exp(-x))


def round_to_dec_or_cent_log_space1(r):
    return from_log_space(round_to_neirest(to_log_space(r), [to_log_space(x) for x in percentages_1]))


def round_to_dec_or_cent_log_space2(r):
    return from_log_space(round_to_neirest(to_log_space(r), [to_log_space(x) for x in percentages_2]))


def fake_row():
    return {"comparison_number": np.random.randint(400)}


def fake_answers(d, size):
    return [(i, fake_row()) for i in range(size)]


# %%
def get_approx(player_model_idx=None, round_function=no_round, fake_rows=None, q_idx=questions_idxs_1, title=""):
    """Get an approximation of the log loss

    player_model_idx means human
    otherwise it's the id of the model (see below for which is which)"""
    model_ref_idx = 1

    all_sum_bits = []
    loss_bits = []
    for comparison_id in q_idx:
        d = docs[comparison_id]
        answers = (
            df[df["comparison_id"] == comparison_id].iterrows() if fake_rows is None else fake_answers(d, fake_rows)
        )

        sum_bits = []
        for i, row in answers:
            if player_model_idx is not None:
                pgood_player_model = np.exp(d["correct_logprobs"][player_model_idx])
                pbad_player_model = np.exp(d["generated_logprobss"][player_model_idx][row["comparison_number"]])

            ratio = (
                row["guess"] / 100
                if player_model_idx is None
                else (pgood_player_model / (pgood_player_model + pbad_player_model))
            )
            ratio = round_function(ratio)

            pbad_over_pgood = 1 / ratio - 1

            pgood_ref = np.exp(d["correct_logprobs"][model_ref_idx])

            pbad_generator = np.exp(d["generated_logprobss"][model_ref_idx][row["comparison_number"]])

            sum_bits.append(pgood_ref * pbad_over_pgood / pbad_generator)

        loss_bit = np.log(sum(sum_bits) / len(sum_bits))
        loss_bits.append(loss_bit)
        all_sum_bits.append(sum_bits)

    a = np.array(loss_bits)
    return loss_bits, all_sum_bits


def print_approx(title="", **approx_args):
    """Get the approximation and print the loss"""
    loss_bits, all_sum_bits = get_approx(**approx_args)
    a = np.array(loss_bits)
    n = len(loss_bits)

    p_stds = [np.array(sum_bit).std() / np.sqrt(len(sum_bit)) for sum_bit in all_sum_bits]
    p_avgs = [np.array(sum_bit).mean() for sum_bit in all_sum_bits]
    p_std_contributions = np.array([p_std / (p * n) for (p, p_std) in zip(p_avgs, p_stds)])
    total_uncertainty = np.sqrt((p_std_contributions**2).sum())

    std = a.std() / np.sqrt(len(approx_args["q_idx"]))
    print(f"{title}{a.mean():.2f} L_std {std:.2f} P_std {total_uncertainty:.2f}")


# Print results for the first set
print_approx(title="human: ", q_idx=questions_idxs_1)
print_approx(title="2l: ", player_model_idx=0, q_idx=questions_idxs_1)
print_approx(title="12l: ", player_model_idx=1, q_idx=questions_idxs_1)
print_approx(title="24l: ", player_model_idx=2, q_idx=questions_idxs_1)
print_approx(title="2l rounded: ", round_function=round_basic, player_model_idx=0, q_idx=questions_idxs_1)
print_approx(title="12l rounded:: ", round_function=round_basic, player_model_idx=1, q_idx=questions_idxs_1)
print_approx(title="24l rounded:: ", round_function=round_basic, player_model_idx=2, q_idx=questions_idxs_1)
# %% check true loss
q_idx = questions_idxs_1
for i, model_name in enumerate(["2layers", "12layers", "24layers"]):
    a = np.array([-docs[j]["correct_logprobs"][i] for j in q_idx])
    print(model_name, f"{a.mean():.2f} +- {a.std() / np.sqrt(len(q_idx)):.2f}")
# %% check true loss for all questions
q_idx = all_q_idxs
for i, model_name in enumerate(["2layers", "12layers", "24layers"]):
    a = np.array([-docs[j]["correct_logprobs"][i] for j in q_idx])
    # weights = [len(good_users1)] * len(questions_idxs_1) + [len(good_users2)] * len(questions_idxs_2)=
    print(model_name, f"{a.mean():.2f} +- {a.std() / np.sqrt(len(q_idx)):.2f}")
# %%
# check true loss for more questions (including questions outside the sample)
q_idx = more_questions_idxs_1
true_losses = []
for i, model_name in enumerate(["2layers", "12layers", "24layers"]):
    a = np.array([-docs[j]["correct_logprobs"][i] for j in q_idx])
    # weights = [len(good_users1)] * len(questions_idxs_1) + [len(good_users2)] * len(questions_idxs_2)
    true_losses.append(a.mean())
    print(model_name, f"{a.mean():.2f} +- {a.std() / np.sqrt(len(q_idx)):.2f}")
# %%
# Print results for the second set
print_approx(title="human: ", q_idx=all_q_idxs)
print_approx(title="2l: ", player_model_idx=0, q_idx=all_q_idxs)
print_approx(title="12l: ", player_model_idx=1, q_idx=all_q_idxs)
print_approx(title="24l: ", player_model_idx=2, q_idx=all_q_idxs)
print_approx(title="2l rounded: ", round_function=round_basic, player_model_idx=0, q_idx=all_q_idxs)
print_approx(title="12l rounded:: ", round_function=round_basic, player_model_idx=1, q_idx=all_q_idxs)
print_approx(title="24l rounded:: ", round_function=round_basic, player_model_idx=2, q_idx=all_q_idxs)
# %%
# Plot results
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

use_stat_uncertainty = True  # CHANGE THIS to False to obtain the graph with the other uncertainty estimator


# sum_bits, all_sum_bits = get_approx(q_idx=all_q_idxs)
def plot_error_bar(height, value, two_sigma_err, title, color, true_value):
    plt.scatter(value, [title], marker=".", c=color)
    plt.errorbar(
        [value],
        [title],
        xerr=two_sigma_err,
        fmt="none",  # don't connect data points
        ecolor=color,  # color of error lines
        elinewidth=1,  # width of error lines
        capsize=4,  # length of error caps
        zorder=-1,  # put error bars behind scatter points
    )
    if true_value is not None:
        plt.scatter(true_value, [title], marker="*", c=color)


def plot_approx(height=0, title="", color="red", true_value=None, **approx_args):
    q_idx = approx_args["q_idx"]
    loss_bits, all_sum_bits = get_approx(**approx_args)
    a = np.array(loss_bits)
    n = len(loss_bits)

    p_stds = [np.array(sum_bit).std() / np.sqrt(len(sum_bit)) for sum_bit in all_sum_bits]
    p_avgs = [np.array(sum_bit).mean() for sum_bit in all_sum_bits]
    p_std_contributions = np.array([p_std / (p * n) for (p, p_std) in zip(p_avgs, p_stds)])
    total_uncertainty = np.sqrt((p_std_contributions**2).sum())

    std = a.std() / np.sqrt(len(q_idx))
    print(f"{title}{a.mean():.2f} L_std {std:.2f} P_std {total_uncertainty:.2f}")

    error_used = std if use_stat_uncertainty else total_uncertainty
    perplexity_estimate = np.exp(a.mean() + true_losses[1])
    perplexity_lower_estimate = np.exp(a.mean() + true_losses[1] - 2 * error_used)
    perplexity_higher_estimate = np.exp(a.mean() + true_losses[1] + 2 * error_used)
    x_err_low = perplexity_estimate - perplexity_lower_estimate
    x_err_high = perplexity_higher_estimate - perplexity_estimate
    errs = np.array([x_err_low, x_err_high])[:, None]
    print(errs.shape)

    plot_error_bar(height, perplexity_estimate, errs, title, color, true_value)


import matplotlib

fig1, ax1 = plt.subplots(figsize=(10, 3))
ax1.set_xscale("log")
ax1.set_xticks([10, 20, 40, 80, 100, 160])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


plt.grid(axis="y", zorder=-2)

plot_approx(title="human", q_idx=all_q_idxs, color="blue")
plot_approx(title="a 14M-parameter model rounded", round_function=round_basic, player_model_idx=0, q_idx=all_q_idxs)
plot_approx(title="GPT-2 small rounded:", round_function=round_basic, player_model_idx=1, q_idx=all_q_idxs)
plot_approx(title="GPT-2 medium rounded:", round_function=round_basic, player_model_idx=2, q_idx=all_q_idxs)
plot_approx(title="a 14M-parameter model", player_model_idx=0, q_idx=all_q_idxs, true_value=np.exp(true_losses[0]))
plot_approx(title="GPT-2 small", player_model_idx=1, q_idx=all_q_idxs, true_value=np.exp(true_losses[1]))
plot_approx(title="GPT-2 medium", player_model_idx=2, q_idx=all_q_idxs, true_value=np.exp(true_losses[2]))

star_legend = mlines.Line2D([], [], linestyle="", marker="*", color="k", label="true perplexity on the samples")
dot_legend = mlines.Line2D(
    [], [], linestyle="", marker=".", color="k", label="measured perplexity on the samples\nusing the estimator"
)
plt.legend(handles=[star_legend, dot_legend], loc="upper right")
plt.xlabel("per token perplexity")
plt.tight_layout()
plt.savefig(f"figures/perplexity{suffix}.pdf")
plt.show()
# %%
# Plot results with boostrapped uncertainty
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def plot_approx_boostrap(height=0, title="", color="red", true_value=None, **approx_args):
    global df

    orig_df = df

    perplexity_estimates = []
    for i in range(100):
        np.random.seed(i)
        # reject half of users
        good_users_ = np.random.choice(good_users, len(good_users) // 2, replace=False)
        df = df[df["username"].isin(good_users_)]

        loss_bits, _ = get_approx(**approx_args)
        a = np.array(loss_bits)
        perplexity_estimate = np.exp(a.mean() + true_losses[1])
        perplexity_estimates.append(perplexity_estimate)

        df = orig_df

    perplexity_estimate = np.mean(perplexity_estimates)
    perplexity_lower_estimate = np.quantile(perplexity_estimates, 0.05)
    perplexity_higher_estimate = np.quantile(perplexity_estimates, 0.95)
    x_err_low = perplexity_estimate - perplexity_lower_estimate
    x_err_high = perplexity_higher_estimate - perplexity_estimate
    errs = np.array([x_err_low, x_err_high])[:, None]

    plot_error_bar(height, perplexity_estimate, errs, title, color, true_value)


import matplotlib

fig1, ax1 = plt.subplots(figsize=(10, 3))
ax1.set_xscale("log")
ax1.set_xticks([10, 20, 40, 80, 100, 160])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


plt.grid(axis="y", zorder=-2)

plot_approx_boostrap(title="human", q_idx=all_q_idxs, color="blue")
plot_approx_boostrap(
    title="a 14M-parameter model rounded", round_function=round_basic, player_model_idx=0, q_idx=all_q_idxs
)
plot_approx_boostrap(title="GPT-2 small rounded:", round_function=round_basic, player_model_idx=1, q_idx=all_q_idxs)
plot_approx_boostrap(title="GPT-2 medium rounded:", round_function=round_basic, player_model_idx=2, q_idx=all_q_idxs)
plot_approx_boostrap(
    title="a 14M-parameter model", player_model_idx=0, q_idx=all_q_idxs, true_value=np.exp(true_losses[0])
)
plot_approx_boostrap(title="GPT-2 small", player_model_idx=1, q_idx=all_q_idxs, true_value=np.exp(true_losses[1]))
plot_approx_boostrap(title="GPT-2 medium", player_model_idx=2, q_idx=all_q_idxs, true_value=np.exp(true_losses[2]))

star_legend = mlines.Line2D([], [], linestyle="", marker="*", color="k", label="true perplexity on the samples")
dot_legend = mlines.Line2D(
    [], [], linestyle="", marker=".", color="k", label="measured perplexity on the samples\nusing the estimator"
)
plt.legend(handles=[star_legend, dot_legend], loc="upper right")
plt.xlabel("per token perplexity")
plt.tight_layout()
plt.savefig(f"figures/perplexity_estimation_bootstrapped{suffix}.pdf")
plt.show()
# %%
import matplotlib

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xscale("log")
ax1.set_xticks([10, 20, 40, 80, 100, 160])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


plt.grid(axis="y", zorder=-2)

for name, round_fn in [
    ("even more round", round_to_dec_or_cent_p_extreme),
    ("more round", round_to_dec_or_cent_p_quite_extreme),
    # ("round to [0.25, 0.75]", round_to_dec_or_cent_p_very_extreme),
]:
    plot_approx(title=f"human {name}", q_idx=all_q_idxs, color="blue", round_function=round_fn)
    plot_approx(
        title=f"a 14M-parameter model rounded {name}", round_function=round_fn, player_model_idx=0, q_idx=all_q_idxs
    )
    plot_approx(title=f"GPT-2 small rounded {name}", round_function=round_fn, player_model_idx=1, q_idx=all_q_idxs)
    plot_approx(title=f"GPT-2 medium rounded {name}", round_function=round_fn, player_model_idx=2, q_idx=all_q_idxs)

plot_approx(title="human", q_idx=all_q_idxs, color="blue")
plot_approx(title="a 14M-parameter model rounded", round_function=round_basic, player_model_idx=0, q_idx=all_q_idxs)
plot_approx(title="GPT-2 small rounded", round_function=round_basic, player_model_idx=1, q_idx=all_q_idxs)
plot_approx(title="GPT-2 medium rounded", round_function=round_basic, player_model_idx=2, q_idx=all_q_idxs)
plot_approx(title="a 14M-parameter model", player_model_idx=0, q_idx=all_q_idxs, true_value=np.exp(true_losses[0]))
plot_approx(title="GPT-2 small", player_model_idx=1, q_idx=all_q_idxs, true_value=np.exp(true_losses[1]))
plot_approx(title="GPT-2 medium", player_model_idx=2, q_idx=all_q_idxs, true_value=np.exp(true_losses[2]))

star_legend = mlines.Line2D([], [], linestyle="", marker="*", color="k", label="true perplexity on the samples")
dot_legend = mlines.Line2D(
    [], [], linestyle="", marker=".", color="k", label="measured perplexity on the samples\nusing the estimator"
)
plt.legend(handles=[star_legend, dot_legend], loc="upper right")
plt.xlabel("per token perplexity")

plt.tight_layout()
plt.savefig(f"figures/perplexity_estimation{suffix}.pdf")
# %%

# %%
# compare roundings
for round_func in [
    no_round,
    round_basic,
    round_to_dec_or_cent_p1,
    round_to_dec_or_cent_log_space1,
    round_to_dec_or_cent_p2,
    round_to_dec_or_cent_log_space2,
]:
    print(round_func.__name__)
    print_approx(title="2l rounded: ", round_function=round_func, player_model_idx=0, q_idx=questions_idxs_1)
    print_approx(title="12l rounded:: ", round_function=round_func, player_model_idx=1, q_idx=questions_idxs_1)
    print_approx(title="24l rounded:: ", round_function=round_func, player_model_idx=2, q_idx=questions_idxs_1)

# %%
# check bias in the limit of many data points per question
print_approx(title="2l: ", player_model_idx=0, q_idx=all_q_idxs, fake_rows=10000)
print_approx(title="12l: ", player_model_idx=1, q_idx=all_q_idxs, fake_rows=1000)
print_approx(title="24l: ", player_model_idx=2, q_idx=all_q_idxs, fake_rows=1000)
print_approx(title="2l rounded: ", round_function=round_basic, player_model_idx=0, q_idx=all_q_idxs, fake_rows=1000)
print_approx(title="12l rounded:: ", round_function=round_basic, player_model_idx=1, q_idx=all_q_idxs, fake_rows=1000)
print_approx(title="24l rounded:: ", round_function=round_basic, player_model_idx=2, q_idx=all_q_idxs, fake_rows=1000)
# %%
print_approx(title="2l: ", player_model_idx=0, q_idx=all_q_idxs, fake_rows=400)


# %%
# Bias on the questions asked to the participants
def plot_theoritical(q_idx, attempts=1, n_inners=[1, 4, 10, 40, 100, 400]):
    def get_loss_mean(model_idx):
        a = np.array([-docs[i]["correct_logprobs"][model_idx] for i in q_idx])
        return a.mean()

    results = []
    true_delta_2 = get_loss_mean(2) - get_loss_mean(1)
    true_delta_0 = get_loss_mean(0) - get_loss_mean(1)

    for n_inner in n_inners:
        print(n_inner)
        for _ in range(attempts):
            effective_delta_2 = np.array(get_approx(player_model_idx=2, q_idx=q_idx, fake_rows=n_inner)[0]).mean()
            effective_delta_0 = np.array(get_approx(player_model_idx=0, q_idx=q_idx, fake_rows=n_inner)[0]).mean()
            results.append((n_inner, effective_delta_2, effective_delta_0))

    delta_0 = [(n, d0) for n, d2, d0 in results]
    delta_2 = [(n, d2) for n, d2, d0 in results]

    plt.scatter(
        [x for (x, y) in delta_0],
        [y - true_delta_0 for (x, y) in delta_0],
        label="error in 14M-parameter loss estimation",
    )
    plt.scatter(
        [x for (x, y) in delta_2],
        [y - true_delta_2 for (x, y) in delta_2],
        label="error in 345M-parameter loss estimation",
    )

    # plt.title(
    #     f"""Error in the loss estimation for different language model
    # using the 117M-parameter model as generator
    # N={len(q_idx)}"""
    # )
    plt.axhline(0, color="black", label="no error line")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Samples for a given prompt (n)")
    plt.ylabel(f"Estimated loss - true loss (in bits)")
    plt.savefig(f"figures/loss_underestimation{suffix}.pdf")
    # plt.show()


plot_theoritical(all_q_idxs)
# %%
# On 1000 questions
plot_theoritical(more_questions_idxs_1)
# %%
