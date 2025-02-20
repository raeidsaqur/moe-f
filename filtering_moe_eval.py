import os
from pprint import pprint
from typing import List

import pandas as pd
pd.set_option("display.max_colwidth", None)

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import fire
import prettytable

PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
EXPERT_FILTER = "moe-f"  # the outout filename is f"df_{EXPERT_FILTER}"
EXPERT_FILTER_LABEL = "MoE-F"  # display ina
EXPERIMENTS_PATH = "MoE-F_supplementary.materials/experiments"
LABELS = ["Fall", "Neutral", "Rise"]
LABELS_SUPPORT = [73, 143, 101]  # Fall -> 73, Neutral -> 143, Rise -> 101
NUMERIC_LABELS = [0, 1, 2]
LABEL_MAP = {"Fall": 0, "Neutral": 1, "Rise": 2}
ADAPTERS = ["nifty", "acl18", "cikm18", "bigdata22"]
SEEDS = [0, 13, 42]
ALL_LLMS = [
    "Llama-2-7b-chat-hf",
    "Llama-2-70b-chat-hf",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-70B-Instruct",
    "Mixtral-8x7B-Instruct-v0.1",
    "dbrx-instruct",
    "OpenAI-gpt-4o"
]

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_experts(
    model_name: str, model_variant: str, with_results: bool = False, with_adapters=False
) -> List[str]:
    """Llama-2-7b-chat-hf experts -> ['Llama-2-7b-chat-hf', 'nifty', 'acl18', 'cikm18', 'bigdata22']
    Meta-Llama-3-8B-Instruct experts -> ['Meta-Llama-3-8B-Instruct', 'nifty', 'acl18', 'cikm18', 'bigdata22']
     :param with_adapters:
    """
    chatlm_id = f"{model_name}-{model_variant}"
    experts = [chatlm_id]
    _with_adapters = chatlm_id in ["Llama-2-7b-chat-hf", "Meta-Llama-3-8B-Instruct"]
    if with_adapters:
        experts += ADAPTERS  # ["nifty", "acl18", "cikm18", "bigdata22"]
    if with_results:
        experts += [EXPERT_FILTER]  # ["moe-f"]
    return experts


def get_experts_results_fn(
    model_name: str, model_variant: str, with_results: bool = False, with_adapters=False
) -> List[str]:
    """Returns a list of only the results file name of an expert: f"df_{expert}", nothing else
    Llama-2-7b-chat experts -> ['Llama-2-7b-chat', 'nifty', 'acl18', 'cikm18', 'bigdata22']
    Meta-Llama-3-8B-Instruct experts -> ['Meta-Llama-3-8B-Instruct', 'nifty', 'acl18', 'cikm18', 'bigdata22']
    N.B. doesn't return the expert "moe-f" -> "df_moe-f.csv"
    :param with_adapters:
    """
    chatlm_id = f"{model_name}-{model_variant}"
    _with_adapters = f"{model_name}-{model_variant}" in [
        "Meta-Llama-3-8B-Instruct",
        "Llama-2-7b-chat",
    ]
    experts = get_experts(
        model_name,
        model_variant,
        with_results=with_results,
        with_adapters=(_with_adapters or with_adapters),
    )
    expert_filenames = [f"df_{expert}.csv" for expert in experts]
    return expert_filenames


def get_performance_from(
    csv_file_path: str = None,
    df: pd.DataFrame = None,
    is_return_report=True,
    average="weighted",
):
    """
    Original method used to calculate and report Table 2 performance metrics in paper.

    :param csv_file_path:
    :param df:
    :param is_return_report:
    :param average: 'weighted' by default
    :return:
    average: ['micro', 'macro', 'weighted'] In the ternary case, it should be 'micro' or 'weighted'
    """

    if df is None:
        if csv_file_path is None:
            raise ValueError(
                "Must provide a DataFrame (df) or a csv file path (csv_file_path)"
            )
        df = pd.read_csv(csv_file_path)

    y = true_labels = df["label"]
    y_hat = predicted_labels = df["predicted_label"]
    try:
        accuracy = accuracy_score(y, y_hat, normalize=True)
        precision = precision_score(
            y, y_hat, labels=LABELS, average=average, zero_division=0
        )
        recall = recall_score(
            true_labels,
            predicted_labels,
            labels=LABELS,
            average=average,
            zero_division=0,
        )
        f1 = f1_score(
            true_labels,
            predicted_labels,
            labels=LABELS,
            average=average,
            zero_division=0,
        )
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        accuracy, precision, recall, f1 = 0, 0, 0, 0

    performance = {
        "F1 Score": f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }

    table = prettytable.PrettyTable()
    table.field_names = ["Metric", "Value"]
    for key, value in performance.items():
        table.add_row([key, value])
    print(table)

    report = (
        classification_report(
            true_labels,
            predicted_labels,
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        )
        if is_return_report
        else None
    )
    return report, performance


def main(
    model_name: str = "Meta-Llama-3",
    model_variant: str = "8B-Instruct",
    seed: int = 0,
    average="weighted",
):
    """
    :param model_name: default Llama-3
    :param model_variant: default 8B-Instruct
    :param seed: in [0, 13, 42]
    :param average: average in ["weighted", "micro", "macro"]:
    :return:
    """

    # Ensure GPT-4o only allows seed 42
    if model_name == "OpenAI" and seed != 42:
        raise ValueError("Only seed 42 is available for OpenAI GPT-4o.")

    chatlm_id = f"{model_name}-{model_variant}"

    # e.g. ['df_Meta-Llama-3-70B-Instruct.csv']
    csv_fns = get_experts_results_fn(
        model_name, model_variant, with_results=False, with_adapters=False
    )
    csv_fp = f"{PROJ_DIR}/{EXPERIMENTS_PATH}/{chatlm_id}_seed-{seed}/{csv_fns[0]}"
    report, results = get_performance_from(
        csv_fp, is_return_report=True, average=average
    )

    print("=" * 50)
    print(f"Performance Report (average = {average}): \n{chatlm_id}_seed_{seed}")
    pprint(report)
    pprint(results)
    print(f"-" * 50)


if __name__ == "__main__":
    fire.Fire(main)
