from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.dataloader import load_convfinqa_dataset_eval

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "Qwen/Qwen3-8B"
NUM_WORKERS = 32


def call_llm(prompt: str) -> bool:
    try:
        response = requests.post(
            VLLM_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 4096,
            },
        )
        content = response.json()['choices'][0]['message']['content'].strip().upper()
        return True if "TRUE" in content else False
    except Exception as e:
        print("Error:", e)
        return None


def evaluate_answer(row):
    prompt = f"""
    You are a helpful and precise grader. 
    Determine whether the model's final answer is equivalent to the expected answer, ignoring formatting differences such as precision errors, commas, currency symbols, and spacing.
    
    Question: {row['conv_questions']}
    Model Answer: {row['answer']}
    Expected Answer: {row['conv_answers']}
    Does the model's answer match the expected answer in value and meaning? Respond only with 'TRUE' or 'FALSE'.
    """
    return call_llm(prompt)


def evaluate_retriever(row):
    prompt = f"""
    Check whether the retrieved information contains all the values needed to compute the expected answer.
    
    Question: {row['conv_questions']}
    Retriever Output: {row['retriever']}
    Target Computation Program: {row['turn_program']}
    
    Respond only with 'TRUE' or 'FALSE'.
    """
    return call_llm(prompt)


def evaluate_step_quality(row):
    prompt = f"""
    Determine if the model's generated code or steps are logically correct and lead to the same result as the expected answer.
    
    Question: {row['conv_questions']}
    Generated Steps: {row['steps_generated']}
    Expected Final Answer: {row['conv_answers']}
    
    Respond only with 'TRUE' or 'FALSE'.
    """
    return call_llm(prompt)


def evaluate_format_match(row):
    prompt = f"""
    Determine whether the model's final answer has the same format as the expected answer. Consider currency symbols, percentage signs, commas, and number formatting.
    Model Answer: {row['answer']}
    Expected Answer: {row['conv_answers']}
    Respond only with 'TRUE' or 'FALSE'.
    """
    return call_llm(prompt)


def flatten_df(data) -> pd.DataFrame:
    rows = []
    for record in data:
        for index in range(record.features.num_dialogue_turns):
            rows.append({
                'id': record.id,
                'index': index,
                'conv_questions': record.dialogue.conv_questions[index],
                'conv_answers': record.dialogue.conv_answers[index],
                'turn_program': record.dialogue.turn_program[index],
                'executed_answers': record.dialogue.executed_answers[index],
                'qa_split': record.dialogue.qa_split[index],
                'reformulated_question': record.response_state[index].question,
                'retriever': record.response_state[index].retriever,
                'skip_generation': record.response_state[index].skip_generation,
                'steps_generated': record.response_state[index].steps_generated,
                'executor': record.response_state[index].executor,
                'answer': record.response_state[index].answer,
                'history': record.response_state[index].history,
            })
    return pd.DataFrame(rows)


def run_evaluation(df: pd.DataFrame, evaluator_fn, column_name: str, output_csv: str) -> pd.DataFrame:
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(evaluator_fn, row): idx for idx, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {column_name}"):
            idx = futures[future]
            try:
                result = future.result()
            except Exception:
                result = "ERROR"
            results.append((idx, result))

    result_dict = dict(results)
    df[column_name] = df.index.map(result_dict.get)
    df.to_csv(output_csv, index=False)
    return df


def visualize_turn_level_accuracy(df: pd.DataFrame):
    turn_acc = df.groupby("index")["llm_answer_correct"].mean().reset_index()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=turn_acc, x="index", y="llm_answer_correct", marker="o")
    plt.title("Turn-Level Answer Accuracy")
    plt.xlabel("Turn Index")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../plots/turn_level_accuracy.png")


def visualize_correlation_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    corr = df[['llm_answer_correct', 'llm_retriever_quality', 'llm_step_quality', 'llm_format_match']].astype(
        int).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Evaluation Metrics")
    plt.tight_layout()
    plt.savefig("../plots/correlation_heatmap.png")


def visualize_skip_generation(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    conf_matrix = df.groupby(["code_gen_needed", "skip_generation"]).size().unstack(fill_value=0)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
    plt.title("Skip Generation Confusion Matrix")
    plt.xlabel("Model Chose to Skip")
    plt.ylabel("Code Generation Needed")
    plt.tight_layout()
    plt.savefig("../plots/skip_generation_stats.png")


def visualize_answer_accuracy(df: pd.DataFrame, save_path="../plots/answer_accuracy_overview.png"):
    df["llm_answer_correct"] = df["llm_answer_correct"].astype(bool)
    df["llm_retriever_quality"] = df["llm_retriever_quality"].astype(bool)
    df["llm_step_quality"] = df["llm_step_quality"].astype(bool)
    df["code_gen_needed"] = df["code_gen_needed"].astype(bool)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))

    cm1 = confusion_matrix(df["llm_retriever_quality"], df["llm_answer_correct"], labels=[True, False])
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[0][0])
    axes[0][0].set_title("Retriever Quality vs Answer Accuracy")
    axes[0][0].set_xlabel("Answer Correct")
    axes[0][0].set_ylabel("Retriever Correct")
    axes[0][0].set_xticklabels(["True", "False"])
    axes[0][0].set_yticklabels(["True", "False"])

    cm2 = confusion_matrix(df["llm_step_quality"], df["llm_answer_correct"], labels=[True, False])
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=axes[0][1])
    axes[0][1].set_title("Step Quality vs Answer Accuracy")
    axes[0][1].set_xlabel("Answer Correct")
    axes[0][1].set_ylabel("Step Correct")
    axes[0][1].set_xticklabels(["True", "False"])
    axes[0][1].set_yticklabels(["True", "False"])

    cm3 = confusion_matrix(~df["llm_retriever_quality"], df["llm_answer_correct"], labels=[True, False])
    sns.heatmap(cm3, annot=True, fmt="d", cmap="Reds", ax=axes[1][0])
    axes[1][0].set_title("Incorrect Retriever vs Answer Accuracy")
    axes[1][0].set_xlabel("Answer Correct")
    axes[1][0].set_ylabel("Retriever Incorrect")
    axes[1][0].set_xticklabels(["True", "False"])
    axes[1][0].set_yticklabels(["True", "False"])

    cm4 = confusion_matrix(~df["llm_step_quality"], df["llm_answer_correct"], labels=[True, False])
    sns.heatmap(cm4, annot=True, fmt="d", cmap="Oranges", ax=axes[1][1])
    axes[1][1].set_title("Incorrect Steps vs Answer Accuracy")
    axes[1][1].set_xlabel("Answer Correct")
    axes[1][1].set_ylabel("Steps Incorrect")
    axes[1][1].set_xticklabels(["True", "False"])
    axes[1][1].set_yticklabels(["True", "False"])

    errors = df[df["llm_answer_correct"] == False].copy()
    errors["error_reason"] = errors.apply(
        lambda row: (
            "Retriever" if not row["llm_retriever_quality"]
            else "Step" if not row["llm_step_quality"]
            else "Format" if not row["llm_format_match"]
            else "Other"
        ),
        axis=1
    )
    sns.countplot(data=errors, x="error_reason", ax=axes[2][0], palette="Set2")
    axes[2][0].set_title("Primary Cause of Incorrect Answers")
    axes[2][0].set_ylabel("Count")
    axes[2][0].grid(True, linestyle="--", alpha=0.5)

    turn_acc = df.groupby("index")["llm_answer_correct"].mean().reset_index()
    sns.barplot(data=turn_acc, x="index", y="llm_answer_correct", ax=axes[2][1], palette="coolwarm")
    axes[2][1].set_title("Turn-Level Answer Accuracy")
    axes[2][1].set_xlabel("Turn Index")
    axes[2][1].set_ylabel("Accuracy")
    axes[2][1].set_ylim(0, 1.05)
    axes[2][1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.suptitle("Detailed Answer Accuracy Analysis", fontsize=16, y=1.02)
    plt.savefig(save_path)
    plt.close()


def visualize_retriever_quality(df: pd.DataFrame, save_path="../plots"):
    df["llm_answer_correct"] = df["llm_answer_correct"].astype(bool)
    df["llm_retriever_quality"] = df["llm_retriever_quality"].astype(bool)

    no_code = df[df['code_gen_needed'] == False]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    sns.countplot(data=no_code, x='llm_retriever_quality', ax=axes[0][0], palette="Greens")
    axes[0][0].set_title("Retriever Quality (No Code Gen Needed)")
    axes[0][0].set_xlabel("Retriever Correct (True/False)")
    axes[0][0].set_ylabel("Count")
    axes[0][0].grid(True, linestyle='--', alpha=0.5)

    sns.countplot(data=df, x='llm_retriever_quality', ax=axes[0][1], palette="Blues")
    axes[0][1].set_title("Retriever Quality (All Cases)")
    axes[0][1].set_xlabel("Retriever Correct (True/False)")
    axes[0][1].set_ylabel("Count")
    axes[0][1].grid(True, linestyle='--', alpha=0.5)

    cm_nc = confusion_matrix(
        no_code["llm_answer_correct"],
        no_code["llm_retriever_quality"],
        labels=[True, False],
        normalize='true'
    )
    sns.heatmap(cm_nc, annot=True, fmt=".2f", cmap="Greens", ax=axes[1][0],
                xticklabels=["Correct", "Incorrect"], yticklabels=["Correct", "Incorrect"])
    axes[1][0].set_title("Confusion Matrix (No Code Gen)")
    axes[1][0].set_xlabel("Retriever Correct (Pred)")
    axes[1][0].set_ylabel("Answer Correct (True)")

    cm_all = confusion_matrix(
        df["llm_answer_correct"],
        df["llm_retriever_quality"],
        labels=[True, False],
        normalize='true'
    )
    sns.heatmap(cm_all, annot=True, fmt=".2f", cmap="Blues", ax=axes[1][1],
                xticklabels=["Correct", "Incorrect"], yticklabels=["Correct", "Incorrect"])
    axes[1][1].set_title("Confusion Matrix (All Cases)")
    axes[1][1].set_xlabel("Retriever Correct (Pred)")
    axes[1][1].set_ylabel("Answer Correct (True)")

    plt.suptitle("Retriever Quality Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../plots/retriever_quality_combined.png")
    plt.close()


def plot_code_generation_analysis(df: pd.DataFrame, save_path="../plots/code_generation_analysis.png"):
    df = df.copy()

    # Cast to bool
    for col in ["llm_step_quality", "llm_answer_correct", "code_gen_needed", "skip_generation",
                "llm_retriever_quality"]:
        df[col] = df[col].astype(bool)

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 18))

    bad_retrieval = df[df["llm_retriever_quality"] == False]
    cm_a = confusion_matrix(bad_retrieval["llm_retriever_quality"], bad_retrieval["llm_step_quality"],
                            labels=[True, False])
    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Reds", ax=axes[0][0])
    axes[0][0].set_title("Impact of Incorrect Retrieval on Step Quality")
    axes[0][0].set_xlabel("Step Quality")
    axes[0][0].set_ylabel("Retrieval Correct")
    axes[0][0].set_xticklabels(["True", "False"])
    axes[0][0].set_yticklabels(["True", "False"])

    good_retrieval = df[df["llm_retriever_quality"] == True]
    cm_b = confusion_matrix(good_retrieval["llm_step_quality"], good_retrieval["llm_answer_correct"],
                            labels=[True, False])
    sns.heatmap(cm_b, annot=True, fmt="d", cmap="YlGnBu", ax=axes[0][1])
    axes[0][1].set_title("Step Quality vs Answer Accuracy (Good Retrieval)")
    axes[0][1].set_xlabel("Answer Correct")
    axes[0][1].set_ylabel("Step Correct")
    axes[0][1].set_xticklabels(["True", "False"])
    axes[0][1].set_yticklabels(["True", "False"])

    df = good_retrieval.copy()

    cm1 = confusion_matrix(df["llm_step_quality"], df["llm_answer_correct"], labels=[True, False])
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[1][0])
    axes[1][0].set_title("Step Quality vs Answer Accuracy")
    axes[1][0].set_xlabel("Answer Correct")
    axes[1][0].set_ylabel("Step Correct")
    axes[1][0].set_xticklabels(["True", "False"])
    axes[1][0].set_yticklabels(["True", "False"])

    cm2 = confusion_matrix(df["code_gen_needed"], df["skip_generation"], labels=[True, False])
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Purples", ax=axes[1][1])
    axes[1][1].set_title("Code Gen Needed vs Model Skip Decision")
    axes[1][1].set_xlabel("Skip Generation (Predicted)")
    axes[1][1].set_ylabel("Code Gen Needed")
    axes[1][1].set_xticklabels(["True", "False"])
    axes[1][1].set_yticklabels(["True", "False"])

    cm3 = confusion_matrix(df["skip_generation"], df["llm_step_quality"], labels=[True, False])
    sns.heatmap(cm3, annot=True, fmt="d", cmap="Oranges", ax=axes[2][0])
    axes[2][0].set_title("Skip Decision vs Step Quality")
    axes[2][0].set_xlabel("Step Correct")
    axes[2][0].set_ylabel("Skip Generation")
    axes[2][0].set_xticklabels(["True", "False"])
    axes[2][0].set_yticklabels(["True", "False"])

    cm4 = confusion_matrix(df["code_gen_needed"], df["llm_step_quality"], labels=[True, False])
    sns.heatmap(cm4, annot=True, fmt="d", cmap="Greens", ax=axes[2][1])
    axes[2][1].set_title("Code Gen Needed vs Step Quality")
    axes[2][1].set_xlabel("Step Correct")
    axes[2][1].set_ylabel("Code Gen Needed")
    axes[2][1].set_xticklabels(["True", "False"])
    axes[2][1].set_yticklabels(["True", "False"])

    bad_steps = df[df["llm_step_quality"] == False].copy()
    bad_steps["step_error_reason"] = bad_steps.apply(
        lambda row: (
            "Skipped but Needed" if row["code_gen_needed"] and row["skip_generation"]
            else "Bad Execution" if not row["skip_generation"]
            else "Other"
        ),
        axis=1
    )
    sns.countplot(data=bad_steps, x="step_error_reason", ax=axes[3][0], palette="Set2")
    axes[3][0].set_title("Causes of Bad Code Generation")
    axes[3][0].set_ylabel("Count")
    axes[3][0].grid(True, linestyle="--", alpha=0.5)

    turn_acc = df.groupby("index")["llm_step_quality"].mean().reset_index()
    sns.barplot(data=turn_acc, x="index", y="llm_step_quality", ax=axes[3][1], palette="coolwarm")
    axes[3][1].set_title("Turn-Level Code Generation Accuracy")
    axes[3][1].set_xlabel("Turn Index")
    axes[3][1].set_ylabel("Accuracy")
    axes[3][1].set_ylim(0, 1.05)
    axes[3][1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.suptitle("Code Generation Analysis", fontsize=18, y=1.02)
    plt.savefig(save_path)
    plt.close()


def compute_and_visualize_metrics(df: pd.DataFrame):
    for col in ['llm_answer_correct', 'llm_retriever_quality', 'llm_step_quality', 'llm_format_match']:
        df[col] = df[col].astype(bool)

    metrics = {
        "Correctness Accuracy": df['llm_answer_correct'].mean(),
        "Step Quality Accuracy": df['llm_step_quality'].mean(),
        "Retriever Accuracy": df['llm_retriever_quality'].mean(),
        "Formatting Accuracy": df['llm_format_match'].mean(),
    }

    TP = ((df['llm_retriever_quality']) & (df['llm_answer_correct'])).sum()
    FP = ((df['llm_retriever_quality']) & (~df['llm_answer_correct'])).sum()
    FN = ((~df['llm_retriever_quality']) & (df['llm_answer_correct'])).sum()
    metrics["Retriever Precision"] = TP / (TP + FP + 1e-6)
    metrics["Retriever Recall"] = TP / (TP + FN + 1e-6)

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    visualize_turn_level_accuracy(df)
    visualize_correlation_heatmap(df)
    visualize_skip_generation(df)
    visualize_retriever_quality(df)
    visualize_answer_accuracy(df)
    plot_code_generation_analysis(df)


if __name__ == "__main__":
    EVALUATE_RESULTS = False
    if EVALUATE_RESULTS:
        data = load_convfinqa_dataset_eval("../data/convfinqa_with_agent_predictions.json")
        df = flatten_df(data)
        tqdm.pandas()
        df = run_evaluation(df, evaluate_answer, "llm_answer_correct", "llm_eval_answer.csv")
        df = run_evaluation(df, evaluate_retriever, "llm_retriever_quality", "llm_eval_retriever.csv")
        df = run_evaluation(df, evaluate_step_quality, "llm_step_quality", "llm_eval_final.csv")
        df = run_evaluation(df, evaluate_format_match, "llm_format_match", "convfinqa_llm_eval_predictions.csv")

    df = pd.read_csv("../data/convfinqa_llm_eval_predictions.csv")
    df["code_gen_needed"] = df["turn_program"].apply(
        lambda x: any(op in x for op in {"add", "subtract", "multiply", "divide", "exp"} if isinstance(x, str)))
    compute_and_visualize_metrics(df)
