"""
Main typer app for ConvFinQA
"""

import typer
from langchain_openai import ChatOpenAI
from langgraph.graph import START
from rich import print as rich_print

from src.agent import FinancialReasoningAgent
from src.dataloader import load_convfinqa_dataset, State
from src.prompts import PromptBuilder
from src.utils import get_context

app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)

MODEL_ID = "Qwen/Qwen3-8B"
BASE_URL = "http://localhost:8000/v1"
DATA_PATH = "../data/convfinqa_dataset.json"
train_df, test_df = load_convfinqa_dataset(DATA_PATH)
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="API_KEY",  # Replace with your actual API key
    model=MODEL_ID,
)


@app.command()
def chat(
        record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record"""

    record = next((r for r in train_df if r.id == record_id), None)
    if not record:
        rich_print(f"[red]Record with ID {record_id} not found.[/red]")
        return
    rich_print(f"[blue][bold]Record ID:[/bold] {record.id}[/blue]")
    rich_print(f"[blue][bold]Document:[/bold] {record.doc.pre_text}[/blue]")
    rich_print(f"[blue][bold]Table:[/bold] {record.doc.table}[/blue]")
    rich_print(f"[blue][bold]Post Text:[/bold] {record.doc.post_text}[/blue]")

    prompt_builder = PromptBuilder(train_df[:3])
    agent = FinancialReasoningAgent(llm, prompt_builder)
    graph = agent.build_graph()
    context = get_context(record)
    state = State(
        question="",
        context=context,
        retriever="",
        steps_generated="",
        executor="",
        answer="",
        retrieval_history=[],
        generation_history=[],
        history=[],
        skip_generation=False,
    )
    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        state.question = message
        state_dict = graph.invoke(state, start=START)
        state = State(**state_dict)
        response = state.answer.strip()

        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")


if __name__ == "__main__":
    app()
