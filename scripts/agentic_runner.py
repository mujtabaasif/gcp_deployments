import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langgraph.graph import START
from tqdm import tqdm

from src.agent import FinancialReasoningAgent
from src.dataloader import load_convfinqa_dataset, State
from src.prompts import PromptBuilder
from src.utils import get_context


def process_record(record, graph):
    try:
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
            skip_generation=False
        )

        for i, question in enumerate(record.dialogue.conv_questions):
            state.question = question
            state_dict = graph.invoke(state, start=START)
            state = State(**state_dict)
            record.response_state.append(state)

        return record
    except Exception as e:
        print(f"Error processing record {record.id}: {e}")
        return record


if __name__ == "__main__":
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="API_KEY",
        model="Qwen/Qwen3-8B",
    )

    train_df, test_df = load_convfinqa_dataset('../data/convfinqa_dataset.json')

    # Setup prompt builder and agent
    prompt_builder = PromptBuilder(train_df[:3])
    agent = FinancialReasoningAgent(llm, prompt_builder)
    graph = agent.build_graph()
    img_data = graph.get_graph().draw_mermaid_png(
        output_file_path="../plots/financial_graph.png")
    results = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_record, record, graph): record for record in test_df}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing records"):
            print(future.result())
            results.append(future.result())

    with open("../data/convfinqa_with_agent_predictions.json", "w") as f:
        json.dump([record.model_dump() for record in results], f, indent=2)
