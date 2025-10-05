import asyncio
import json
from typing import Dict, List

import aiohttp

from scripts.evaluator import MODEL_ID
from src.dataloader import load_convfinqa_dataset
from src.prompts import PromptBuilder, build_user_prompt
from src.utils import get_context


async def chat(session, system_prompt: str, user_prompt: str, history: List[Dict], base_url="http://localhost:8000",
               model=MODEL_ID):
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        async with session.post(f"{base_url}/v1/chat/completions", json={
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096
        }) as resp:
            result = await resp.json()
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error during chat: {e}")
        return None


async def process_record(session, record, system_prompt, cot):
    context = get_context(record)
    history = []
    llm_answers = []

    for question in record.dialogue.conv_questions:
        user_prompt = build_user_prompt(question, context, cot)
        answer = await chat(session, system_prompt, user_prompt, history)
        llm_answers.append(answer)
        history.append({"user": question, "assistant": answer})
        print(f"Q: {question}\nA: {answer}\n Expected Answer: {record.dialogue.conv_answers[len(llm_answers) - 1]}\n")

    record.dialogue.llm_answers = llm_answers
    return record


async def main(df, system_prompt, cot, MODEL_ID=MODEL_ID):
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[
            process_record(session, record, system_prompt, cot)
            for record in df
        ])

    with open(f"../data/convfinqa_with_{MODEL_ID.replace('/', '_')}.json", "w") as f:
        json.dump([record.model_dump() for record in results], f, indent=2)


if __name__ == '__main__':
    MODEL_ID = "Qwen/Qwen3-8B"  # or "meta-llama/Llama-3.1-8B-Instruct"
    train_df, test_df = load_convfinqa_dataset('../data/convfinqa_dataset.json')
    train_df = train_df[:3]
    prompt_builder = PromptBuilder(train_df[:3])
    asyncio.run(main(train_df, prompt_builder.build_system_prompt(), prompt_builder.build_cot_prompt(train_df[0])))
