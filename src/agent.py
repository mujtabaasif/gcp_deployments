from typing import List, Dict

from langgraph.graph import StateGraph, END

from .dataloader import State
from .prompts import PromptBuilder
from .utils import extract_steps_with_regex


class FinancialReasoningAgent:
    def __init__(self, llm, prompt_builder: PromptBuilder):
        self.llm = llm
        self.prompts = prompt_builder

    def reformulate_question(self, question: str, history: List[Dict[str, str]]) -> str:
        try:
            prompt = self.prompts.build_reformulation_prompt(history, question)
            return self.llm.invoke(prompt).content.strip()
        except Exception as e:
            print(f"Error in reformulate_question: {e}")
            return question

    def retrieve(self, state: State) -> str:
        prompt = self.prompts.build_retrieval_prompt(
            state=state,
            history=state.retrieval_history
        )
        return self.llm.invoke(prompt).content.strip()

    def generate_steps(self, state: State) -> str:
        prompt = self.prompts.build_step_generation_prompt(
            state=state,
            history=state.generation_history
        )
        return self.llm.invoke(prompt).content.strip()

    def execute_program(self, state: State) -> str:
        steps = extract_steps_with_regex(state.steps_generated)
        if isinstance(steps, str):
            return state.steps_generated
        if not steps:
            return state.retriever
        return str(steps[-1][-1])

    def format_answer(self, state: State) -> str:
        prompt = self.prompts.build_formatting_prompt(state.question, state.executor)
        return self.llm.invoke(prompt).content.strip()

    def update_history(self, state: State) -> State:
        state.retrieval_history.append({'user': state.question, 'assistant': state.retriever})
        state.generation_history.append({'user': state.question, 'assistant': state.steps_generated})
        state.history.append({'user': state.question, 'assistant': state.answer})
        return state

    def build_graph(self):
        builder = StateGraph(State)

        # LangGraph node functions bound to instance
        builder.add_node("reformulate", self._reformulate_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("generate_steps", self._generate_steps_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("format_answer", self._format_answer_node)
        builder.add_node("update_history", self._update_history_node)

        builder.set_entry_point("reformulate")
        builder.add_edge("reformulate", "retrieve")

        def should_skip_generation(state: State) -> str:
            return "format_answer" if state.skip_generation else "generate_steps"

        builder.add_conditional_edges("retrieve", should_skip_generation, {
            "format_answer": "format_answer",
            "generate_steps": "generate_steps"
        })

        builder.add_edge("generate_steps", "execute")
        builder.add_edge("execute", "format_answer")
        builder.add_edge("format_answer", "update_history")
        builder.add_edge("update_history", END)

        return builder.compile()

    def _looks_like_final_answer(self, text: str) -> bool:
        text = text.strip().lower()
        if text.startswith("answer:"):
            return True
        # if only contains a number or boolean
        if text.isdigit() or text.replace('.', '', 1).isdigit() or text in ['true', 'false']:
            return True
        return False

    def _reformulate_node(self, state: State) -> State:
        state.question = self.reformulate_question(state.question, state.history)
        return state

    def _retrieve_node(self, state: State) -> State:
        state.retriever = self.retrieve(state)
        if self._looks_like_final_answer(state.retriever):
            state.skip_generation = True
            state.steps_generated = state.retriever
            state.executor = state.retriever
        else:
            state.skip_generation = False
        return state

    def _generate_steps_node(self, state: State) -> State:
        state.steps_generated = self.generate_steps(state)
        return state

    def _execute_node(self, state: State) -> State:
        state.executor = self.execute_program(state)
        return state

    def _format_answer_node(self, state: State) -> State:
        state.answer = self.format_answer(state)
        return state

    def _update_history_node(self, state: State) -> State:
        return self.update_history(state)
