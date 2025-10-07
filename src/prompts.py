from typing import List, Dict, Tuple
from .dataloader import State, ConvFinQARecord
from .utils import get_context, format_history, extract_steps


def build_user_prompt(question: str, context: str, cot: str) -> str:
    user_prompt = (
        f"Examples:\n{cot}\n\n"
        f"Now answer the following question: {question} \n\n"
        f"DOCUMENT:\n{context}\n\n"
    )
    return user_prompt


class PromptBuilder:
    def __init__(self, record_examples: List[ConvFinQARecord], generate_examples: str | None = None,
                 retrieve_examples: str | None = None):
        self.record_examples = record_examples
        if generate_examples is None and retrieve_examples is None:
            self.build_few_shot_prompt()
        self.generate_examples = generate_examples
        self.retrieve_examples = retrieve_examples

    def build_few_shot_prompt(self) -> Tuple[str, str]:
        self.retrieve_examples = ""
        self.generate_examples = ""
        for record in self.record_examples:
            context = get_context(record)
            for i, q in enumerate(record.dialogue.conv_questions):
                a = record.dialogue.turn_program[i]
                if '(' in a and ')' in a:
                    self.generate_examples += f"Q: {q}\nContext: {context}\nProgram: {a}\n\n"
                    self.retrieve_examples += f"Q: {q}\nContext: {context}\n NO DIRECT RELEVANT INFORMATION FOUND PERFORM PROGRAM GENERATION{a}\n\n"
                else:
                    self.retrieve_examples += f"Q: {q}\nContext: {context}\nAnswer: {a}\n\n"
        return self.generate_examples.strip(), self.retrieve_examples.strip()

    def build_system_prompt(self) -> str:
        system_function = (
            "You are an expert assistant that helps solve complex Conversational FinQA "
            "problems involving numerical and logical reasoning over financial documents."
        )

        context_part = (
            "You will be provided with a DOCUMENT that includes a financial table. "
            "The table is clearly marked between [Table Begin] and [Table End]."
            "Table is structured in following format:"
            "Column 1: contains the expense categories, corporation names, countries etc."
            "The rest of the columns contain the value by year, date or time (refer to table header)\n"
        )

        reasoning_steps = (
            "Your task is to reason step by step as follows:\n"
            "1. Reformulate the user's question to make it clear and self-contained.\n"
            "2. Retrieve relevant information from the context to answer the question.\n"
            "3. Generate a program that can answer the question based on the retrieved information.\n"
            "4. Execute the program to get the final answer.\n"
            "5. Format the final answer to ensure it is a valid representation of the answer.\n"
        )
        return f"{system_function}\n\n{context_part}\n\n{reasoning_steps}"


    @staticmethod
    def build_reformulation_prompt(history: List[Dict[str, str]], question: str) -> str:
        return f"""
        You are a helpful assistant that reformulates vague or coreferential user questions in a conversation.
        Rewrite the user's current question so that it is fully self-contained using the prior conversation history.
        
        Ensure that you follow these rules:
        0. ALWAYS maintain the order from the conversation history. Never change the flow of the conversation.
        1. Do not add any new information that is not present in the conversation history.
        2. Make the question clear and specific, avoiding any ambiguity.
        3. Maintain the original intent of the question while making it more explicit.
        4. Do not change the meaning of the question.
        5. Do not include any additional context or explanations.
        6. Only use the relevant parts of the conversation history to rewrite the question.
        7. If the conversation history does not provide enough information to rewrite the question, reformulate it as best as possible based on the available context.
        
        Conversation History: {format_history(history)}
        Current Question: {question}
        Rewritten: """

    def build_retrieval_prompt(self, state: State, history: List[Dict[str, str]]) -> str:
        return f"""
        You are a financial data retriever step. 
        Your task is to only extract relevant information from the context to help answer the user's question.
        Here is how the information is structured:
        - You are provided with a DOCUMENT that includes a financial table.
        - The table is clearly marked between [Table Begin] and [Table End].
            - Table is structured in following format:
                - Column 1: contains the expense categories, corporation names, countries etc.
                - Top row: contains the header of the table, which may include years, dates or time.
                - The rest of the columns contain the value by year, date or time
        - You are provided with EXAMPLES of how to extract information from the context.
        - You also have access to the conversation history, which may provide additional context.
        
        Always follow these rules:
        1. Only extract information that is directly relevant to the user's question.
        2. Do not include any information that is not directly related to the question.
        3. Do not perform any calculations or generate programs.
        
        Note: ONLY Return the retrieved information, do not include any additional explanations or context. Refer to the examples for guidance on how to extract information.
        Here are some examples for reference:
        {self.retrieve_examples}
        
        Conversation History: {format_history(history)}
        CURRENT DOCUMENT: {state.context}
        Question: {state.question}
        Answer:
        """

    def build_step_generation_prompt(self, state: State, history: List[Dict[str, str]]) -> str:
        return f"""
        You are a financial step generator.
        Think step by step to generate a program that can answer the user's question based on the retrieved information.
        
        Always follow these rules:
        1. Use only the operators: add, subtract, multiply, divide, exp. e.g add(1, 2), subtract(3, 1), multiply(2, 3), divide(6, 2), exp(2, 3), "subtract(60.94, 25.14), divide(#0, 25.14)", "divide(3.8, 35.1), multiply(#0, 100)" .
        2. The order of operations is important, so ensure the program reflects the correct sequence.
        4. If value can be directly answered from the context, return it as is without any additional information or explanation. e.g., if the answer is 100, return 100.
        5. If no program can be generated, return: NO PROGRAM GENERATED.
        6. Don't perform any calculations, just generate the program.
        
        Here are some examples for reference:
        ### Examples ###
        {self.generate_examples}
        
        ### Conversation History ###
        {format_history(history)}
        
        ### Current Turn ###
        Question: {state.question}
        Context: {state.context}
        Retrieved Info: {state.retriever}
        Steps:
        """

    @staticmethod
    def build_formatting_prompt(question: str, raw_answer: str) -> str:
        return f"""
        You are content formatter.
        Your task is to format the answer to ensure it is a valid representation of the final answer.
        Current Question: {question}
        Current Answer: {raw_answer}
        
        Only respond with the formatted answer.
        - If the answer is a number, return it as a number.
        - If the answer is a boolean, return it as 'True' or 'False'.
        - If the answer is a date, return it in 'YYYY-MM-DD' format.
        - If the answer is a string, return it as is.
        - If the question asks for a percentage, convert the number to a percentage string (e.g., 0.25 -> '25%')."""

    def build_cot_prompt(self, record: ConvFinQARecord) -> str:
        cot = []
        for index, (question, final_answer, turn_program) in enumerate(
            zip(record.dialogue.conv_questions, record.dialogue.conv_answers, record.dialogue.turn_program)
        ):
            prompt = f"Q{index}: {question}\n"
            context = get_context(record)
            prompt += f"Context: {context}\n"
            prompt += f"A{index}: {final_answer}\n"

            if turn_program:
                operators = extract_steps(turn_program)
                if operators:
                    prompt += "Let's think step by step and breakdown operation below into small reasoning:\n"
                    prompt += f"Operation: {turn_program}\n"
                    for i, op in enumerate(operators):
                        operator, value1, value2, result = op
                        operation_prompt = ''
                        if operator == 'add':
                            operation_prompt = f"Let's Add {value1} and {value2}. {operator}({value1}, {value2}) = {result}"
                        elif operator == 'subtract':
                            operation_prompt = f"Let's Subtract {value2} from {value1}. {operator}({value1}, {value2}) = {result}"
                        elif operator == 'multiply':
                            operation_prompt = f"Let's Multiply {value1} and {value2}. {operator}({value1}, {value2}) = {result}"
                        elif operator == 'divide':
                            operation_prompt = f"Let's Divide {value1} by {value2}. {operator}({value1}, {value2}) = {result}"
                        elif operator == 'exp':
                            operation_prompt = f"Let's Raise {value1} to the power of {value2}. {operator}({value1}, {value2}) = {result}"

                        prompt += f"{operation_prompt}\n"

            cot.append(prompt)

        return "\n\n".join(cot)
