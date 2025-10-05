from typing import Dict, List, Tuple

import pandas as pd


def table_to_markdown(table: Dict[str, Dict[str, float]]) -> str:
    df = pd.DataFrame(table)
    df = df.map(lambda x: x if isinstance(x, float) else x)
    return "\n\n[Table Begin]\n" + df.to_markdown(tablefmt="github") + "\n[Table End]\n\n"


def get_context(record) -> str:
    return record.doc.pre_text + table_to_markdown(record.doc.table) + record.doc.post_text


def perform_operation(operator: str, value1: float, value2: float) -> float:
    if operator == 'add':
        return value1 + value2
    elif operator == 'subtract':
        return value1 - value2
    elif operator == 'multiply':
        return value1 * value2
    elif operator == 'divide':
        if value2 != 0:
            return value1 / value2
        else:
            raise ValueError("Division by zero.")
    elif operator == 'exp':
        return value1 ** value2
    else:
        raise ValueError(f"Unknown operator: {operator}")


def extract_steps(turn_program: str):
    steps = turn_program.split('),')
    memory = {}
    step_list = []
    for step in steps:
        if '(' in step:
            try:
                operator, values = step.split('(')
                operator = operator.strip()
                if operator in ('add', 'subtract', 'multiply', 'divide', 'exp'):
                    value1, value2 = values.rstrip(')').split(',')
                    value1 = float(memory.get(value1.strip(), value1.strip()))
                    value2 = float(memory.get(value2.strip(), value2.strip()))
                    result = perform_operation(operator.strip(), float(value1), float(value2))
                    memory[f"#{len(memory)}"] = result
                    step_list.append((operator.strip(), float(value1), float(value2), round(result, 4)))
                else:
                    return "Invalid operator in turn program."
            except ValueError:
                print(step)
                return "Invalid format in turn program."
    return step_list


def parse_and_eval(expr: str, steps: List[Tuple[str, float, float, float]]) -> float:
    expr = expr.strip()

    try:
        return float(expr)
    except ValueError:
        pass

    # Find outermost operator and its arguments
    op_end = expr.find('(')
    if op_end == -1 or not expr.endswith(')'):
        raise ValueError(f"Malformed expression: {expr}")

    operator = expr[:op_end]
    args_str = expr[op_end + 1:-1]

    # Handle nested arguments with correct comma splitting
    depth = 0
    comma_index = None
    for i, char in enumerate(args_str):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            comma_index = i
            break

    if comma_index is None:
        raise ValueError(f"Could not parse arguments in: {expr}")

    arg1_str = args_str[:comma_index]
    arg2_str = args_str[comma_index + 1:]

    val1 = parse_and_eval(arg1_str, steps)
    val2 = parse_and_eval(arg2_str, steps)
    result = perform_operation(operator, val1, val2)
    steps.append((operator, val1, val2, round(result, 4)))
    return result


def extract_steps_with_regex(turn_program: str) -> List[Tuple[str, float, float, float]]:
    if not turn_program:
        return []
    # check if turn_program contains any operators
    if not any(op in turn_program for op in ['add', 'subtract', 'multiply', 'divide', 'exp']):
        return []
    turn_program = turn_program.lower().replace(' ', '')

    try:
        steps = extract_steps(turn_program)
        if isinstance(steps, str):
            steps = []
            parse_and_eval(turn_program, steps)
        return steps
    except ValueError as e:
        print(f"Error extracting steps from turn program '{turn_program}': {e}")


def format_history(history: List[Dict[str, str]]) -> str:
    return "\n".join([f"Q: {turn['user']}\nA: {turn['assistant']}" for turn in history])
