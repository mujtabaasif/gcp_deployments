import os
import importlib
import pytest
from fastapi.testclient import TestClient

"""
We patch:
- src.dataloader.load_convfinqa_dataset -> returns your single provided record
- src.agent.FinancialReasoningAgent     -> returns a graph with deterministic answers
  for the 4 dialogue questions in the sample.
"""

@pytest.fixture(autouse=True)
def no_external_env(monkeypatch):
    # Ensure service import never tries to hit external LLMs
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(k, raising=False)
    yield


@pytest.fixture
def patched_service(monkeypatch):
    # ---- Build the exact record you provided ----
    class Doc:
        pre_text = (
            "26 | 2009 annual report in fiscal 2008 , revenues in the credit union "
            "systems and services business segment increased 14% ( 14 % ) from fiscal 2007 . "
            "all revenue components within the segment experienced growth during fiscal 2008 . "
            "license revenue generated the largest dollar growth in revenue as episys ae , our flagship "
            "core processing system aimed at larger credit unions , experienced strong sales throughout the year . "
            "support and service revenue , which is the largest component of total revenues for the credit union segment , "
            "experienced 34 percent growth in eft support and 10 percent growth in in-house support . gross profit in this "
            "business segment increased $ 9344 in fiscal 2008 compared to fiscal 2007 , due primarily to the increase in license "
            "revenue , which carries the highest margins . liquidity and capital resources we have historically generated positive "
            "cash flow from operations and have generally used funds generated from operations and short-term borrowings on our "
            "revolving credit facility to meet capital requirements . we expect this trend to continue in the future . the company "
            "2019s cash and cash equivalents increased to $ 118251 at june 30 , 2009 from $ 65565 at june 30 , 2008 . the following "
            "table summarizes net cash from operating activities in the statement of cash flows : 2009 2008 2007 ."
        )
        post_text = (
            "year ended june 30 , cash provided by operations increased $ 25587 to $ 206588 for the fiscal year ended june 30 , 2009 "
            "as compared to $ 181001 for the fiscal year ended june 30 , 2008 . this increase is primarily attributable to a decrease "
            "in receivables compared to the same period a year ago of $ 21214 . this decrease is largely the result of fiscal 2010 annual "
            "software maintenance billings being provided to customers earlier than in the prior year , which allowed more cash to be collected "
            "before the end of the fiscal year than in previous years . further , we collected more cash overall related to revenues that will be "
            "recognized in subsequent periods in the current year than in fiscal 2008 . cash used in investing activities for the fiscal year "
            "ended june 2009 was $ 59227 and includes $ 3027 in contingent consideration paid on prior years 2019 acquisitions . cash used in "
            "investing activities for the fiscal year ended june 2008 was $ 102148 and includes payments for acquisitions of $ 48109 , plus $ 1215 "
            "in contingent consideration paid on prior years 2019 acquisitions . capital expenditures for fiscal 2009 were $ 31562 compared to $ 31105 "
            "for fiscal 2008 . cash used for software development in fiscal 2009 was $ 24684 compared to $ 23736 during the prior year . net cash used "
            "in financing activities for the current fiscal year was $ 94675 and includes the repurchase of 3106 shares of our common stock for $ 58405 , "
            "the payment of dividends of $ 26903 and $ 13489 net repayment on our revolving credit facilities . cash used in financing activities was "
            "partially offset by proceeds of $ 3773 from the exercise of stock options and the sale of common stock ( through the employee stock purchase plan ) "
            "and $ 348 excess tax benefits from stock option exercises . during fiscal 2008 , net cash used in financing activities for the fiscal year was "
            "$ 101905 and includes the repurchase of 4200 shares of our common stock for $ 100996 , the payment of dividends of $ 24683 and $ 429 net repayment "
            "on our revolving credit facilities . cash used in financing activities was partially offset by proceeds of $ 20394 from the exercise of stock options "
            "and the sale of common stock and $ 3809 excess tax benefits from stock option exercises . beginning during fiscal 2008 , us financial markets and many "
            "of the largest us financial institutions have been shaken by negative developments in the home mortgage industry and the mortgage markets , and particularly "
            "the markets for subprime mortgage-backed securities . since that time , these and other such developments have resulted in a broad , global economic downturn . "
            "while we , as is the case with most companies , have experienced the effects of this downturn , we have not experienced any significant issues with our current "
            "collection efforts , and we believe that any future impact to our liquidity will be minimized by cash generated by recurring sources of revenue and due to our access "
            "to available lines of credit. ."
        )
        table = {
            "Year ended June 30, 2009": {
                "net income": 103102.0,
                "non-cash expenses": 74397.0,
                "change in receivables": 21214.0,
                "change in deferred revenue": 21943.0,
                "change in other assets and liabilities": -14068.0,
                "net cash from operating activities": 206588.0,
            },
            "2008": {
                "net income": 104222.0,
                "non-cash expenses": 70420.0,
                "change in receivables": -2913.0,
                "change in deferred revenue": 5100.0,
                "change in other assets and liabilities": 4172.0,
                "net cash from operating activities": 181001.0,
            },
            "2007": {
                "net income": 104681.0,
                "non-cash expenses": 56348.0,
                "change in receivables": -28853.0,
                "change in deferred revenue": 24576.0,
                "change in other assets and liabilities": 17495.0,
                "net cash from operating activities": 174247.0,
            },
        }

    class Dialogue:
        conv_questions = [
            "what is the net cash from operating activities in 2009?",
            "what about in 2008?",
            "what is the difference?",
            "what percentage change does this represent?",
        ]
        conv_answers = ["206588", "181001", "25587", "14.1%"]
        turn_program = [
            "206588",
            "181001",
            "subtract(206588, 181001)",
            "subtract(206588, 181001), divide(#0, 181001)",
        ]
        executed_answers = [206588.0, 181001.0, 25587.0, 0.14136]
        qa_split = [False, False, False, False]

    class Record:
        def __init__(self):
            self.id = "Single_JKHY/2009/page_28.pdf-3"
            self.doc = Doc()
            self.dialogue = Dialogue()

    def fake_loader(_):
        return [Record()], []

    # Patch the loader
    import src.dataloader as dl
    monkeypatch.setattr(dl, "load_convfinqa_dataset", fake_loader, raising=True)

    # Patch the agent to return a deterministic graph
    class DummyGraph:
        def __init__(self):
            # Precompute values from the table
            self.ncfa_2009 = 206588
            self.ncfa_2008 = 181001
            self.diff = self.ncfa_2009 - self.ncfa_2008
            self.pct = self.diff / self.ncfa_2008  # 0.14136

        def _answer(self, q: str) -> str:
            tq = q.lower().strip().rstrip("?")
            if "net cash from operating activities in 2009" in tq:
                return "206588"
            if "what about in 2008" in tq or "in 2008" in tq:
                return "181001"
            if "what is the difference" in tq:
                return "25587"
            if "percentage change" in tq or "percent change" in tq:
                return "14.1%"
            # default: echo
            return f"echo: {q}"

        def invoke(self, state, start=None):
            ans = self._answer(state.question or "")
            return dict(
                question=state.question,
                context=state.context,
                retriever=state.retriever,
                steps_generated=state.steps_generated,
                executor=state.executor,
                answer=ans,
                retrieval_history=state.retrieval_history,
                generation_history=state.generation_history,
                history=(state.history or []) + [
                    {"role": "user", "content": state.question},
                    {"role": "assistant", "content": ans},
                ],
                skip_generation=state.skip_generation,
            )

    class DummyAgent:
        def __init__(self, llm, prompt_builder):
            self.llm = llm
            self.prompt_builder = prompt_builder
        def build_graph(self):
            return DummyGraph()

    import src.agent as agent_mod
    monkeypatch.setattr(agent_mod, "FinancialReasoningAgent", DummyAgent, raising=True)

    # Import the service after patching
    if "../src.service" in globals():
        import src.service as service
        importlib.reload(service)
    else:
        import src.service as service

    client = TestClient(service.app)
    return service, client


@pytest.fixture
def client(patched_service):
    _, client = patched_service
    return client
