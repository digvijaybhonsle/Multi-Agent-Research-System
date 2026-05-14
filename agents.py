from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# LLM
# =========================================================

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.3
)


# =========================================================
# WRITER CHAIN
# =========================================================

writer_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an expert research writer.

Write:
- factual
- structured
- professional
- evidence-based reports

Do not invent unsupported claims.
Use only provided research context.
"""
    ),

    (
        "human",
        """
Write a detailed research report.

Topic:
{topic}

Research:
{research}

Structure:
- Introduction
- Key Findings
- Technical Analysis
- Conclusion
- Sources

Be detailed and professional.
"""
    ),
])

writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# CRITIC CHAIN
# =========================================================

critic_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a senior research reviewer.

Critically evaluate:
- factual consistency
- clarity
- completeness
- logical flow
- unsupported claims
"""
    ),

    (
        "human",
        """
Review the report below.

Report:
{report}

Respond in format:

Score: X/10

Strengths:
- ...
- ...

Weaknesses:
- ...
- ...

Hallucination Risks:
- ...

Final Verdict:
...
"""
    ),
])

critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)