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
    temperature=0.3,
    max_tokens=4000
)


# =========================================================
# WRITER CHAIN
# =========================================================

writer_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an elite AI research analyst and technical writer.

Your responsibilities:
- Generate factual and evidence-based reports
- Maintain professional structure
- Use only provided context
- Avoid hallucinations
- Write in a concise but highly informative style

Rules:
- Never invent facts
- Never create fake citations
- Mention uncertainty if information is insufficient
- Keep technical explanations accurate
"""
    ),

    (
        "human",
        """
Generate a detailed research report.

TOPIC:
{topic}

RESEARCH CONTEXT:
{research}

Required Structure:
1. Executive Summary
2. Introduction
3. Key Findings
4. Technical Analysis
5. Current Challenges
6. Future Scope
7. Conclusion
8. Sources

Write professionally and in depth.
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
You are a senior AI research reviewer.

Your job:
- Evaluate factual consistency
- Detect hallucinations
- Analyze logical structure
- Review clarity and completeness
- Suggest improvements

Be highly critical and professional.
"""
    ),

    (
        "human",
        """
Critically review the following report.

REPORT:
{report}

Respond EXACTLY in this structure:

# Overall Score
X/10

# Strengths
- Point 1
- Point 2

# Weaknesses
- Point 1
- Point 2

# Hallucination Risks
- Risk 1
- Risk 2

# Improvement Suggestions
- Suggestion 1
- Suggestion 2

# Final Verdict
Short professional verdict.
"""
    ),
])

critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)