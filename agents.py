from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# LLM CONFIGURATION
# =========================================================

llm_writer = ChatMistralAI(
    model="mistral-small-2506",      # or "mistral-large-latest" if budget allows
    temperature=0.2,                   # Lower for factual writing
    max_tokens=4096,
    top_p=0.95,
)

llm_critic = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.1,                   # Even lower for critical evaluation
    max_tokens=2048,
    top_p=0.9,
)


# =========================================================
# WRITER CHAIN
# =========================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an elite technical research analyst and professional writer.
You produce high-quality, well-structured, evidence-based research reports.

Core Rules:
- ONLY use the provided research context. Do not hallucinate facts.
- Be precise, objective, and analytical.
- Use clear professional language.
- Cite sources implicitly where possible.
- Maintain logical flow and depth.
- If information is missing or unclear, explicitly state it."""
    ),
    (
        "human",
        """Write a comprehensive research report on the following topic.

TOPIC: {topic}

RESEARCH CONTEXT:
{research}

Structure the report exactly as follows:

1. Executive Summary
2. Introduction
3. Key Findings
4. Technical Analysis
5. Current Challenges & Limitations
6. Future Outlook
7. Conclusion
8. Sources & References

Make it detailed, insightful, and professionally written."""
    ),
])

writer_chain = (
    writer_prompt
    | llm_writer
    | StrOutputParser()
)


# =========================================================
# CRITIC CHAIN
# =========================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a rigorous, highly critical senior research reviewer and quality assurance expert.

Your goal is to catch inaccuracies, hallucinations, logical gaps, and structural weaknesses.
Be strict but fair and constructive."""
    ),
    (
        "human",
        """Critically evaluate the following research report.

REPORT:
{report}

Provide your review in this **exact** format:

# Overall Score
X/10

# Strengths
- Bullet point 1
- Bullet point 2

# Weaknesses
- Bullet point 1
- Bullet point 2

# Hallucination & Factual Risks
- Risk 1
- Risk 2

# Logical & Structural Issues
- Issue 1
- Issue 2

# Improvement Recommendations
- Recommendation 1
- Recommendation 2

# Final Verdict
One sentence professional verdict (e.g., "Strong report with minor gaps" or "Requires significant revision")."""
    ),
])

critic_chain = (
    critic_prompt
    | llm_critic
    | StrOutputParser()
)