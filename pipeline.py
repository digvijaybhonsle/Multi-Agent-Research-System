from agents import (
    writer_chain,
    critic_chain
)

from tools import (
    search_web,
    search_arxiv,
    scrape_urls_async,
    store_document,
    retrieve_documents
)

import re
import asyncio
import time


# =========================================================
# MAIN RESEARCH PIPELINE
# =========================================================

def run_research_pipeline(topic: str) -> dict:
    """
    Full autonomous research pipeline.
    Returns complete state dictionary.
    """
    state = {}
    start_time = time.time()

    print("\n" + "="*80)
    print(f"🚀 STARTING RESEARCH: {topic}")
    print("="*80)

    # =====================================================
    # STEP 1 — SEARCH INTELLIGENCE
    # =====================================================
    step_start = time.time()
    print("\n🔍 STEP 1 — Search Intelligence...")

    try:
        web_results = search_web.invoke(topic)
    except Exception as e:
        web_results = f"Web Search Error: {str(e)}"

    try:
        arxiv_results = search_arxiv.invoke(topic)
    except Exception as e:
        arxiv_results = f"Arxiv Error: {str(e)}"

    state["search_results"] = f"""
WEB RESULTS:
{web_results}

ACADEMIC RESULTS:
{arxiv_results}
"""

    print(f"   ✅ Search completed in {time.time() - step_start:.2f}s")

    # =====================================================
    # STEP 2 — URL EXTRACTION
    # =====================================================
    step_start = time.time()
    print("\n🔗 STEP 2 — Extracting URLs...")

    urls = re.findall(r"https?://[^\s]+", state["search_results"])
    urls = list(set(urls))[:5]          # Limit for stability & cost

    state["urls"] = urls
    print(f"   Found {len(urls)} unique URLs")
    print(f"   ✅ URL extraction completed in {time.time() - step_start:.2f}s")

    # =====================================================
    # STEP 3 — DEEP WEB SCRAPING
    # =====================================================
    step_start = time.time()
    print("\n📄 STEP 3 — Scraping Web Content...")

    try:
        scraped_results = asyncio.run(scrape_urls_async(urls))
    except Exception as e:
        print(f"   Scraping failed: {e}")
        scraped_results = [{"url": "N/A", "content": f"Scraping failed: {str(e)}"}]

    # Format scraped content
    formatted_scraped = [
        f"URL: {item.get('url', 'N/A')}\nCONTENT:\n{item.get('content', '')[:4000]}"
        for item in scraped_results
    ]

    state["scraped_content"] = "\n\n" + "="*60 + "\n\n".join(formatted_scraped)
    print(f"   ✅ Scraping completed in {time.time() - step_start:.2f}s")

    # =====================================================
    # STEP 4 — STORE IN SEMANTIC MEMORY
    # =====================================================
    step_start = time.time()
    print("\n🧠 STEP 4 — Storing in Vector Memory...")

    memory_text = f"""
TOPIC: {topic}

SEARCH RESULTS:
{state['search_results']}

SCRAPED CONTENT:
{state['scraped_content']}
"""

    try:
        memory_status = store_document.invoke({
            "text": memory_text,
            "topic": topic
        })
    except Exception as e:
        memory_status = f"Memory Storage Error: {str(e)}"

    state["memory_status"] = memory_status
    print(f"   ✅ Memory storage completed in {time.time() - step_start:.2f}s")

    # =====================================================
    # STEP 5 — RETRIEVE CONTEXT
    # =====================================================
    step_start = time.time()
    print("\n🔎 STEP 5 — Retrieving Relevant Context...")

    try:
        retrieved = retrieve_documents.invoke({"query": topic})
    except Exception as e:
        retrieved = f"Retrieval Error: {str(e)}"

    state["retrieved_context"] = retrieved
    print(f"   ✅ Context retrieval completed in {time.time() - step_start:.2f}s")

    # =====================================================
    # STEP 6 — WRITER CHAIN
    # =====================================================
    step_start = time.time()
    print("\n✍️  STEP 6 — Generating Research Report...")

    try:
        report = writer_chain.invoke({
            "topic": topic,
            "research": retrieved
        })
    except Exception as e:
        report = f"Writer Chain Error: {str(e)}"

    state["report"] = report
    print(f"   ✅ Report generation completed in {time.time() - step_start:.2f}s")

    # =====================================================
    # STEP 7 — CRITIC CHAIN
    # =====================================================
    step_start = time.time()
    print("\n🧐 STEP 7 — Critic Review...")

    try:
        feedback = critic_chain.invoke({"report": report})
    except Exception as e:
        feedback = f"Critic Chain Error: {str(e)}"

    state["feedback"] = feedback
    print(f"   ✅ Critic review completed in {time.time() - step_start:.2f}s")

    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"🎉 RESEARCH PIPELINE COMPLETED in {total_time:.2f} seconds")
    print("="*80)

    return state


# =========================================================
# CLI ENTRY POINT
# =========================================================

if __name__ == "__main__":
    print("NEXUS AI — Autonomous Research System")
    topic = input("\nEnter research topic: ").strip()

    if not topic:
        print("❌ Topic cannot be empty.")
    else:
        final_state = run_research_pipeline(topic)
        print("\n✅ Final Report generated successfully!")