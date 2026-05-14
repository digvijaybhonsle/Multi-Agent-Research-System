from agents import (
    writer_chain,
    critic_chain
)

from tools import (
    search_web,
    search_arxiv,
    store_document,
    retrieve_documents,
    scrape_urls_async
)

import re
import asyncio


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_research_pipeline(topic: str) -> dict:

    state = {}

    # =====================================================
    # STEP 1 — SEARCH
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 1 — SEARCHING WEB + ARXIV...")
    print("=" * 60)

    try:

        web_results = search_web.invoke(topic)

    except Exception as e:

        web_results = f"Web Search Error: {str(e)}"

    try:

        arxiv_results = search_arxiv.invoke(topic)

    except Exception as e:

        arxiv_results = f"Arxiv Error: {str(e)}"

    combined_search = f"""
WEB RESULTS:
{web_results}

ACADEMIC RESULTS:
{arxiv_results}
"""

    state["search_results"] = combined_search

    print("\nSEARCH RESULTS:\n")
    print(combined_search[:2000])



    # =====================================================
    # STEP 2 — EXTRACT URLS
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 2 — EXTRACTING URLS...")
    print("=" * 60)

    urls = re.findall(
        r"https?://[^\s]+",
        combined_search
    )

    # Remove duplicates
    urls = list(set(urls))

    # Limit URLs for stability
    urls = urls[:3]

    state["urls"] = urls

    print("\nEXTRACTED URLS:\n")

    for url in urls:
        print(url)



    # =====================================================
    # STEP 3 — SCRAPE URLS
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 3 — SCRAPING URLS...")
    print("=" * 60)

    try:

        scraped_results = asyncio.run(
            scrape_urls_async(urls)
        )

    except Exception as e:

        scraped_results = [
            {
                "url": "N/A",
                "content": f"Scraping Failed: {str(e)}"
            }
        ]

    formatted_scraped_content = []

    for item in scraped_results:

        formatted_scraped_content.append(
            f"""
URL:
{item['url']}

CONTENT:
{item['content']}
"""
        )

    state["scraped_content"] = (
        "\n\n====================\n\n"
        .join(formatted_scraped_content)
    )

    print("\nSCRAPED CONTENT:\n")
    print(state["scraped_content"][:3000])



    # =====================================================
    # STEP 4 — STORE IN CHROMADB
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 4 — STORING IN VECTOR MEMORY...")
    print("=" * 60)

    combined_memory = f"""
TOPIC:
{topic}

SEARCH RESULTS:
{state['search_results']}

SCRAPED CONTENT:
{state['scraped_content']}
"""

    try:

        memory_result = store_document.invoke({
            "text": combined_memory,
            "topic": topic
        })

    except Exception as e:

        memory_result = (
            f"Memory Storage Error: {str(e)}"
        )

    state["memory_status"] = memory_result

    print("\nMEMORY STATUS:\n")
    print(memory_result)



    # =====================================================
    # STEP 5 — RETRIEVE CONTEXT
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 5 — RETRIEVING CONTEXT...")
    print("=" * 60)

    try:

        retrieved_context = retrieve_documents.invoke({
            "query": topic
        })

    except Exception as e:

        retrieved_context = (
            f"Retrieval Error: {str(e)}"
        )

    state["retrieved_context"] = retrieved_context

    print("\nRETRIEVED CONTEXT:\n")
    print(retrieved_context[:3000])



    # =====================================================
    # STEP 6 — WRITER CHAIN
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 6 — GENERATING REPORT...")
    print("=" * 60)

    try:

        report = writer_chain.invoke({
            "topic": topic,
            "research": retrieved_context
        })

    except Exception as e:

        report = f"Writer Chain Error: {str(e)}"

    state["report"] = report

    print("\nFINAL REPORT:\n")
    print(report[:4000])



    # =====================================================
    # STEP 7 — CRITIC CHAIN
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 7 — REVIEWING REPORT...")
    print("=" * 60)

    try:

        feedback = critic_chain.invoke({
            "report": report
        })

    except Exception as e:

        feedback = f"Critic Chain Error: {str(e)}"

    state["feedback"] = feedback

    print("\nCRITIC FEEDBACK:\n")
    print(feedback)



    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    return state



# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter a research topic: "
    )

    final_state = run_research_pipeline(
        topic
    )