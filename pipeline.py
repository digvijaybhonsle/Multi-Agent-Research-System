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
    # STEP 1 — SEARCH AGENT
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 1 — SEARCH AGENT IS WORKING...")
    print("=" * 60)

    web_results = search_web.invoke(topic)

    arxiv_results = search_arxiv.invoke(topic)

    combined_search = f"""

    WEB RESULTS:
    {web_results}

    ACADEMIC RESULTS:
    {arxiv_results}
    """

    state["search_results"] = combined_search

    print("\nSEARCH RESULTS:\n")
    print(state["search_results"][:2000])


    # =====================================================
    # STEP 2 — EXTRACT URLS
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 2 — EXTRACTING URLS...")
    print("=" * 60)

    urls = re.findall(
        r"https?://[^\s]+",
        state["search_results"]
    )

    # Remove duplicates
    urls = list(set(urls))

    # Limit URLs for stability
    urls = urls[:5]

    state["urls"] = urls

    print("\nEXTRACTED URLS:\n")

    for url in urls:
        print(url)


    # =====================================================
    # STEP 3 — SCRAPE URLS ASYNCHRONOUSLY
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 3 — SCRAPING URLS...")
    print("=" * 60)

    scraped_results = asyncio.run(
        scrape_urls_async(urls)
    )

    formatted_scraped_content = []

    for item in scraped_results:

        formatted_scraped_content.append(
            f"URL: {item['url']}\n\n"
            f"CONTENT:\n{item['content']}\n"
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
    print("STEP 4 — STORING RESEARCH IN CHROMADB...")
    print("=" * 60)

    combined_memory = f"""

    TOPIC:
    {topic}

    SEARCH RESULTS:
    {state['search_results']}

    SCRAPED CONTENT:
    {state['scraped_content']}
    """

    memory_result = store_document.invoke({
        "text": combined_memory,
        "topic": topic
    })

    state["memory_status"] = memory_result

    print("\nMEMORY STATUS:\n")
    print(memory_result)


    # =====================================================
    # STEP 5 — RETRIEVE RELEVANT CONTEXT
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 5 — RETRIEVING RELEVANT CONTEXT...")
    print("=" * 60)

    retrieved_context = retrieve_documents.invoke({
        "query": topic
    })

    state["retrieved_context"] = retrieved_context

    print("\nRETRIEVED CONTEXT:\n")
    print(retrieved_context[:3000])


    # =====================================================
    # STEP 6 — WRITER CHAIN
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 6 — WRITER CHAIN IS GENERATING REPORT...")
    print("=" * 60)

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": state["retrieved_context"]
    })

    print("\nFINAL REPORT:\n")
    print(state["report"][:4000])


    # =====================================================
    # STEP 7 — CRITIC CHAIN
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 7 — CRITIC CHAIN IS REVIEWING REPORT...")
    print("=" * 60)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\nCRITIC FEEDBACK:\n")
    print(state["feedback"])


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