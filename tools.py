from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import uuid
import streamlit as st

# =========================================================
# LAZY LOADING + CACHING (Critical for Render)
# =========================================================

@st.cache_resource(show_spinner="Loading embedding model...", ttl=3600)
def get_embedding_model():
    """Lazy load SentenceTransformer - only loads when first called"""
    from sentence_transformers import SentenceTransformer
    print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu",                    # Force CPU on Render Free Tier
        model_kwargs={"torch_dtype": "float32"}
    )
    print("✅ Embedding model loaded successfully")
    return model


@st.cache_resource(show_spinner="Initializing vector database...", ttl=3600)
def get_chroma_client():
    """Persistent ChromaDB client optimized for Render"""
    import chromadb
    import tempfile
    print("🔄 Initializing ChromaDB...")
    
    # Use /tmp for Render (ephemeral but sufficient)
    persist_directory = "/tmp/chroma_db"
    
    client = chromadb.PersistentClient(path=persist_directory)
    print("✅ ChromaDB initialized")
    return client


@st.cache_resource
def get_collection():
    """Get or create collection"""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="research_reports",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# =========================================================
# TEXT SPLITTER (Global is fine)
# =========================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# =========================================================
# TAVILY CLIENT
# =========================================================
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# =========================================================
# TOOL 1 — WEB SEARCH
# =========================================================
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web using Tavily"""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=4
        )

        out = []
        for result in response.get("results", []):
            out.append(
                f"Title: {result.get('title', 'No Title')}\n"
                f"URL: {result.get('url', 'No URL')}\n"
                f"Content: {result.get('content', '')[:400]}\n"
            )
        return "\n---\n".join(out) if out else "No results found."

    except Exception as e:
        return f"Search Error: {str(e)}"


# =========================================================
# ASYNC SCRAPER
# =========================================================
import aiohttp
from bs4 import BeautifulSoup

async def fetch_page(session, url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NexusAI/1.0)"}
        
        async with session.get(
            url, 
            timeout=aiohttp.ClientTimeout(total=12),
            headers=headers
        ) as response:
            if response.status != 200:
                return {"url": url, "content": f"HTTP Error: {response.status}"}
            
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()
            
            text = soup.get_text(separator=" ", strip=True)
            return {
                "url": url,
                "content": text[:3500]
            }
    except Exception as e:
        return {"url": url, "content": f"Scraping Error: {str(e)}"}


async def scrape_urls_async(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        cleaned = []
        for result in results:
            if isinstance(result, Exception):
                cleaned.append({"url": "Unknown", "content": str(result)})
            else:
                cleaned.append(result)
        return cleaned


# =========================================================
# TOOL 2 — STORE DOCUMENT (Optimized)
# =========================================================
@tool
def store_document(text: str, topic: str = "general") -> str:
    try:
        chunks = splitter.split_text(text)
        if not chunks:
            return "No content to store."

        model = get_embedding_model()
        embeddings = model.encode(chunks).tolist()

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"topic": topic, "chunk_id": i} for i in range(len(chunks))]

        get_collection().add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return f"✅ Stored {len(chunks)} chunks successfully."

    except Exception as e:
        return f"Storage Error: {str(e)}"


# =========================================================
# TOOL 3 — RETRIEVE DOCUMENTS
# =========================================================
@tool
def retrieve_documents(query: str) -> str:
    try:
        model = get_embedding_model()
        query_embedding = model.encode([query]).tolist()

        results = get_collection().query(
            query_embeddings=query_embedding,
            n_results=6,
            include=["documents", "metadatas", "distances"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return "No relevant documents found in memory."

        formatted = []
        for doc, meta in zip(docs, metas):
            formatted.append(f"TOPIC: {meta.get('topic', 'general')}\nCONTENT:\n{doc}")

        return "\n\n" + "="*60 + "\n\n".join(formatted)

    except Exception as e:
        return f"Retrieval Error: {str(e)}"


# =========================================================
# TOOL 4 — ARXIV SEARCH
# =========================================================
import arxiv

@tool
def search_arxiv(query: str) -> str:
    try:
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in search.results():
            results.append(
                f"Title: {result.title}\n"
                f"Authors: {', '.join([a.name for a in result.authors])}\n"
                f"Published: {result.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {result.summary[:600]}\n"
                f"PDF: {result.pdf_url}\n"
            )
        return "\n" + "="*50 + "\n".join(results) if results else "No papers found."
        
    except Exception as e:
        return f"Arxiv Error: {str(e)}"