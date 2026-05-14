from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import aiohttp
import chromadb
import arxiv
import uuid

from tavily import TavilyClient
from pypdf import PdfReader
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# CHROMADB SETUP
# =========================================================

chroma_client = chromadb.PersistentClient(
    path="/tmp/chroma_db"
)

embedding_model = None

def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return embedding_model


collection = chroma_client.get_or_create_collection(
    name="research_reports"
)

# =========================================================
# TEXT SPLITTER
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# =========================================================
# TAVILY CLIENT
# =========================================================

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# =========================================================
# TOOL 1 — WEB SEARCH
# =========================================================

@tool
def search_web(query: str) -> str:
    """
    Search the web and return
    titles, URLs and snippets.
    """

    try:

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5
        )

        out = []

        for result in response.get("results", []):

            title = result.get(
                "title",
                "No Title"
            )

            url = result.get(
                "url",
                "No URL"
            )

            content = result.get(
                "content",
                "No Content"
            )

            out.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content[:300]}\n"
            )

        return "\n----\n".join(out)

    except Exception as e:

        return f"Search Error: {str(e)}"


# =========================================================
# ASYNC SCRAPER FUNCTIONS
# =========================================================

async def fetch_page(session, url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        async with session.get(
            url,
            timeout=10,
            headers=headers
        ) as response:

            html = await response.text()

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            for tag in soup([
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside"
            ]):
                tag.decompose()

            text = soup.get_text(
                separator=" ",
                strip=True
            )

            return {
                "url": url,
                "content": text[:5000]
            }

    except Exception as e:

        return {
            "url": url,
            "content": f"Async Scraping Error: {str(e)}"
        }


async def scrape_urls_async(urls):

    async with aiohttp.ClientSession() as session:

        tasks = [
            fetch_page(session, url)
            for url in urls
        ]
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True
        )

        return results


# =========================================================
# TOOL 2 — PDF READER
# =========================================================

@tool
def read_pdf(pdf_path: str) -> str:
    """
    Read and extract text from PDF.
    """

    try:

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

        return text[:8000]

    except Exception as e:

        return f"PDF Reading Error: {str(e)}"


# =========================================================
# TOOL 3 — STORE DOCUMENTS
# =========================================================

@tool
def store_document(
    text: str,
    topic: str = "general"
) -> str:
    """
    Store research documents in ChromaDB.
    """

    try:

        # Split text into chunks
        chunks = splitter.split_text(text)

        # Generate embeddings
        model = get_embedding_model()

        embeddings = model.encode(
            chunks
        ).tolist()

        # Unique IDs
        ids = [
            str(uuid.uuid4())
            for _ in range(len(chunks))
        ]

        # Metadata
        metadatas = [
            {
                "topic": topic,
                "chunk": i
            }
            for i in range(len(chunks))
        ]

        # Store in ChromaDB
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return (
            f"Successfully stored "
            f"{len(chunks)} chunks in ChromaDB."
        )

    except Exception as e:

        return f"ChromaDB Storage Error: {str(e)}"


# =========================================================
# TOOL 4 — RETRIEVE DOCUMENTS
# =========================================================

@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieve semantically relevant documents.
    """

    try:

        model = get_embedding_model()

        query_embedding = model.encode(
            [query]
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        formatted = []

        for doc, meta in zip(docs, metas):

            formatted.append(
                f"TOPIC: {meta['topic']}\n"
                f"CONTENT:\n{doc}"
            )

        return "\n\n====================\n\n".join(
            formatted
        )

    except Exception as e:

        return f"Retrieval Error: {str(e)}"


# =========================================================
# TOOL 5 — ARXIV SEARCH
# =========================================================

@tool
def search_arxiv(query: str) -> str:
    """
    Search academic papers from Arxiv.
    """

    try:

        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )

        out = []

        for result in search.results():

            out.append(
                f"Title: {result.title}\n"
                f"Authors: "
                f"{', '.join([a.name for a in result.authors])}\n"
                f"Published: {result.published}\n"
                f"Summary: {result.summary[:500]}\n"
                f"PDF: {result.pdf_url}\n"
            )

        return "\n====================\n".join(out)

    except Exception as e:

        return f"Arxiv Search Error: {str(e)}"