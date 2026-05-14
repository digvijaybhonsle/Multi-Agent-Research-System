# NEXUS AI — Autonomous Multi-Agent Research Intelligence System

![NEXUS AI](https://img.shields.io/badge/NEXUS%20AI-Production%20Ready-FF5A1A?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C5E)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-000000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An enterprise-grade autonomous AI research platform** that conducts deep, multi-source research and generates professional reports with built-in quality evaluation — powered by multi-agent systems and semantic memory.

---

## ✨ Project Overview

**NEXUS AI** is a production-ready autonomous research intelligence system that mimics the workflow of a professional research team. It autonomously gathers information from the web and academic sources, deeply understands content through scraping and semantic indexing, generates high-quality structured reports, and self-evaluates its own output.

Built with cutting-edge AI engineering practices, NEXUS demonstrates advanced capabilities in **multi-agent orchestration**, **retrieval-augmented generation (RAG)**, **asynchronous data pipelines**, and **self-reflective AI systems**.

### Why This Project Matters

In an era of information overload, NEXUS AI represents the future of knowledge work — transforming vague research topics into comprehensive, cited, high-quality reports in minutes, with transparency and quality control built in.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[User Input] --> B[Search Intelligence Layer]
    B --> C[Deep Web Reader]
    C --> D[Semantic Memory Layer]
    D --> E[Writer Chain]
    E --> F[Critic Chain]
    F --> G[Final Report + Evaluation]
    
    subgraph "Intelligence Layer"
    B
    C
    end
    
    subgraph "Memory & Knowledge"
    D
    end
    
    subgraph "Agent Layer"
    E
    F
    end

### Core Pipeline Workflow

1. **Search Intelligence Layer**  
   - Tavily Web Search + Arxiv API  
   - Retrieves latest web results and academic papers

2. **Deep Web Reader Layer**  
   - URL extraction and deduplication  
   - Asynchronous scraping using `aiohttp` + `Playwright`  
   - Handles both static and JavaScript-rendered pages

3. **Semantic Memory Layer**  
   - Document chunking and embedding  
   - Storage in ChromaDB vector database  
   - Contextual retrieval using similarity search

4. **Writer Chain**  
   - LangChain-based LLM chain  
   - Generates professional, structured research report

5. **Critic Chain**  
   - Independent evaluation agent  
   - Detects hallucinations, logical gaps, and missing information  
   - Provides scoring and improvement suggestions

---

## 🚀 Key Features

- **Fully Autonomous Research** — Zero manual intervention required
- **Multi-Agent Architecture** — Specialized Writer + Critic agents
- **Hybrid Intelligence** — Web + Academic sources
- **Dynamic Web Scraping** — Async + headless browser support
- **Long-term Semantic Memory** — Persistent vector database
- **Self-Improving Output** — Built-in critic and quality gate
- **Production UI** — Beautiful, responsive Streamlit interface
- **Professional Reports** — Structured with sources and citations
- **RAG-Powered** — Retrieval Augmented Generation for accuracy

---

## 🛠️ Tech Stack

| Layer                    | Technology                          | Purpose |
|-------------------------|-------------------------------------|--------|
| **Frontend**            | Streamlit                           | Production-grade UI |
| **Orchestration**       | LangChain + Pydantic                | Agent chains & validation |
| **LLM**                 | Mistral AI                          | Reasoning & generation |
| **Search**              | Tavily + Arxiv API                  | Intelligence gathering |
| **Scraping**            | aiohttp + Playwright + BeautifulSoup| Deep web reading |
| **Vector Database**     | ChromaDB                            | Semantic memory |
| **Embeddings**          | Sentence Transformers               | High-quality vectors |
| **Async Runtime**       | asyncio + aiohttp                   | Performance |
| **UI/UX**              | Custom CSS + Responsive Design      | Enterprise feel |

---

## 📁 Project Structure

```bash
nexus-ai/
├── agents.PY                 # LangChain agent chaiN
├── tools.py                 # Core tools and utilities
├── pipeline.py                # Main file
├── app.py        # Main UI
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AkarshVyas/Multi-agent-research-system.git
cd nexus-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt 
```

### 4. Environment Setup

Copy the example environment file:

```bash
cp .env.example .env
```

### 5. API Keys Required

```env
# .env file
TAVILY_API_KEY=your_tavily_key_here
MISTRAL_API_KEY=your_mistral_key_here
# Optional: GROQ_API_KEY, OPENAI_API_KEY etc.
```

---

## ▶️ Running Locally

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`

---


## 📸 Screenshots

![Hero-section](image-1.png)
![Report](image.png)


---

## 🧠 AI Architecture Deep Dive

### Multi-Agent Systems
NEXUS uses a **multi-agent** design where specialized agents (Writer & Critic) collaborate. This significantly improves output quality compared to single LLM calls.

### Semantic Memory & Vector Database
- Uses **ChromaDB** — a lightweight, high-performance vector store
- **Sentence Transformers** embeddings for semantic understanding
- Enables **long-term memory** across research sessions

### Retrieval Augmented Generation (RAG)
Instead of relying solely on LLM parametric knowledge, NEXUS retrieves relevant, up-to-date context from its vector database before generation — dramatically reducing hallucinations.

### Async Scraping Architecture
Parallel asynchronous scraping using `asyncio` + `aiohttp` + `Playwright` ensures high speed and resilience when processing multiple sources.

---

## ⚡ Performance Optimizations

- Asynchronous web scraping
- Efficient document chunking
- Smart URL deduplication
- Vector similarity search with top-k retrieval
- Streaming UI updates

---

## 🌟 Why This Project is Advanced

- Demonstrates **production AI engineering** practices
- Implements true **autonomous agentic workflows**
- Combines **web intelligence + academic research**
- Features **self-reflection** through critic agent
- Production-ready UI/UX with enterprise aesthetics
- Scalable architecture ready for cloud deployment

### Resume-Worthy Highlights
- Built a complete **multi-agent autonomous system**
- Implemented **RAG** with vector databases
- Designed **asynchronous scraping pipelines**
- Created **self-evaluating AI** using critic chains
- Production deployment-ready Streamlit application

---

## 📈 Real-World Applications

- Market research automation
- Academic literature reviews
- Competitive intelligence
- Technical due diligence
- Scientific trend analysis
- Investment research

---

## 🛣️ Future Roadmap

- [ ] Multi-modal support (PDFs, images)
- [ ] Agent memory persistence across sessions
- [ ] WebSocket streaming responses
- [ ] User feedback loop for continuous improvement
- [ ] Multi-LLM routing (Mistral + Groq + Claude)
- [ ] Team collaboration features
- [ ] API endpoint for integration

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💼 Author

**Digvijay**  
*AI Engineer | Building autonomous intelligence systems*

---

<div align="center">

**Made with ❤️ for the future of AI research**

**NEXUS AI — Where Intelligence Meets Autonomy**

</div>
