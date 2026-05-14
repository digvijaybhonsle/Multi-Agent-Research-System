import streamlit as st
import time
import re
import asyncio

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

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS AI · Autonomous Research System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS (Improved & Cleaned)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0b0f19;
    color: #f5f7fa;
}

.stApp {
    background: 
        radial-gradient(circle at top left, rgba(255,115,0,0.12), transparent 40%),
        radial-gradient(circle at bottom right, rgba(255,80,20,0.08), transparent 40%),
        #0b0f19;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* HERO */
.hero-wrapper {
    text-align: center;
    padding: 4.5rem 0 3.5rem 0;
}

.hero-badge {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 9999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,140,50,0.25);
    color: #ff9d57;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

.hero-title {
    font-size: clamp(3.2rem, 8vw, 6.2rem);
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -0.07em;
    margin-bottom: 1.4rem;
    background: linear-gradient(90deg, #ffffff, #ff9d57);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 780px;
    margin: 0 auto;
    font-size: 1.1rem;
    line-height: 1.85;
    color: #9ba4b5;
}

/* DIVIDER */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent);
    margin: 2.5rem 0 3rem 0;
}

/* INPUT CARD */
.input-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 24px;
    padding: 2.2rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

/* PIPELINE */
.pipeline-heading {
    font-size: 1.4rem;
    font-weight: 800;
    margin-bottom: 1.8rem;
    color: white;
}

.step-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.step-card:hover {
    border-color: #ff9d57;
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(255, 157, 87, 0.15);
}

.step-number {
    font-size: 0.75rem;
    font-weight: 700;
    color: #ff9d57;
    letter-spacing: 0.15em;
    margin-bottom: 0.6rem;
}

/* RESULT BOX */
.result-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 2.2rem;
    margin-top: 2rem;
}

.result-title {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff9d57;
    margin-bottom: 1.6rem;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff8c32, #ff5a1a) !important;
    color: #0b0f19 !important;
    font-weight: 800;
    font-size: 1.02rem;
    padding: 1rem;
    border-radius: 16px;
    box-shadow: 0 8px 30px rgba(255, 115, 0, 0.3);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(255, 115, 0, 0.4);
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb {
    background: rgba(255,157,87,0.3);
    border-radius: 20px;
}

/* Mobile */
@media (max-width: 768px) {
    .hero-wrapper { padding: 3rem 0 2.5rem 0; }
    .input-card { padding: 1.8rem; }
    .result-box { padding: 1.6rem; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">Autonomous Multi-Agent Research System</div>
    <div class="hero-title">NEXUS <span style="background: linear-gradient(135deg, #ff8c32, #ff5a1a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI</span></div>
    <div class="hero-subtitle">
        Production-grade autonomous research intelligence powered by collaborative AI agents, 
        semantic memory, deep web scraping, and professional report generation.
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}

# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
col_input, col_pipeline = st.columns([5, 4])

# ===================== INPUT =====================
with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    topic = st.text_input(
        "RESEARCH TOPIC",
        placeholder="e.g. Future of Autonomous AI Agents in 2026",
        label_visibility="visible"
    )

    run_btn = st.button("⚡ Launch Autonomous Research", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== PIPELINE =====================
with col_pipeline:
    st.markdown('<div class="pipeline-heading">Pipeline Architecture</div>', unsafe_allow_html=True)
    
    steps = [
        ("01", "Search Intelligence", "Web + Academic retrieval"),
        ("02", "Deep Web Reader", "Async content extraction"),
        ("03", "Vector Memory", "Semantic storage & retrieval"),
        ("04", "Writer Chain", "Professional report generation"),
        ("05", "Critic Chain", "Research quality evaluation"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">STEP {num}</div>
            <div style="font-weight:700; font-size:1.05rem; margin-bottom:0.4rem;">{title}</div>
            <div style="color:#98a2b3; font-size:0.95rem; line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RUN PIPELINE (Logic Unchanged)
# ─────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.error("Please enter a research topic.")
        st.stop()

    state = {}

    with st.spinner("🔍 Running search intelligence..."):
        web_results = search_web.invoke(topic)
        arxiv_results = search_arxiv.invoke(topic)
        state["search"] = f"WEB RESULTS:\n{web_results}\n\nACADEMIC RESULTS:\n{arxiv_results}"

    urls = list(set(re.findall(r"https?://[^\s]+", state["search"])))[:5]

    with st.spinner("📄 Scraping and extracting content..."):
        scraped_results = asyncio.run(scrape_urls_async(urls))
        state["scraped"] = "\n\n".join([f"URL: {item['url']}\nCONTENT:\n{item['content']}" for item in scraped_results])

    with st.spinner("🧠 Building semantic memory..."):
        memory_text = f"TOPIC: {topic}\n\nSEARCH:\n{state['search']}\n\nSCRAPED:\n{state['scraped']}"
        store_document.invoke({"text": memory_text, "topic": topic})
        state["memory"] = retrieve_documents.invoke({"query": topic})

    with st.spinner("✍️ Generating research report..."):
        state["report"] = writer_chain.invoke({"topic": topic, "research": state["memory"]})

    with st.spinner("🧐 Critic agent reviewing report..."):
        state["feedback"] = critic_chain.invoke({"report": state["report"]})

    st.session_state.results = state
    st.rerun()

# ─────────────────────────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📘 Final Research Report", 
        "🧐 Critic Analysis",
        "🔍 Raw Search Results",
        "📄 Scraped Content"
    ])

    with tab1:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">FINAL RESEARCH REPORT</div>', unsafe_allow_html=True)
        st.markdown(r["report"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.download_button(
            label="⬇ Download Full Report",
            data=r["report"],
            file_name=f"nexus_ai_report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with tab2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">CRITIC ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown(r["feedback"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.text_area("Raw Search Results", r["search"][:8000], height=400)

    with tab4:
        st.text_area("Scraped Content", r["scraped"][:8000], height=400)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:5rem; color:#667085; font-size:0.8rem; letter-spacing:0.5px;">
    NEXUS AI · Autonomous Research Intelligence Platform
</div>
""", unsafe_allow_html=True)