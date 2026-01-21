
# ==========================================================
# LANGGRAPH PDF EXTRACTION AGENT (USING ResearchAgentState)
# Extraction-only: sections, concepts, methods, findings,
# citations, tables/figures/statistics
# ==========================================================
import os
import operator
from typing import TypedDict, Sequence, List, Dict
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END

from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



# ---------------------------
# ENV SETUP
# ---------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



# ==========================================================
# STATE (AS REQUESTED)
# ==========================================================

class ResearchAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    papers: list[dict]                 # {"name": str, "text": str}
    query: str
    extracted_info: dict               # All extracted structured data
    search_results: list[dict]         # Not used here (kept for compatibility)
    comparison_matrix: dict | None     # Not used
    research_gaps: list[str]            # Not used
    iteration_count: int
    reflection: str
    folder_path: str

    chunks: list
    vectorstore: FAISS | None
    retrieved_docs: list
    extracted_info: str
    comparison_matrix: str
    conflicts: List[str]
    research_gaps: List[str]
    reranked_docs: list
    



# ==========================================================
# CONFIG
# ==========================================================

PDF_FOLDER = "C:/Users/sashi/OneDrive/Documents/langgraphProjects/researchPaper/papers"


# ==========================================================
# TOOLS / NODES
# ==========================================================

# ==========================================================
# COMMON PDF HANDLING NODES
# ==========================================================
def load_pdfs_common(state: ResearchAgentState):
    papers = []
    for file in os.listdir(state["folder_path"]):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(state["folder_path"], file))
            docs = loader.load()
            # Combine pages into one text per paper
            text = "\n".join(d.page_content for d in docs)
            papers.append({"name": file, "text": text})
    print(f"✅ Loaded {len(papers)} PDFs")
    return {"papers": papers}


def split_documents_common(state: ResearchAgentState):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = []
    for paper in state["papers"]:
        paper_chunks = splitter.split_text(paper["text"])
        paper["chunks"] = paper_chunks
        chunks.extend(paper_chunks)
    print(f"✅ Created {len(chunks)} text chunks")
    return {"chunks": chunks, "papers": state["papers"]}


def create_vectorstore_common(state: ResearchAgentState):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(state["chunks"], embeddings)
    print("✅ Vector store created")
    return {"vectorstore": vectorstore}




# ------------------ SECTION EXTRACTION ---------------------
import re
SECTION_HEADERS = {
    "abstract": ["abstract"],
    "introduction": ["introduction"],
    "methods": ["methodology", "methods", "approach"],
    "results": ["results", "experiments"],
    "conclusion": ["conclusion", "future work"]
}

def extract_sections_node(state: ResearchAgentState):
    for paper in state["papers"]:
        text = paper["text"].lower()
        sections = {}

        for sec, headers in SECTION_HEADERS.items():
            for h in headers:
                pattern = rf"{h}\n(.+?)(?=\n[A-Z][a-z]+|\Z)"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    sections[sec] = match.group(1).strip()
                    break

        paper["sections"] = sections

    return {"papers": state["papers"]}


# ------------------ KEY CONCEPTS --------------------------

def extract_key_concepts_node(state: ResearchAgentState):
    for paper in state["papers"]:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=500
        )
        X = vectorizer.fit_transform(paper["chunks"])
        scores = X.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()

        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        paper["concepts"] = [t for t, _ in ranked[:20]]

    return {"papers": state["papers"]}


# ------------------ METHODOLOGIES -------------------------

AI_ML_KEYWORDS = [
    "machine learning", "deep learning", "neural network",
    "cnn", "rnn", "transformer", "bert", "gpt",
    "random forest", "svm", "xgboost",
    "gradient descent", "backpropagation",
    "accuracy", "precision", "recall", "f1 score"
]

def extract_methodologies_node(state: ResearchAgentState):
    for paper in state["papers"]:
        found = set()
        for chunk in paper["chunks"]:
            c = chunk.lower()
            for kw in AI_ML_KEYWORDS:
                if kw in c:
                    found.add(kw)

        paper["methodologies"] = sorted(found)

    return {"papers": state["papers"]}


# ------------------ FINDINGS ------------------------------

def extract_findings_node(state: ResearchAgentState):
    for paper in state["papers"]:
        findings = []
        for chunk in paper["chunks"]:
            for sent in re.split(r"[.!?]", chunk):
                if any(w in sent.lower() for w in [
                    "outperform", "improve", "increase",
                    "significant", "accuracy"
                ]):
                    findings.append(sent.strip())

        paper["findings"] = findings[:25]

    return {"papers": state["papers"]}


# ------------------ CITATIONS -----------------------------

def extract_citations_node(state: ResearchAgentState):
    patterns = [
        r"\([A-Za-z]+ et al\., \d{4}\)",
        r"\[\d+\]"
    ]

    for paper in state["papers"]:
        citations = set()
        for p in patterns:
            citations.update(re.findall(p, paper["text"]))
        paper["citations"] = list(citations)

    return {"papers": state["papers"]}


# ------------------ TABLES / FIGURES / STATS --------------

def extract_tables_figures_node(state: ResearchAgentState):
    for paper in state["papers"]:
        extracted = []
        for chunk in paper["chunks"]:
            if any(k in chunk.lower() for k in [
                "table", "figure", "fig.", "%",
                "mean", "std", "accuracy"
            ]):
                extracted.append(chunk)

        paper["tables_figures_stats"] = extracted[:20]

    return {"papers": state["papers"]}


# ------------------ FINAL STRUCTURING ---------------------

def build_extracted_info_node(state: ResearchAgentState):
    extracted_info = {}

    for paper in state["papers"]:
        extracted_info[paper["name"]] = {
            "sections": paper.get("sections", {}),
            "concepts": paper.get("concepts", []),
            "methodologies": paper.get("methodologies", []),
            "findings": paper.get("findings", []),
            "citations": paper.get("citations", []),
            "tables_figures_stats": paper.get("tables_figures_stats", [])
        }

    return {"extracted_info": extracted_info}


# ==========================================================
# COMPARE METHODOLOGIES AND IDENTIFY GAPS
# ==========================================================



# ---------------------------
# NODE 4: RETRIEVE SECTIONS
# ---------------------------
def retrieve_relevant_sections(state: ResearchAgentState):
    docs = state["vectorstore"].similarity_search(
        "methodology experimental design dataset results findings",
        k=12
    )
    print(f"✅ Retrieved {len(docs)} relevant sections")
    return {"retrieved_docs": docs}


# ---------------------------
# NODE 5: COMPARE METHODOLOGIES
# ---------------------------
def compare_methodologies(state: ResearchAgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context = "\n\n".join(
        doc.page_content[:1500] for doc in state["retrieved_docs"]
    )

    prompt = f"""
Analyze the methodologies used in the following research papers.

Extract for each paper:
- Methodology
- Dataset
- Model / Algorithm
- Evaluation Metrics
- Key Strengths
- Limitations

Return structured text.

Papers:
{context}
"""

    response = llm.invoke(prompt)
    print("✅ Methodology comparison completed")
    return {"extracted_info": response.content}


# ---------------------------
# NODE 6: BUILD COMPARISON MATRIX
# ---------------------------
def build_comparison_matrix(state: ResearchAgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
Using the extracted information below, build a comparison matrix.

Rows = Papers  
Columns = Method, Dataset, Strengths, Limitations

Extracted Info:
{state["extracted_info"]}
"""

    response = llm.invoke(prompt)
    print("✅ Comparison matrix built")
    return {"comparison_matrix": response.content}


# ---------------------------
# NODE 7: IDENTIFY CONFLICTS
# ---------------------------
def identify_conflicts(state: ResearchAgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
Identify conflicting or contradicting findings across the papers.

Comparison Matrix:
{state["comparison_matrix"]}

Return bullet points.
"""

    response = llm.invoke(prompt)
    print("✅ Conflicts identified")
    return {"conflicts": response.content.split("\n")}


# ---------------------------
# NODE 8: FIND RESEARCH GAPS
# ---------------------------
def find_research_gaps(state: ResearchAgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
Based on:
- Comparison matrix
- Conflicting results

Identify:
- Research gaps
- Missing datasets
- Underexplored methods
- Future research directions

Return bullet points.
"""

    response = llm.invoke(prompt)
    print("✅ Research gaps identified")
    return {"research_gaps": response.content.split("\n")}



# ==========================================================
# ANSWERING QUESTIONS
# ==========================================================



def retrieve_documents(state: ResearchAgentState):
    docs = state["vectorstore"].similarity_search(
        state["query"], k=10
    )
    return {"search_results": docs}


def rerank_documents(state: ResearchAgentState):
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    pairs = [(state["query"], doc.page_content) for doc in state["search_results"]]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(state["search_results"], scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, _ in reranked[:5]]
    return {"reranked_docs": top_docs}


def generate_answer(state: ResearchAgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context = "\n\n".join([doc.page_content for doc in state["reranked_docs"]])

    prompt = f"""
Answer the following research question using the provided context.

Question:
{state["query"]}

Context:
{context}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [AIMessage(content=response.content)],
        "reflection": "Answer generated using reranked evidence."
    }


from langchain_core.messages import AIMessage

def respond(state: ResearchAgentState):
    """
    Returns a user-friendly message based on the selected task.
    """
    content = ""

    if state.get("task") == "extract":
        # Summarize extracted info
        parts = []
        for paper, info in state.get("extracted_info", {}).items():
            parts.append(f"📄 {paper}")
            for k, v in info.items():
                if isinstance(v, dict):
                    for sec, txt in v.items():
                        parts.append(f"[{sec.upper()}] {txt[:800]}...")
                else:
                    parts.append(f"{k.upper()}: {', '.join(str(x) for x in v[:10])}")
        content = "\n\n".join(parts)

    elif state.get("task") == "compare":
        content = state.get("comparison_matrix", "")
        if state.get("conflicts"):
            content += "\n\n⚠️ Conflicts:\n" + "\n".join(f"- {c}" for c in state["conflicts"])

    elif state.get("task") == "gaps":
        content = "\n".join(state.get("research_gaps", []))

    elif state.get("task") == "question":
        # Pull the answer from the generate_answer node
        if state.get("messages"):
            content = state["messages"][-1].content
        else:
            content = "No answer generated."

    return {"messages": [AIMessage(content=content)]}


def router_node_llm(state: ResearchAgentState):
    """
    Decides which branch to follow: extract_info, question_answering, comparison_analysis
    """
    assert state.get("task") is not None
    mapping = {
        "extract": "Extract_key_concepts",
        "question": "answering_llm",
        "compare": "Compare_methodologies",
    }
    return mapping[state["task"]]



def parse_paper_structure(state: ResearchAgentState):
    """
    Entry point: pre-parses PDFs so all branches can access sections + chunks + vectorstore.
    """
    # 1️⃣ Load PDFs
    state.update(load_pdfs_common(state))
    
    # 2️⃣ Split PDFs into chunks
    state.update(split_documents_common(state))
    
    # 3️⃣ Create vectorstore
    state.update(create_vectorstore_common(state))
    
    
    return state




# ==========================================================
# LANGGRAPH DEFINITION
# ==========================================================

graph = StateGraph(ResearchAgentState)

# Router
graph.add_node("router_node_llm", router_node_llm)
# Common PDF nodes
graph.add_node("load_pdfs", load_pdfs_common)
graph.add_node("split_documents", split_documents_common)
graph.add_node("create_vectorstore", create_vectorstore_common)

# Extraction nodes
graph.add_node("extract_sections", extract_sections_node)
graph.add_node("extract_concepts", extract_key_concepts_node)
graph.add_node("extract_methods", extract_methodologies_node)
graph.add_node("extract_findings", extract_findings_node)
graph.add_node("extract_citations", extract_citations_node)
graph.add_node("extract_tables", extract_tables_figures_node)
graph.add_node("build_extracted_info", build_extracted_info_node)

# Comparison & gaps nodes
graph.add_node("retrieve_relevant_sections", retrieve_relevant_sections)
graph.add_node("compare_methodologies", compare_methodologies)
graph.add_node("build_comparison_matrix", build_comparison_matrix)
graph.add_node("identify_conflicts", identify_conflicts)
graph.add_node("find_research_gaps", find_research_gaps)

# Question answering nodes
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("rerank_documents", rerank_documents)
graph.add_node("generate_answer", generate_answer)


graph.set_entry_point("load_pdfs_common")
graph.add_node("parse_paper_structure", parse_paper_structure)
graph.add_edge("parse_paper_structure", "router_node_llm")


def route_selector(state: ResearchAgentState):
    return state["task"]

graph.add_conditional_edges(
    "router_node_llm",
    route_selector,
    {
        "extract": "extract_sections_node",
        "question": "retrieve_documents",
        "compare": "retrieve_relevant_sections",
    },
)

# ==========================================================
# CONNECTING EDGES BASED ON TASK
# ==========================================================

# Extraction workflow
graph.add_edge("extract_sections", "extract_concepts")
graph.add_edge("extract_concepts", "extract_methods")
graph.add_edge("extract_methods", "extract_findings")
graph.add_edge("extract_findings", "extract_citations")
graph.add_edge("extract_citations", "extract_tables")
graph.add_edge("extract_tables", "build_extracted_info")
graph.add_edge("build_extracted_info", END)

# Compare & gaps workflow

graph.add_edge("retrieve_relevant_sections", "compare_methodologies")
graph.add_edge("compare_methodologies", "build_comparison_matrix")
graph.add_edge("build_comparison_matrix", "identify_conflicts")
graph.add_edge("identify_conflicts", "find_research_gaps")
graph.add_edge("find_research_gaps", END)

# Question answering workflow
graph.add_edge("retrieve_documents", "rerank_documents")
graph.add_edge("rerank_documents", "generate_answer")
graph.add_edge("generate_answer", END)

app = graph.compile()


# ==========================================================
# RUNNER
# ==========================================================

def main():
    print("\n===== RESEARCH PAPER AGENT =====\n")

    folder_path = input("Enter PDF folder path: ").strip()

    if not os.path.isdir(folder_path):
        print("❌ Invalid folder path")
        return

    print("\nChoose task:")
    print("1 - Extract information from papers")
    print("2 - Compare papers")
    print("3 - Find research gaps")
    print("4 - Ask a question")

    task_map = {
        "1": "extract",
        "2": "compare",
        "3": "gaps",
        "4": "question"
    }

    task_choice = input("Enter choice (1/2/3/4): ").strip()
    task = task_map.get(task_choice)

    if task is None:
        print("❌ Invalid task selection")
        return

    # Select PDFs
    pdfs = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdfs:
        print("❌ No PDFs found in folder")
        return

    print("\nAvailable PDFs:")
    for p in pdfs:
        print(" -", p)

    pdf_choice = input(
        "\nEnter PDF names separated by comma OR type 'all': "
    ).strip()

    if pdf_choice.lower() != "all":
        selected = [p.strip() for p in pdf_choice.split(",")]
        missing = [p for p in selected if p not in pdfs]
        if missing:
            print("❌ These PDFs were not found:", missing)
            return
        pdfs = selected

    query = ""
    if task == "question":
        query = input("\nEnter your research question: ").strip()

    # Build state
    state = {
        "folder_path": folder_path,
        "query": query,
        "task": task,
        "papers": [],
        "chunks": [],
        "vectorstore": None,
        "retrieved_docs": [],
        "extracted_info": {},
        "comparison_matrix": "",
        "conflicts": [],
        "research_gaps": [],
        "messages": [],
        "reflection": ""
    }

    # Run graph
    result = app.invoke(state)

    # ---------------- OUTPUT ----------------
    print("\n================ RESULT ================\n")

    if task == "extract":
        for paper, info in result["extracted_info"].items():
            print(f"\n📄 {paper}")
            for k, v in info.items():
                print(f"\n--- {k.upper()} ---")
                if isinstance(v, dict):
                    for sec, txt in v.items():
                        print(f"\n[{sec.upper()}]\n{txt[:800]}...")
                else:
                    for item in v[:10]:
                        print("-", str(item)[:300])

    elif task == "compare":
        print("📊 COMPARISON MATRIX:\n")
        print(result["comparison_matrix"])

        if result["conflicts"]:
            print("\n⚠️ CONFLICTS:")
            for c in result["conflicts"]:
                print("-", c)

    elif task == "gaps":
        print("🔍 RESEARCH GAPS:\n")
        for g in result["research_gaps"]:
            print("-", g)

    elif task == "question":
        print("💡 ANSWER:\n")
        print(result["messages"][-1].content)

    print("\n=======================================\n")
