
# Research Agent using LangGraph 📚🤖

An experimental **LLM‑based research assistant** built using **LangGraph** to structure workflows for information retrieval, extraction, and question answering. This project demonstrates how to combine LangGraph workflows with Python to create a research agent capable of extracting text, comparing methodologies, and answering domain questions.

---

## 🧠 Overview

The **Research Agent** uses a LangGraph‑oriented design to coordinate LLM interactions and stepwise reasoning for research‑oriented tasks such as:

- Extracting information from provided research texts or documents  
- Answering domain questions based on extracted content  
- Comparing different research methodologies  
- Providing organized summaries or insights

The project includes both Python modules and Jupyter notebooks to help explore workflows and evaluate results. 

---

## 📁 Repository Structure

```

Research-agent-using-langgraph/
├── papers/                       # Research papers or source documents
├── answerQuestion.ipynb          # Notebook for question answering workflow
├── compareMethodologies.ipynb    # Compare methodologies using agent logic
├── extraction.ipynb              # Notebook for information extraction
├── research_agent.py             # Core agent implementation
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore
└── LICENSE                       # MIT License

````


---

## 🚀 Features

✔ Structured agent workflow using LangGraph  
✔ Natural language based question answering from research content  
✔ Extraction and comparison workflows demonstrated in notebooks  
✔ Core agent logic in `research_agent.py`  
✔ Flexible design for experimenting with different research prompts

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| LLM Backend | OpenAI (or others supported via config) |
| Workflow | LangGraph |
| Experiments | Jupyter Notebooks |
| Dependencies | See `requirements.txt` 

---

## 📥 Installation

1. **Clone the repository**
```bash
git clone https://github.com/sisira214/Research-agent-using-langgraph.git
cd Research-agent-using-langgraph
````



2. **Create & activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate  # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```


4. **(Optional)** Add environment variables (e.g., for API keys)

```
OPENAI_API_KEY=your_openai_api_key
```

---

## ▶️ Usage

### Python module

You can run the core agent logic directly:

```bash
python research_agent.py
```

Modify or extend this script to point to your own research texts or to integrate external APIs.

---

## 📓 Example Notebooks

Notebooks demonstrate typical research tasks:

* **answerQuestion.ipynb** – Ask domain questions and receive formatted answers.
* **compareMethodologies.ipynb** – Compare multiple research methodologies using structured prompts.
* **extraction.ipynb** – Extract relevant content from text sources.

Open these in Jupyter or VS Code to explore the agent behavior interactively. 

---

## 📌 How It Works (Conceptually)

1. **Input documents** (e.g., in `papers/`) are loaded for context.
2. The agent uses LangGraph workflows to structure steps: extraction, reasoning, comparison, or QA.
3. Completed workflows return answers or insights based on provided content.
4. Notebooks show how to call and evaluate results.

---

## 🛠️ Customize & Extend

You can extend this research agent by:

* Adding more workflows (e.g., summarization, literature review synthesis)
* Integrating external search (web or academic APIs)
* Plugging in other LLM backends
* Adding evaluation metrics for quality or citation support

---

## 🤝 Contributing

Contributions welcome! You can help by:

* Adding new research workflows
* Improving extraction capabilities
* Adding structured evaluation and benchmarking
* Documenting best practices

Feel free to open issues or submit pull requests.

---

## 📜 License

This project is licensed under the **MIT License**. 
```

[1]: https://github.com/sisira214/Research-agent-using-langgraph "GitHub - sisira214/Research-agent-using-langgraph"
