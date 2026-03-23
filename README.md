# RESMATCH 

An intelligent resume-to-job-description matcher that uses RAG (Retrieval-Augmented Generation) to analyze candidate fit and provide actionable insights.

##  Overview
RESMATCH goes beyond simple keyword matching. By leveraging LLMs and RAG workflows, it compares resumes against specific job descriptions to extract skills, identify gaps, and even suggest rewrites to better align with the role.

##  Key Features
* **RAG-Powered Analysis:** Uses a retrieval-based approach to contextualize resume data against job requirements.
* **Streamlit Interface:** A clean, interactive UI for uploading resumes and viewing analysis results.
* **Skill Extraction:** Automatically parses and categorizes candidate skills from `PDF` and `CSV` sources.
* **Automated Rewriting:** Suggests improvements to help candidates tailor their profiles for specific JD requirements.

## 🛠️ Tech Stack
* **Language:** Python
* **LLM Orchestration:** [Mention your framework, e.g., LangChain or LlamaIndex]
* **Frontend:** Streamlit
* **Data Handling:** Pandas, JSON

##  Project Structure
* `rag-hat.py`: The core logic for the RAG-based matching system.
* `final_code.py`: The Streamlit application entry point.
* `resume_pipeline.py`: The data processing pipeline for resume parsing.
* `groundtruth.py`: Evaluation scripts to ensure matching accuracy.

##  Quick Start
1. **Clone the repo:**
   ```bash
   git-clone [https://github.com/16GB-Analyzer/RESMATCH.git](https://github.com/16GB-Analyzer/RESMATCH.git)
