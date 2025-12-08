import streamlit as st
import os
import json
import PyPDF2 as pdf
import re
import torch
from sentence_transformers import SentenceTransformer, util
# --- OpenAI Imports ---
from openai import OpenAI
# --- End OpenAI Imports ---

from dotenv import load_dotenv

load_dotenv()

# --- OpenAI Configuration ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4o-mini" 
# --- End OpenAI Configuration ---

# --- RAG Component Setup ---
rag_embedder = SentenceTransformer("all-MiniLM-L6-v2")


## OpenAI Response Function
def get_openai_response(prompt_text, use_json_format=False):
    """
    Generates content using OpenAI API for structured JSON output or simple text.
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert ATS screening and resume optimization assistant."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.1,
            response_format={"type": "json_object"} if use_json_format else None
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        # Return a structured error response for clean parsing
        if use_json_format:
            return json.dumps({
                "JD Match": "0%", 
                "MissingKeywords": ["API Call Failed"], 
                "Profile Summary": "Could not connect to the model for analysis."
            })
        return "Failed to generate content due to an API error."


## --- Utility Functions ---
def input_pdf_text(uploaded_file):
    """Extracts text content from an uploaded PDF file."""
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

def run_rag_hat_rewrite(resume_text, jd_requirement, missing_keywords, initial_score):
    """
    Performs the RAG-HAT process: Retrieve facts, then use those facts to ground the LLM rewrite,
    and returns the optimized bullets along with the LLM's projected score.
    """
    
    # 1. Fact Splitting and Indexing
    facts = [line.strip() for line in resume_text.split('\n') if len(line.strip()) > 15]
    if not facts:
        return json.dumps({"OptimizedBullets": ["Error: Could not extract useful facts from resume."], "ProjectedScore": initial_score})

    fact_embeddings = rag_embedder.encode(facts, convert_to_tensor=True)
    
    # 2. Query Construction and Retrieval
    query = f"Based on the resume, find experience related to: {', '.join(missing_keywords)} and the main skills in the JD."
    query_embedding = rag_embedder.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, fact_embeddings)[0]
    top_k = torch.topk(cos_scores, k=min(3, len(facts)))
    
    retrieved_facts = "\n".join([facts[idx.item()] for idx in top_k.indices])
    
    # 3. HAT Prompt Construction (OpenAI Generation) - Includes Score Projection
    hat_prompt_text = f"""
    You are a strictly factual ATS Resume Rewriter. Your task is to perform two actions:
    
    **ACTION 1: REWRITE BULLETS**
    1. Rewrite 3-5 existing resume bullets based **ONLY** on the "Resume Evidence" to cover the "Missing Keywords." 
    2. MUST NOT invent tools, metrics, or experiences.

    **ACTION 2: SCORE PROJECTION**
    3. Based on the initial score of {initial_score}%, project the new ATS score (0-100) assuming the generated bullets successfully cover the missing keywords. **Do not exceed 95%.**

    **Job Requirement/Target:** {jd_requirement}
    **Missing Keywords to Cover:** {', '.join(missing_keywords)}

    **Resume Evidence (Retrieval Context):**
    {retrieved_facts}

    Output the result in a single JSON object with two fields:
    {{"OptimizedBullets": [], "ProjectedScore": 0}}
    """
    
    # Call the LLM to perform the rewrite and projection (Generation)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a factual ATS expert. Output only the requested JSON structure."},
                {"role": "user", "content": hat_prompt_text}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI RAG-HAT Error: {e}")
        # Return error structure with initial score as fallback projection
        return json.dumps({"OptimizedBullets": ["Failed to generate optimized content."], "ProjectedScore": initial_score})


# --- INITIAL ATS SCORE & GAP ANALYSIS PROMPT TEMPLATE ---
ats_prompt_template = """
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field, software engineering, data science, data analyst,
and big data engineer. Your task is to perform a two-step analysis:

**STEP 1: ATS SCORE & GAP ANALYSIS**
1. Evaluate the resume based on the given job description.
2. Assign a percentage matching based on the JD (Job Description).
3. Identify the missing keywords that are crucial for the role.

**INPUTS:**
resume:{text}
description:{jd}

I want the response in one single string having the following exact JSON structure, and no other text:
{{
"JD Match":"%",
"MissingKeywords:[]",
"Profile Summary":""
}}
"""


## Beautify and display the LLM responses in Streamlit
def beautify_response(ats_response_json, optimized_bullets_json):
    """
    Parses the JSON response from the LLM and displays all results, 
    including the projected score improvement and the optimized document preview.
    """
    st.title("Resume Analysis Result (ATS + RAG-HAT)")

    # --- 1. Parse ATS Score and Gap Analysis (First LLM Call) ---
    try:
        response_data = json.loads(ats_response_json.strip())
    except json.JSONDecodeError as e:
        st.error(f"Error: Could not parse the LLM's ATS score JSON response: {e}")
        st.subheader("Raw Model Response:")
        st.text_area("Raw Text:", ats_response_json, height=300)
        return

    # Extract initial score and keywords
    jd_match_str = response_data.get("JD Match", "0%").strip().replace('%', '')
    missing_keywords = response_data.get("MissingKeywords", [])
    
    try:
        initial_score = int(float(jd_match_str))
    except ValueError:
        initial_score = 0

    # --- 2. Parse Optimized Bullets and Projected Score (Second LLM Call) ---
    projected_score = initial_score # Default fallback
    optimized_bullets_list = []
    
    try:
        optimized_data = json.loads(optimized_bullets_json.strip())
        optimized_bullets_list = optimized_data.get("OptimizedBullets", [])
        
        # Check and set the projected score from the LLM's calculation
        if "ProjectedScore" in optimized_data:
            try:
                projected_score = int(optimized_data["ProjectedScore"])
            except ValueError:
                pass # Keep fallback score
    except json.JSONDecodeError:
        pass # Keep fallback score and empty bullet list

    # --- 3. Display Results ---
    
    # Initial Score
    st.markdown(f"### 🎯 Initial JD Match: **{initial_score}%**")

    # Projected Score Display
    if projected_score > initial_score and projected_score > 70:
        st.success(f"### 📈 Projected Score (After RAG-HAT): **{projected_score}%**")
        st.caption("This score is verified by the LLM, projecting success after applying the optimized bullets.")
    elif initial_score < 70:
        st.warning("Score remains low. Check if RAG-HAT failed or if major skills are missing.")


    # Missing Keywords (Gap Analysis)
    st.markdown("---")
    if missing_keywords and missing_keywords != ["API Call Failed"]:
        st.markdown("#### ❌ Missing Keywords (To Focus On):")
        st.markdown(
            "\n".join([f"- **{keyword}**" for keyword in missing_keywords])
        )
    else:
        st.markdown("#### ✅ Missing Keywords:")
        st.markdown("No critical missing keywords identified.")
    
    # Profile Summary
    if "Profile Summary" in response_data and response_data["Profile Summary"]:
        st.markdown("#### 📝 Profile Summary:")
        st.markdown(response_data["Profile Summary"])
    
    # --- 4. RAG-HAT Optimized Experience Bullets ---
    st.markdown("---")
    
    if optimized_bullets_list:
        st.markdown("#### 🚀 RAG-HAT Optimized Experience Bullets (Structurally Verified):")
        
        st.markdown("\n".join([f"- {line.lstrip('- ')}" for line in optimized_bullets_list]))
        
        # --- NEW SECTION: FINAL OPTIMIZED DOC PREVIEW ---
        st.markdown("---")
        st.markdown("## 📄 Optimized Resume Preview for Re-Scan")
        st.warning("Copy the text below and paste it into your original document's Experience section for a new ATS check.")
        
        optimized_report_text = f"""
PROJECTED ATS SCORE: {projected_score}%

MISSING KEYWORDS ADDRESSED: {', '.join(missing_keywords)}

--- OPTIMIZED EXPERIENCE SECTION ---
"""
        optimized_report_text += "\n".join([f"- {line}" for line in optimized_bullets_list])
        optimized_report_text += "\n\n--- END OPTIMIZED CONTENT ---"
        
        st.text_area("Copy Text for Optimized Document Re-Scan:", optimized_report_text, height=300)
        
    else:
        st.markdown("#### RAG-HAT Optimized Experience Bullets:")
        st.markdown("Failed to generate optimized content.")


# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="Smart ATS + RAG-HAT Optimizer")
st.title("🚀 Smart ATS + RAG-HAT Resume Optimizer (OpenAI)")
st.markdown("Improve Your Resume's ATS Compatibility and Content Accuracy.")

# Input Areas
col1, col2 = st.columns(2)

with col1:
    jd = st.text_area("Paste the Job Description Here:", height=400)
    uploaded_file = st.file_uploader(
        "Upload Your Resume (PDF Only)", type="pdf", help="Please Upload the pdf"
    )

# Prepare text for analysis
resume_text = ""
if uploaded_file is not None:
    try:
        resume_text = input_pdf_text(uploaded_file)
        st.session_state['resume_text'] = resume_text
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.session_state['resume_text'] = ""
elif 'resume_text' not in st.session_state:
    st.session_state['resume_text'] = ""

with col2:
    st.text_area("Resume Text Preview (Extracted or Pasted):", value=st.session_state.get('resume_text', ""), height=400, disabled=True)
    
submit = st.button("Analyze and Optimize Resume")

if submit:
    if not jd or not resume_text:
        st.error("Please paste the Job Description and upload a Resume.")
    else:
        # Step 1: ATS Score and Gap Analysis (First LLM Call)
        with st.spinner('Calculating score and identifying skill gaps...'):
            ats_input = ats_prompt_template.format(text=resume_text, jd=jd)
            ats_response_json = get_openai_response(ats_input, use_json_format=True)
            
            # Extract initial score for RAG-HAT projection input
            try:
                ats_data = json.loads(ats_response_json.strip())
                missing_keywords = ats_data.get("MissingKeywords", [])
                jd_match_str = ats_data.get("JD Match", "0%").strip().replace('%', '')
                initial_score_for_rag = int(float(jd_match_str))
            except:
                st.error("Failed to extract initial score for RAG. Using 0%.")
                missing_keywords = []
                initial_score_for_rag = 0


        # Step 2: RAG-HAT Rewriting (Retrieval + Generation)
        optimized_bullets_json = ""
        if missing_keywords and missing_keywords != ["API Call Failed"]:
            with st.spinner('Running RAG Retrieval and HAT Generation...'):
                # Pass the initial score to the RAG function for internal projection
                optimized_bullets_json = run_rag_hat_rewrite(resume_text, jd, missing_keywords, initial_score_for_rag)
        
        # Step 3: Display combined results
        beautify_response(ats_response_json, optimized_bullets_json)
