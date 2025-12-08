import json
import os
# Updated imports for latest LangChain versions
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate # <-- FIXED IMPORT
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
# 1. Your input file from Person 2
# Using raw string (r"...") to handle Windows backslashes correctly
INPUT_FILE = r'C:\Users\gayat\OneDrive\Desktop\humanAI\ADV NLP\resmatch (1)\final_deliverable.json'

# 2. Your output file (The "Rewritten Resume")
OUTPUT_FILE = 'rewritten_resumes.json'

# 3. The Target Job Description
TARGET_JOB_DESCRIPTION = {
    "title": "Senior Data Scientist",
    "requirements": [
        "Experience building predictive models using Python and Scikit-Learn.",
        "Ability to visualize complex data using Tableau or D3.js.",
        "Strong communication skills and ability to lead cross-functional teams.",
        "Experience with cloud platforms like AWS or Google Cloud."
    ]
}

# --- SETUP AI ---
# Make sure your API key is set in your environment variables or uncomment below:
# os.environ["OPENAI_API_KEY"] = "sk-..."

# We use temperature=0.0 to strictly enforce "Factuality" (The 'HAT' in RAG-HAT)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
embeddings = OpenAIEmbeddings()

# --- THE "HAT" PROMPT ---
hat_prompt = ChatPromptTemplate.from_template("""
You are an expert Resume Writer. Your task is to rewrite a candidate's experience to match a specific Job Requirement.

*** STRICT GROUNDING RULES ***
1. SOURCE OF TRUTH: You must ONLY use the facts provided in the "Evidence" section below.
2. NO HALLUCINATIONS: Do not invent skills, numbers, or company names that are not in the Evidence.
3. INTEGRITY: If the Evidence does not support the Job Requirement, reply exactly: "NO_MATCH".
4. TONE: Professional, active voice, results-oriented.

TARGET REQUIREMENT: {requirement}
EVIDENCE FROM RESUME: {evidence}

REWRITTEN BULLET POINT:
""")

chain = hat_prompt | llm | StrOutputParser()

def generate_rewrites():
    print("🚀 Starting RAG-HAT Pipeline...")
    
    # 1. Load Data
    try:
        with open(INPUT_FILE, 'r') as f:
            resumes = json.load(f)
        print(f"✅ Loaded data from: {INPUT_FILE}")
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        print("   Please check the path or run the normalization script first!")
        return

    rewritten_resumes = []

    # 2. Process Each Resume
    for resume in resumes:
        print(f"\n📄 Processing Candidate ID: {resume.get('id', 'Unknown')}")
        
        # --- A. BUILD SCOPED VECTOR STORE ---
        candidate_facts = []
        
        # Add Work History
        for job in resume.get('work', []):
            # Handle cases where highlights might be strings or lists
            highlights = job.get('highlights', [])
            if isinstance(highlights, list):
                candidate_facts.extend(highlights)
            elif isinstance(highlights, str):
                candidate_facts.append(highlights)
        
        # Add Skills
        for skill in resume.get('skills_normalized', []):
            candidate_facts.append(f"Skill: {skill.get('esco_label', 'Unknown Skill')}")
            
        # If resume is empty, skip
        if not candidate_facts:
            print("   ⚠️ Skipping (No data found)")
            continue

        # Create temporary vector store for THIS candidate only
        vectorstore = Chroma.from_texts(texts=candidate_facts, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # --- B. GENERATE CONTENT ---
        new_highlights = []
        
        for req in TARGET_JOB_DESCRIPTION["requirements"]:
            # 1. Retrieve Evidence
            docs = retriever.invoke(req)
            combined_evidence = "\n".join([d.page_content for d in docs])
            
            # 2. Generate Rewrite
            result = chain.invoke({
                "requirement": req,
                "evidence": combined_evidence
            })
            
            # 3. Filter "NO_MATCH" results
            if "NO_MATCH" not in result:
                print(f"   ✅ Matched Req: '{req[:30]}...'")
                # print(f"      -> New Bullet: {result}")
                new_highlights.append(result)
            else:
                # print(f"   ❌ No evidence for: '{req[:30]}...'")
                pass
        
        # --- C. UPDATE JSON STRUCTURE ---
        resume['rewritten_profile'] = {
            "target_role": TARGET_JOB_DESCRIPTION['title'],
            "generated_summary": new_highlights
        }
        
        rewritten_resumes.append(resume)
        
        # Clean up vector store
        vectorstore.delete_collection()

    # 3. Save Final Output
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(rewritten_resumes, f, indent=2)

    print(f"\n🎉 SUCCESS! Rewritten resumes saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    generate_rewrites()