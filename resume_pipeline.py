import json
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
INPUT_RESUMES = r'C:\Users\gayat\OneDrive\Desktop\humanAI\ADV NLP\resmatch (1)\resmatch\datascience_parsed.json'
INPUT_ESCO = r'C:\Users\gayat\OneDrive\Desktop\humanAI\ADV NLP\resmatch (1)\skills_en.csv'
OUTPUT_FILE = 'final_deliverable.json'

# --- 1. DEFINE SCHEMA & MAPPINGS ---
# Maps messy keys from Person 1's output to our standard schema
KEY_MAPPING = {
    "education":      ["Education", "Academic Background", "University", "Education & Certifications"],
    "experience":     ["Experience", "Work Experience", "Employment History", "Career Summary"],
    "skills":         ["Skills", "Core Skills", "Technical Skills", "Competencies", "Expertise", "Hard Skills", "Soft Skills"],
    "summary":        ["Summary", "Profile", "Objective", "Personal Profile", "About Me"],
    "certifications": ["Certifications", "Certificates", "Awards", "Honors"],
    "contact":        ["Other", "Contact", "Personal Info", "Contact Details"]
}

def clean_text_list(text_block):
    """Turns a big string block into a clean list of strings."""
    if not text_block: return []
    # Split by newline, strip whitespace
    lines = [line.strip() for line in re.split(r'[\n]', str(text_block)) if len(line.strip()) > 2]
    return lines

def main():
    print("🚀 STARTING PIPELINE")

    # --- 2. LOAD ESCO DATA ---
    print(f"⏳ Loading ESCO Dictionary from {INPUT_ESCO}...")
    try:
        # We load 'preferredLabel' and also 'altLabels' if available for better matching
        esco_df = pd.read_csv(INPUT_ESCO)
        
        # Create lookups
        # We match against 'preferredLabel' primarily
        esco_labels = esco_df['preferredLabel'].dropna().tolist()
        esco_uris = dict(zip(esco_df['preferredLabel'], esco_df['conceptUri']))
        
        # Load AI Model
        print("⏳ Loading AI Model (sentence-transformers)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("⏳ Encoding ESCO database (One-time setup)...")
        esco_embeddings = model.encode(esco_labels, convert_to_tensor=True)
        
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_ESCO} not found. Please download it first.")
        return

    # --- 3. LOAD & CLEAN RESUMES ---
    print(f"⏳ Loading Resumes from {INPUT_RESUMES}...")
    try:
        with open(INPUT_RESUMES, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_RESUMES} not found.")
        return

    final_resumes = []
    
    print(f"🧹 Processing {len(raw_data)} resumes...")

    for filename, content in raw_data.items():
        resume_id = filename.split('\\')[-1].replace('.jpg', '').replace('.png', '')
        
        # -- A. STANDARDIZE STRUCTURE --
        std_resume = {
            "id": resume_id,
            "source_file": filename,
            "basics": { "raw_contact": [], "summary": [] },
            "work": [],
            "education": [],
            "skills_raw": [],
            "skills_normalized": [] 
        }

        # Helper to find data using synonyms
        def extract_field(field_name):
            text = ""
            for key in KEY_MAPPING[field_name]:
                if key in content:
                    text += content[key] + "\n"
            return clean_text_list(text)

        # Fill fields
        std_resume["education"] = extract_field("education")
        std_resume["skills_raw"] = extract_field("skills")
        std_resume["basics"]["summary"] = extract_field("summary")
        std_resume["basics"]["raw_contact"] = extract_field("contact")[:5] # Keep it short
        
        # Work is special (nested list)
        work_text = extract_field("experience")
        if work_text:
            std_resume["work"] = [{"company": "Unknown", "highlights": work_text}]

        # -- B. FILTER ZOMBIES (Quality Control) --
        # Only keep if it has Skills OR Experience
        if not std_resume["skills_raw"] and not std_resume["work"]:
            # print(f"   Skipping empty resume: {resume_id}")
            continue

        # -- C. NORMALIZE SKILLS (The AI Part) --
        if std_resume["skills_raw"]:
            # Encode resume skills
            resume_skill_embeddings = model.encode(std_resume["skills_raw"], convert_to_tensor=True)
            
            # Find semantic matches
            hits = util.semantic_search(resume_skill_embeddings, esco_embeddings, top_k=1)
            
            normalized_list = []
            for i, hit in enumerate(hits):
                match = hit[0]
                if match['score'] > 0.45: # Similarity threshold
                    esco_name = esco_labels[match['corpus_id']]
                    normalized_list.append({
                        "raw_skill": std_resume["skills_raw"][i],
                        "esco_label": esco_name,
                        "esco_uri": esco_uris.get(esco_name, ""),
                        "match_score": round(match['score'], 2)
                    })
            
            std_resume["skills_normalized"] = normalized_list

        final_resumes.append(std_resume)

    # --- 4. SAVE FINAL OUTPUT ---
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_resumes, f, indent=2)

    print(f"\n✅ PIPELINE COMPLETE.")
    print(f"   - Input Resumes: {len(raw_data)}")
    print(f"   - Valid Resumes Kept: {len(final_resumes)}")
    print(f"   - Saved to: {OUTPUT_FILE}")
    print("👉 Send this file to Person 3 (Generation) and Person 4 (Verification).")

if __name__ == "__main__":
    main()