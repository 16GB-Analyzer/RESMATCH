import json
import re

# --- CONFIGURATION ---
INPUT_FILE = r'C:\Users\gayat\OneDrive\Desktop\humanAI\ADV NLP\resmatch (1)\resmatch\datascience_parsed.json' # The messy file from Person 1
OUTPUT_FILE = 'ground_truth_cleaned.json' # The clean file for you

# 1. Define the "Synonym Dictionary"
# This maps the random column names Person 1 found to your Standard Keys.
KEY_MAPPING = {
    # Target Key      : [List of possible messy keys in the raw file]
    "education":      ["Education", "Academic Background", "University", "Education & Certifications"],
    "experience":     ["Experience", "Work Experience", "Employment History", "Career Summary"],
    "skills":         ["Skills", "Core Skills", "Technical Skills", "Competencies", "Expertise"],
    "summary":        ["Summary", "Profile", "Objective", "Personal Profile", "About Me"],
    "certifications": ["Certifications", "Certificates", "Awards", "Honors"],
    "contact":        ["Other", "Contact", "Personal Info", "Contact Details"]
}

def clean_text_list(text_block):
    """Turns a big string into a clean list of strings."""
    if not text_block: return []
    # Split by newline, strip whitespace, remove empty/short lines
    return [line.strip() for line in re.split(r'[\n]', str(text_block)) if len(line.strip()) > 2]

def normalize_schema():
    try:
        with open(INPUT_FILE, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    cleaned_resumes = []
    
    print(f"🧹 Scrubbing {len(raw_data)} resumes...")

    for filename, content in raw_data.items():
        resume_id = filename.split('\\')[-1].replace('.jpg', '').replace('.png', '')
        
        # --- BUILD THE STANDARD OBJECT ---
        # We start with a blank canvas for this resume
        std_resume = {
            "id": resume_id,
            "source_file": filename,
            "basics": {},
            "work": [],
            "education": [],
            "skills_raw": [],
            "certifications": [],
            "skills_normalized": [] # Empty for now, for your next step
        }

        # --- FILL FIELDS USING MAPPING ---
        
        # 1. Experience (Handle synonyms)
        # We look through our mapping. If we find ANY matching key in the raw data, we take it.
        raw_exp_text = ""
        for key in KEY_MAPPING["experience"]:
            if key in content:
                raw_exp_text += content[key] + "\n"
        
        if raw_exp_text:
            std_resume["work"] = [{
                "company": "Unknown", # Person 1 didn't extract this specifically
                "highlights": clean_text_list(raw_exp_text)
            }]

        # 2. Education
        raw_edu_text = ""
        for key in KEY_MAPPING["education"]:
            if key in content:
                raw_edu_text += content[key] + "\n"
        std_resume["education"] = clean_text_list(raw_edu_text)

        # 3. Skills
        raw_skills_text = ""
        for key in KEY_MAPPING["skills"]:
            if key in content:
                raw_skills_text += content[key] + "\n"
        std_resume["skills_raw"] = clean_text_list(raw_skills_text)

        # 4. Certifications
        raw_cert_text = ""
        for key in KEY_MAPPING["certifications"]:
            if key in content:
                raw_cert_text += content[key] + "\n"
        std_resume["certifications"] = clean_text_list(raw_cert_text)

        # 5. Basics / Contact
        raw_contact_text = ""
        for key in KEY_MAPPING["contact"]:
            if key in content:
                raw_contact_text += content[key] + "\n"
        
        # Simple heuristic: First few lines are contact info
        contact_lines = clean_text_list(raw_contact_text)
        std_resume["basics"]["raw_contact"] = contact_lines[:5] 
        
        # Also grab summary if it exists
        raw_summary_text = ""
        for key in KEY_MAPPING["summary"]:
            if key in content:
                raw_summary_text += content[key] + "\n"
        std_resume["basics"]["summary"] = clean_text_list(raw_summary_text)

        cleaned_resumes.append(std_resume)

    # Save the standardized file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(cleaned_resumes, f, indent=2)

    print(f"✅ Success! Standardized data saved to '{OUTPUT_FILE}'.")
    print(f"   - Input Count: {len(raw_data)}")
    print(f"   - Output Count: {len(cleaned_resumes)}")
    print(f"   - Now every resume has 'education', 'work', and 'skills_raw' keys.")

if __name__ == "__main__":
    normalize_schema()