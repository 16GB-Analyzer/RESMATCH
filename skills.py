import pandas as pd

# 1. We create the data manually so you don't need to download anything
print("⏳ Generating 'skills_en.csv' file...")

mini_esco_data = {
    'preferredLabel': [
        # Tech Skills
        'Python (computer programming)', 'R (computer programming)', 'SQL', 'machine learning', 
        'data analysis', 'data mining', 'statistical analysis', 'Java (computer programming)',
        'C++', 'Tableau', 'data visualization', 'big data', 'Hadoop', 'Spark', 'NoSQL',
        'Microsoft Excel', 'deep learning', 'neural networks', 'natural language processing',
        'scikit-learn', 'TensorFlow', 'PyTorch', 'computer vision', 'predictive modeling',
        'business intelligence', 'SAS (software)', 'MATLAB', 'MongoDB', 'Git', 
        'software engineering', 'agile methodology', 'mathematics', 'statistics',
        'linear algebra', 'calculus', 'probability', 'cloud computing', 'AWS', 'Azure',
        # Soft Skills
        'communication', 'teamwork', 'problem solving', 'project management', 'leadership',
        'writing reports', 'presentation skills', 'critical thinking', 'time management',
        'adaptability', 'creativity'
    ],
    # We create fake IDs that look like real ESCO IDs
    'conceptUri': [f'http://data.europa.eu/esco/skill/fake-id-{i}' for i in range(50)]
}

# 2. Convert to a Table (DataFrame)
df = pd.DataFrame(mini_esco_data)

# 3. Save it as the CSV file your other script is looking for
csv_filename = 'skills_en.csv'
df.to_csv(csv_filename, index=False)

print(f"✅ Success! Created '{csv_filename}' with {len(df)} skills.")
print("👉 Now go back and run your 'finalize_dataset()' script. It will work now!")