import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = [
    "python", "java", "c++", "sql", "machine learning",
    "deep learning", "tensorflow", "pytorch", "pandas",
    "numpy", "scikit-learn", "opencv", "nlp", "git",
    "data science", "html", "css", "javascript", "react"
]

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS if skill in text]

def screen_resumes(job_description, resume_folder):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    jd_skills = extract_skills(job_description)
    jd_embedding = model.encode([job_description])

    results = []

    for filename in os.listdir(resume_folder):
        if filename.endswith(".txt"):
            path = os.path.join(resume_folder, filename)

            with open(path, "r", encoding="utf-8") as file:
                resume_text = file.read()

            resume_skills = extract_skills(resume_text)
            resume_embedding = model.encode([resume_text])

            similarity = cosine_similarity(
                jd_embedding,
                resume_embedding
            )[0][0]

            matched = list(set(jd_skills) & set(resume_skills))
            missing = list(set(jd_skills) - set(resume_skills))

            skill_score = (
                len(matched) / len(jd_skills)
                if jd_skills else 0
            )

            final_score = similarity * 70 + skill_score * 30

            results.append({
                "Candidate": filename.replace(".txt", ""),
                "Score": round(final_score, 2),
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing)
            })

    results.sort(key=lambda x: x["Score"], reverse=True)

    for rank, result in enumerate(results, 1):
        result["Rank"] = rank

    return pd.DataFrame(results)


if __name__ == "__main__":

    job_description = """
    Junior AI/ML Engineer

    Required skills:
    Python, Machine Learning, Deep Learning, TensorFlow,
    Pandas, NumPy, Scikit-learn, NLP, OpenCV and SQL.

    Computer Science or related engineering degree.
    AI/ML projects or internship experience preferred.
    """

    results = screen_resumes(
        job_description,
        "resumes"
    )

    print("\nRESUME SCREENING RESULTS\n")
    print(results.to_string(index=False))

    results.to_csv("ranked_candidates.csv", index=False)

    print("\nResults saved to ranked_candidates.csv")
