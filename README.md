
# Resume Screening AI Agent

## Overview

This project is an AI-powered Resume Screening Agent that ranks candidates
against a given Job Description.

The agent:
- Reads candidate resumes
- Extracts relevant skills
- Uses NLP semantic similarity
- Calculates a relevance score
- Identifies matched and missing skills
- Ranks candidates
- Produces CSV and JSON results

## Technology

- Python
- Sentence Transformers
- Scikit-learn
- Pandas
- Google Colab

## How It Works

Job Description
        ↓
Resume Text
        ↓
Skill Extraction
        ↓
Sentence Transformer Embeddings
        ↓
Cosine Similarity
        ↓
Skill Matching
        ↓
Final Score
        ↓
Candidate Ranking

## Scoring Method

The final score combines:

70% - Semantic similarity between the Job Description and resume

30% - Required skill matching

Final Score = (Semantic Similarity × 70) + (Skill Match × 30)

## Output

The agent generates:

- ranked_candidates.csv
- ranked_candidates.json

Each candidate contains:
- Rank
- Candidate name
- Relevance score
- Matched skills
- Missing skills

## Limitations

This prototype uses sample text resumes. A production version could
support PDF/DOCX files, experience extraction, education matching,
better skill normalization, and a web interface.

## Future Improvements

- PDF and DOCX resume parsing
- LLM-based candidate explanations
- Experience and education scoring
- Web interface
- Database storage
- Human-review workflow
