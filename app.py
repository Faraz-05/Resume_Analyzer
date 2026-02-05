import os
import re
import random
import fitz  # PyMuPDF
import numpy as np

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Lazy-loaded global models
_st_model = None   # SentenceTransformer
_nlp = None        # spaCy nlp


# ========== MODEL LOADERS (LAZY) ==========

def get_sentence_model():
    """
    Lazily load the SentenceTransformer model.
    """
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def get_nlp():
    """
    Lazily load the spaCy English model.
    """
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ========== UTILITY FUNCTIONS ==========

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def extract_text_from_html(html: str) -> str:
    # Remove HTML tags, keep plain text
    return re.sub(r"<[^<]+?>", "", html).replace("\n", " ").strip()


# ---- JD skill extraction (generic for any role) ----

SKILL_STOPWORDS = {
    "data", "analysis", "analyst", "analytics", "business", "role", "roles",
    "position", "job", "team", "teams", "year", "years", "experience",
    "experiences", "ability", "abilities", "working", "work", "strong",
    "excellent", "good", "great", "communication", "stakeholder",
    "stakeholders", "thinking", "detail", "curiosity", "requirement",
    "requirements", "responsibility", "responsibilities", "problem",
    "solving", "environment", "language", "languages", "field", "degree",
    "education",
}


def extract_jd_skill_candidates(text: str, top_n: int = 30) -> list[str]:
    """
    Extract 'skill-like' candidates from any job description, using spaCy but
    filtering out generic words. Returns a ranked list of unique skill terms.
    """
    nlp = get_nlp()
    doc = nlp(text)

    candidates: dict[str, int] = {}

    # 1) Named entities that look like skills/tools (ORG, PRODUCT, etc.)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "FAC", "GPE"]:
            key = ent.text.strip().lower()
            if key and key not in SKILL_STOPWORDS:
                candidates[key] = candidates.get(key, 0) + 3

    # 2) Noun chunks / technical phrases
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if len(phrase) < 3:
            continue
        if any(w in SKILL_STOPWORDS for w in phrase.split()):
            continue
        candidates[phrase] = candidates.get(phrase, 0) + 2

    # 3) Single nouns / proper nouns (tool names, frameworks, etc.)
    for token in doc:
        if token.pos in ["NOUN", "PROPN"] and not token.is_stop:
            lemma = token.lemma_.lower()
            if lemma in SKILL_STOPWORDS:
                continue
            if len(lemma) < 3:
                continue
            candidates[lemma] = candidates.get(lemma, 0) + 1

    # Sort and select top_n
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    skills = [k for k, _ in sorted_items[:top_n]]
    return skills


def get_synonyms_spacy(word: str, job_keywords):
    nlp = get_nlp()
    word_doc = nlp(word)

    for kw in job_keywords:
        kw_doc = nlp(kw)
        if word_doc.similarity(kw_doc) > 0.8:
            return True
    return False


def check_ats_friendly(text: str):
    issues = []
    if re.search(r"table|column|image|graphic", text, re.IGNORECASE):
        issues.append("Avoid tables, columns, images, or graphics for ATS-friendliness.")
    if len(text) > 20000:
        issues.append("Resume is too long. Keep it concise.")
    return issues


def compute_similarity(resume_text: str, job_desc: str) -> float:
    model = get_sentence_model()
    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(float(similarity[0][0]) * 100, 2)


# ========== SCORING & SUGGESTIONS ==========

def compute_ats_score(resume_text: str, job_desc: str):
    # 1. Semantic similarity (0–100)
    sim_score = compute_similarity(resume_text, job_desc)

    # 2. Keyword match (with synonyms) using skill candidates
    job_keywords = extract_jd_skill_candidates(job_desc)
    resume_keywords = set(extract_jd_skill_candidates(resume_text, top_n=50))
    matched_keywords = [
        kw for kw in job_keywords
        if kw in resume_keywords or get_synonyms_spacy(kw, resume_keywords)
    ]
    keyword_score = int(100 * len(matched_keywords) / max(1, len(job_keywords)))

    # 3. ATS issues
    ats_issues = check_ats_friendly(resume_text)
    ats_score = 100 if not ats_issues else max(60, 100 - 20 * len(ats_issues))

    # Weighted total score (no section component)
    total_score = int(
        0.7 * sim_score +    # content similarity matters most
        0.2 * keyword_score +
        0.1 * ats_score
    )

    return total_score, job_keywords, matched_keywords, ats_issues


def generate_suggestions(
    resume_text: str,
    job_desc: str,
    job_keywords,
    matched_keywords,
    ats_issues,
    score: int,
):
    suggestions = []

    # Missing keyword-style skills
    missing_keywords = [kw for kw in job_keywords if kw not in matched_keywords]
    # Limit to top N missing keywords to avoid huge lists
    missing_keywords = missing_keywords[:10]

    resume_lower = resume_text.lower()

    GENERIC_TEMPLATES = [
        "If you have experience with '{kw}', add a clear bullet point that demonstrates it.",
        "Highlight any projects or roles where you used '{kw}' and explain the impact.",
        "Show how you applied '{kw}' in practice, mentioning tools, data, or outcomes.",
        "Add a concise bullet describing your hands-on experience with '{kw}'.",
        "Include a result-focused bullet about using '{kw}' (what you did and what changed).",
        "Mention '{kw}' under your skills and support it with a concrete example in your experience section.",
        "Describe a challenging task where you solved a problem using '{kw}'.",
        "If relevant, add a project where '{kw}' was a key part of the tech stack.",
        "Use action verbs to describe how you used '{kw}' and what you delivered.",
        "Add a measurable achievement that shows your proficiency with '{kw}'.",
    ]

    for kw in missing_keywords:
        if len(kw) < 3:
            continue
        kw_lower = kw.lower()
        # If already in resume text (maybe different form), skip
        if kw_lower in resume_lower:
            continue

        # Hand-crafted patterns for some common families
        if "sql" in kw_lower:
            msg = f"Highlight your experience with {kw} (queries, joins, reporting), if applicable."
        elif any(term in kw_lower for term in ["python", "java", "c++", "javascript", "django", "react", "node", "spring"]):
            msg = f"Add details about your work with {kw}, such as projects, responsibilities, or achievements."
        elif any(term in kw_lower for term in ["power bi", "tableau", "looker", "excel", "sheets"]):
            msg = f"Mention dashboards or reports you built using {kw} and the business impact they had."
        elif any(term in kw_lower for term in ["warehouse", "warehousing", "database", "schema"]):
            msg = f"Describe any experience with {kw}, including design, optimization, or maintenance tasks."
        elif any(term in kw_lower for term in ["cloud", "aws", "azure", "gcp", "kubernetes", "docker"]):
            msg = f"Include your hands-on experience with {kw}, highlighting relevant projects or setups."
        else:
            # Use one of several generic templates to avoid repetition
            template = random.choice(GENERIC_TEMPLATES)
            msg = template.format(kw=kw)

        suggestions.append({
            "category": "Keyword",
            "message": msg,
        })

    # ATS issues
    for issue in ats_issues:
        suggestions.append({
            "category": "ATS",
            "message": issue
        })

    # General score improvement
    if score < 80:
        suggestions.append({
            "category": "Score",
            "message": "Follow the above suggestions to move your score closer to 80% or higher."
        })

    return suggestions


# ========== ROUTES ==========

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "resume" not in request.files:
            return "No file uploaded", 400

        file = request.files["resume"]
        job_desc = request.form.get("job_desc", "")

        if file.filename == "" or job_desc.strip() == "":
            return "Invalid input", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)
        score, job_keywords, matched_keywords, ats_issues = compute_ats_score(
            resume_text, job_desc
        )
        suggestions = generate_suggestions(
            resume_text, job_desc,
            job_keywords,
            matched_keywords, ats_issues,
            score,
        )

        return render_template(
            "index.html",
            score=score,
            suggestions=suggestions,
            matched_keywords=matched_keywords,
            resume_text=resume_text,
        )

    return render_template("index.html", score=None, suggestions=None, matched_keywords=None, resume_text=None)


@app.route("/editor", methods=["GET", "POST"])
def editor():
    if request.method == "POST":
        resume_html = request.form.get("resume_html", "")
        job_desc = request.form.get("job_desc", "")
        resume_text = extract_text_from_html(resume_html)

        score, job_keywords, matched_keywords, ats_issues = compute_ats_score(
            resume_text, job_desc
        )
        suggestions = generate_suggestions(
            resume_text, job_desc,
            job_keywords,
            matched_keywords, ats_issues,
            score,
        )

        return jsonify({
            "score": score,
            "suggestions": suggestions,
            "matched_keywords": matched_keywords,
            "message": "Resume scored and suggestions generated.",
        })

    # GET: open editor with optional prefilled resume_text
    resume_text = request.args.get("resume_text", None)
    return render_template("editor.html", resume_text=resume_text)


if __name__ == "__main__":
    # No debug auto-reload to avoid heavy re-imports
    app.run(debug=False)
