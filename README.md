# 🧠 ATS Resume Analyzer

An ATS-style resume–job description matcher built with **Python, Flask, NLP, SentenceTransformers, and spaCy**.  
It analyzes resumes against job descriptions, extracts skills, computes semantic similarity, and provides **match score, matched skills, and improvement suggestions**.

This project is built for learning, hackathons, and portfolio purposes.

---

## 🚀 Features

- Upload **PDF resume**
- Paste **Job Description**
- ATS-style **match score (0–100)**
- Skill extraction from JD
- Matched skills detection
- Missing skill suggestions
- Resume formatting checks (tables, images, length)
- Simple Flask UI

---

## 🖼️ Screenshots

[🖼️ Screenshots](#-screenshots)

![Home Page](image1.png)
![Results Page](image2.png)

---

## 🏗️ Tech Stack

- Python
- Flask
- spaCy
- SentenceTransformers (MiniLM)
- scikit-learn
- PyMuPDF
- HTML + CSS

---

## 📂 Project Structure

```text
Resume_Analyzer/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── uploads/
└── README.md


## ⚙️ Setup & Usage

1. **Clone the repository**

```bash
git clone https://github.com/Faraz-05/Resume_Analyzer.git
cd Resume_Analyzer

2. **Create & activate virtual environment (optional but recommended)**

```bash
python -m venv .venv

Windows:
```bash
.venv\Scripts\activate

macOS / Linux:
```bash
source .venv/bin/activate

3. **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

4. **Run the app**

```bash
python app.py

Open the URL shown in the terminal (typically http://127.0.0.1:5000) in your browser.

## 🧮 Scoring Logic (High Level)

Semantic similarity: SentenceTransformers embeddings for resume and JD compared with cosine similarity.

Keyword/skill match: spaCy-based skill candidates from the JD versus the resume, producing a matched-skills list and keyword match score.

ATS checks: Flags tables, images, and very long resumes and adjusts an ATS formatting score.

Final score: Weighted combination of semantic similarity, keyword score, and ATS score to produce a 0–100 match score.

 ## 📌 Future Improvements
Support for multiple resumes vs one JD (batch ranking and ranking table).

Export matched and missing skills as CSV for recruiters.

Add user authentication and history of past analyses.

Dockerize the app for easier deployment.

## 📜 License
This project is intended for learning and portfolio purposes.
Feel free to fork and modify it for your own experiments.
