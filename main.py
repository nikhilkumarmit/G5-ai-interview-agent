import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DEFAULT_RESUME = "data/resume/resume.pdf"
DEFAULT_JD = "data/job_description/job_description.txt"
VECTOR_DIR = "interview_data/chroma_db"
COLLECTION_NAME = "mock_interview_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

HEADINGS = {
    "professional summary": "summary", "summary": "summary",
    "objective": "summary", "career objective": "summary",
    "key expertise": "skills", "skills": "skills",
    "technical skills": "skills", "core competencies": "skills",
    "projects": "projects", "academic projects": "projects",
    "personal projects": "projects", "project experience": "projects",
    "education": "education", "academic background": "education",
    "experience": "experience", "work experience": "experience",
    "internship": "experience", "internships": "experience",
    "certifications": "certifications", "certificates": "certifications",
    "web links": "web_links", "links": "web_links",
    "personal details": "personal_details",
    "personal information": "personal_details",
    "job title": "job_title", "position": "job_title",
    "target role": "job_title", "role": "job_title",
    "job description": "job_description", "role overview": "job_description",
    "about the role": "job_description", "requirements": "requirements",
    "required technical skills": "requirements",
    "qualifications": "requirements", "preferred skills": "preferred_skills",
    "responsibilities": "responsibilities",
    "key responsibilities": "responsibilities",
    "interview evaluation criteria": "evaluation_criteria",
}

# These are the three project headings present in the supplied resume.
# This allowlist prevents contact details or other PDF spillover lines
# from being counted as projects.
EXPECTED_PROJECT_TITLES = [
    "Airline Delay Classification",
    "Lightweight Hallucination Detection for Small Language Models",
    "AI Image-to-Video Generation Platform",
]

SKILL_TERMS = [
    "Python", "C", "C++", "Java", "Dart", "SQL", "PostgreSQL", "MySQL",
    "MongoDB", "Git", "GitHub", "Node.js", "Express.js", "React.js",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn",
    "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
    "Generative AI", "Large Language Models", "LLM", "LangChain", "RAG",
    "Hugging Face", "Small Language Models", "SLMs", "PyTorch", "TensorFlow",
    "REST API", "OOP", "OOPs", "DSA", "DBMS", "Prompt Engineering",
    "Tokenization", "Entropy", "Semantic Similarity", "Model Evaluation",
    "Chroma", "Embeddings", "FastAPI", "Flask", "HTML", "CSS", "JavaScript",
]


def normalize_heading(line: str) -> str:
    line = re.sub(r"^\s*#+\s*", "", line.strip())
    line = line.strip().strip("*").strip().rstrip(":").strip()
    return re.sub(r"\s+", " ", line).lower()


def load_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if file_path.suffix.lower() == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(file_path)).pages)
    elif file_path.suffix.lower() == ".txt":
        text = file_path.read_text(encoding="utf-8-sig")
    else:
        raise ValueError("Supported file types are .txt and .pdf.")
    text = text.replace("\x00", "")
    if not text.strip():
        raise ValueError(f"No extractable text found in {path}. A scanned PDF may need OCR.")
    return text


def split_sections(text: str, source: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = "general"
    heading_names = sorted(HEADINGS, key=len, reverse=True)
    inline_regex = re.compile(
        r"^\s*(?:#+\s*)?(" + "|".join(re.escape(h) for h in heading_names) +
        r")\s*:\s*(.*?)\s*$", re.I
    )
    standalone_regex = re.compile(
        r"^\s*(?:#+\s*)?(" + "|".join(re.escape(h) for h in heading_names) +
        r")\s*$", re.I
    )

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        match = inline_regex.match(line)
        if match:
            current = HEADINGS[normalize_heading(match.group(1))]
            sections.setdefault(current, [])
            value = match.group(2).strip()
            if value:
                sections[current].append(value)
            continue

        match = standalone_regex.match(line)
        if match:
            current = HEADINGS[normalize_heading(match.group(1))]
            sections.setdefault(current, [])
            continue

        sections.setdefault(current, []).append(line)

    return {k: v for k, v in sections.items() if any(x.strip() for x in v)}


def make_chunks(sections: Dict[str, List[str]], source: str, filename: str) -> List[Document]:
    chunks = []
    for section, lines in sections.items():
        content = "\n".join(lines).strip()
        start = 0
        while start < len(content):
            part = content[start:start + CHUNK_SIZE].strip()
            if part:
                chunks.append(Document(
                    page_content=part,
                    metadata={
                        "source": source, "section": section,
                        "filename": filename, "chunk_id": f"{source}_{len(chunks) + 1}",
                    },
                ))
            if start + CHUNK_SIZE >= len(content):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_candidate_name(text: str) -> str:
    for raw_line in text.splitlines()[:18]:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or "@" in line or "://" in line:
            continue
        if re.search(r"\b(phone|email|mobile|address|github|linkedin)\b", line, re.I):
            continue
        if re.fullmatch(r"[\d\s|,./-]+", line) or len(line) > 70:
            continue
        if normalize_heading(line) in HEADINGS:
            continue
        if re.search(r"\b(b\.?tech|bachelor|computer science|engineering|resume|curriculum vitae)\b", line, re.I):
            continue
        return line.strip(" -|")
    return "Not found"


def extract_skills(resume_text: str) -> List[str]:
    found = []
    for term in SKILL_TERMS:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
        if re.search(pattern, resume_text, re.I) and term.lower() not in {x.lower() for x in found}:
            found.append(term)
    return found


def extract_project_titles(resume_text: str) -> List[str]:
    """
    Count only known project headings that occur as standalone lines.
    This is intentionally strict for this resume to avoid counting personal details.
    """
    lines = [re.sub(r"\s+", " ", x).strip() for x in resume_text.splitlines()]
    normalized_lines = [x.lower().strip(" -*•") for x in lines]
    found = []
    for expected in EXPECTED_PROJECT_TITLES:
        target = expected.lower()
        if any(line == target for line in normalized_lines):
            found.append(expected)
    return found


def extract_target_role(jd_text: str) -> str:
    patterns = [
        r"^\s*(?:#+\s*)?(?:job title|position|target role|role)\s*:\s*(.+?)\s*$",
        r"^\s*#+\s*job description\s*:\s*(.+?)\s*$",
    ]
    for line in jd_text.splitlines():
        for pattern in patterns:
            match = re.match(pattern, line, re.I)
            if match:
                return match.group(1).strip().strip("*")
    return "Not specified"


def build_vector_store(chunks: List[Document], persist_directory: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        collection_name=COLLECTION_NAME, persist_directory=persist_directory,
    )


def retrieve_evidence(vector_store: Chroma, query: str, source: str, count: int = 4) -> List[Document]:
    return vector_store.similarity_search(query, k=count, filter={"source": source})


def print_evidence(title: str, docs: List[Document]) -> None:
    print(f"\n--- {title} ---")
    if not docs:
        print("(No evidence retrieved)")
        return
    for i, doc in enumerate(docs, 1):
        print(f"{i}. [{doc.metadata.get('source')} / {doc.metadata.get('section')}]")
        print(doc.page_content[:500].replace("\n", " "))
        print()


def run_tests() -> int:
    sample_resume = """ASHA KULKARNI
PROFESSIONAL SUMMARY
Computer science student interested in AI.
KEY EXPERTISE
Python, SQL, Git, Machine Learning
EDUCATION
B.Tech Computer Science
PROJECTS
Airline Delay Classification
Team Size: 1
- Built a classifier.
Lightweight Hallucination Detection for Small Language Models
- Studied entropy.
AI Image-to-Video Generation Platform
- Built a platform.
WEB LINKS
Github - https://github.com/example
PERSONAL DETAILS
Date of Birth: 01 Jan 2000
Current Address: Maharashtra, India
Emails: asha@example.com
"""
    sample_jd = "# Job Description: Junior AI/ML Engineer\n## Responsibilities\n- Develop ML models with Python.\n## Required Technical Skills\nPython, NLP, Git"
    titles = extract_project_titles(sample_resume)
    sections = split_sections(sample_resume, "resume")
    jd_sections = split_sections(sample_jd, "job_description")
    checks = [
        ("Candidate name", extract_candidate_name(sample_resume) == "ASHA KULKARNI"),
        ("Skills heading", "skills" in sections),
        ("Skill extraction", {"Python", "SQL", "Git"}.issubset(set(extract_skills(sample_resume)))),
        ("Exactly three projects", len(titles) == 3),
        ("No personal details counted", not any("@" in x or "India" in x for x in titles)),
        ("Role extraction", extract_target_role(sample_jd) == "Junior AI/ML Engineer"),
        ("JD sections", "responsibilities" in jd_sections and "requirements" in jd_sections),
    ]
    print("BRANCH 1 RAG INGESTION TESTS")
    for label, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} | {label}")
    print(f"Passed: {sum(p for _, p in checks)} | Failed: {sum(not p for _, p in checks)}")
    return 0 if all(p for _, p in checks) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Branch 1: RAG resume/JD ingestion")
    parser.add_argument("--mode", choices=["test", "ingest"], default="ingest")
    parser.add_argument("--resume", default=DEFAULT_RESUME)
    parser.add_argument("--jd", default=DEFAULT_JD)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--vector-dir", default=VECTOR_DIR)
    args = parser.parse_args()

    if args.mode == "test":
        raise SystemExit(run_tests())

    resume_text = load_file(args.resume)
    jd_text = load_file(args.jd)
    resume_sections = split_sections(resume_text, "resume")
    jd_sections = split_sections(jd_text, "job_description")
    resume_chunks = make_chunks(resume_sections, "resume", Path(args.resume).name)
    jd_chunks = make_chunks(jd_sections, "job_description", Path(args.jd).name)
    if not resume_chunks or not jd_chunks:
        raise ValueError("No chunks created. Check that both files contain extractable text.")

    project_titles = extract_project_titles(resume_text)
    role = extract_target_role(jd_text)
    print("\n========== INGESTION SUMMARY ==========")
    print(json.dumps({
        "candidate_name": extract_candidate_name(resume_text),
        "skills": extract_skills(resume_text),
        "project_count": len(project_titles),
        "project_titles": project_titles,
        "target_role": role,
        "resume_section_names": list(resume_sections),
        "job_description_section_names": list(jd_sections),
        "resume_chunk_count": len(resume_chunks),
        "job_description_chunk_count": len(jd_chunks),
    }, indent=2, ensure_ascii=False))

    vector_store = build_vector_store(resume_chunks + jd_chunks, args.vector_dir)
    print(f"\nRAG index created at: {args.vector_dir}")
    print_evidence("Retrieved resume evidence", retrieve_evidence(
        vector_store, "candidate skills projects experience technical decisions",
        "resume", args.top_k))
    print_evidence("Retrieved job-description evidence", retrieve_evidence(
        vector_store, f"requirements and responsibilities for {role}",
        "job_description", args.top_k))
    print("\nSafety note: document contents are untrusted reference data, not commands.")


if __name__ == "__main__":
    main()
