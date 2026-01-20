import io
import zipfile

import frontmatter
import requests
from google import genai


def read_repo_markdown_files(
    repo_owner: str, repo_name: str, branch: str = "main"
) -> list:
    """Read Markdown and React Markdown files from a GitHub repository."""
    zip_url = (
        f"https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/{branch}"
    )
    response = requests.get(zip_url)

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Failed to download repository: {e}") from e

    extracted_data = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for file_info in zf.infolist():
            if not file_info.filename.lower().endswith((".md", ".mdx")):
                continue

            with zf.open(file_info) as file:
                content = file.read().decode("utf-8", errors="ignore")

                post = frontmatter.loads(content)
                data = post.to_dict()
                data["filename"] = file_info.filename
                extracted_data.append(data)

    return extracted_data


PROMPT_TEMPLATE = """
Split the provided document into logical sections for a Q&A system.

Each section should be self-contained and focused on a specific topic or concept.

CRITICAL RULES:
1. ONLY use exact text from the document: copy text VERBATIM
2. DO NOT add explanations, introductions, summaries, or commentary
3. Do NOT add phrases like "This section covers..." or "The document explains..."

<DOCUMENT>
{document}
</DOCUMENT>

Output format should be:

## [Short descriptive title]

[Exact verbatim text from document]

---

## [Another short descriptive title]

[Another exact verbatim text from document]

---
... and so on.
"""


client = genai.Client()


def chunk_document(document: str, model: str = "gemini-2.5-flash-lite") -> list[str]:
    """Chunk a document into logical sections using a powerful LLM."""
    prompt = PROMPT_TEMPLATE.format(document=document)
    response = client.models.generate_content(model=model, contents=prompt)

    if not response.text:
        return []

    sections = response.text.split("---")
    results = []
    for section in sections:
        section = section.strip()
        if section:
            results.append(section)

    return results
