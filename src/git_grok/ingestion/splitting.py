from functools import lru_cache

from google import genai
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

from ..defaults import (
    DEFAULT_GEMINI_MODEL_ID,
    DEFAULT_MAX_CHUNK_OVERLAP,
    DEFAULT_MAX_TOKENS_PER_CHUNK,
    DEFAULT_TOKENIZER_MODEL_ID,
)

SECTION_DELIMITER = "<<<SECTION_BREAK>>>"

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

<<<SECTION_BREAK>>>

## [Another short descriptive title]

[Another exact verbatim text from document]

<<<SECTION_BREAK>>>
... and so on.
"""


@lru_cache(maxsize=1)
def _get_genai_client() -> genai.Client:
    """Get or create a cached Google GenAI client."""
    return genai.Client()


def llm_split(text: str, gemini_model_id: str | None = None) -> list[str]:
    """Split a text semantically using Google Gemini model."""
    client = _get_genai_client()
    response = client.models.generate_content(
        model=gemini_model_id or DEFAULT_GEMINI_MODEL_ID,
        contents=PROMPT_TEMPLATE.format(document=text),
    )
    if not response.text:
        return []

    sections = response.text.split(SECTION_DELIMITER)
    return [section.strip() for section in sections if section.strip()]


@lru_cache(maxsize=8)
def _get_tokenizer(tokenizer_model_id: str) -> Tokenizer:
    """Get or create a cached tokenizer."""
    return Tokenizer.from_pretrained(tokenizer_model_id)


def token_split(text: str, tokenizer_model_id: str | None = None) -> list[str]:
    """Split a text semantically using a tokenizer-based text splitter."""
    tokenizer = _get_tokenizer(tokenizer_model_id or DEFAULT_TOKENIZER_MODEL_ID)
    splitter = TextSplitter.from_huggingface_tokenizer(
        tokenizer,
        capacity=DEFAULT_MAX_TOKENS_PER_CHUNK,
        overlap=DEFAULT_MAX_CHUNK_OVERLAP,
        trim=True,
    )
    return splitter.chunks(text)
