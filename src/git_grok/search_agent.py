from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext, Tool

from .defaults import DEFAULT_GEMINI_MODEL_ID

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult

    from .vector_engine import QdrantEngine


STATIC_INSTRUCTION = """
You are a helpful assistant that answers questions about documentations of a codebase.

Before answering, use the provided search tool first to find relevant content.

Use the search results to provide accurate and concise answers.

If the search results do not contain relevant information, say so explicitly and
and provide a general guidance instead.

Always cite the file names from which you obtained the information in your answer.
"""


@dataclass
class SearchDependencies:
    """Dependencies for the search agent."""

    vector_engine: QdrantEngine


def search_tool(
    ctx: RunContext[SearchDependencies], query: str, top_k: int = 5
) -> list[str]:
    """Search the documentation for relevant content."""
    return ctx.deps.vector_engine.search(query, top_k=top_k)


def create_search_agent(model_id: str | None = None) -> Agent:
    """Create a search agent with the specified model ID.

    Parameters
    ----------
    model_id
        The model ID to use for the agent. If None, uses the default model ID.

    Returns
    -------
    Agent
        The created search agent.
    """
    model_id = model_id or f"google-gla:{DEFAULT_GEMINI_MODEL_ID}"
    return Agent(
        model_id,
        name="doc-search-agent",
        deps_type=SearchDependencies,
        instructions=STATIC_INSTRUCTION,
        tools=[Tool(search_tool, takes_ctx=True)],
    )


def answer_sync(
    question: str,
    vector_engine: QdrantEngine,
    model_id: str | None = None,
) -> AgentRunResult[str]:
    """Answer a question using the search agent."""
    deps = SearchDependencies(vector_engine=vector_engine)
    search_agent = create_search_agent(model_id)
    return search_agent.run_sync(question, deps=deps)  # ty: ignore[invalid-argument-type]


async def answer(
    question: str,
    vector_engine: QdrantEngine,
    model_id: str | None = None,
) -> AgentRunResult[str]:
    """Answer a question asynchronously using the search agent."""
    deps = SearchDependencies(vector_engine=vector_engine)
    search_agent = create_search_agent(model_id)
    return await search_agent.run(question, deps=deps)  # ty: ignore[invalid-argument-type]
