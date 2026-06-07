from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Sequence

import asyncpg
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pgvector.asyncpg import register_vector
from sentence_transformers import SentenceTransformer

OLLAMA_BASE_URL = 'http://localhost:11434'
DATABASE_URL = 'postgresql://admin:admin@localhost:5432/vector_db'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'llama3'
TOP_K = 3


@dataclass(slots=True)
class ReviewerDeps:
    pool: asyncpg.Pool
    embedding_model: SentenceTransformer


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    file_path: str
    chunk_name: str
    code_content: str


async def _init_connection(connection: asyncpg.Connection) -> None:
    await register_vector(connection)


async def create_reviewer_deps() -> ReviewerDeps:
    pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=1,
        max_size=5,
        init=_init_connection,
    )
    embedding_model = await asyncio.to_thread(SentenceTransformer, EMBEDDING_MODEL)
    return ReviewerDeps(pool=pool, embedding_model=embedding_model)


async def close_reviewer_deps(deps: ReviewerDeps) -> None:
    await deps.pool.close()


async def embed_query(deps: ReviewerDeps, query: str) -> list[float]:
    embedding = await asyncio.to_thread(
        deps.embedding_model.encode,
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embedding.tolist()


async def search_relevant_chunks(
    deps: ReviewerDeps,
    query: str,
) -> list[RetrievedChunk]:
    query_embedding = await embed_query(deps, query)
    sql = '''
    SELECT file_path, chunk_name, code_content
    FROM code_chunks
    ORDER BY embedding <=> $1
    LIMIT $2
    '''

    async with deps.pool.acquire() as connection:
        rows = await connection.fetch(sql, query_embedding, TOP_K)

    return [
        RetrievedChunk(
            file_path=row['file_path'],
            chunk_name=row['chunk_name'],
            code_content=row['code_content'],
        )
        for row in rows
    ]


def format_retrieved_chunks(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return 'No relevant code chunks were found.'

    return '\n\n'.join(
        f'File: {chunk.file_path}\nChunk: {chunk.chunk_name}\nCode:\n{chunk.code_content}'
        for chunk in chunks
    )


def build_agent() -> Agent[ReviewerDeps, str]:
    model = OllamaModel(
        LLM_MODEL,
        provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
    )
    agent = Agent[ReviewerDeps, str](
        model,
        deps_type=ReviewerDeps,
        system_prompt=(
            'You are Code-Lens, a senior software engineer. '
            'Always use the retrieve_code_context tool before answering. '
            'Use the retrieved repository context to produce a concise code review or bug fix.'
        ),
    )

    @agent.tool
    async def retrieve_code_context(ctx: RunContext[ReviewerDeps], query: str) -> str:
        chunks = await search_relevant_chunks(ctx.deps, query)
        return format_retrieved_chunks(chunks)

    return agent


async def review_code(query: str) -> str:
    deps = await create_reviewer_deps()
    try:
        agent = build_agent()
        result = await agent.run(query, deps=deps)
        return result.output
    finally:
        await close_reviewer_deps(deps)