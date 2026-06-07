from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import asyncpg
from openai import AsyncOpenAI as AsyncClient
from pgvector.asyncpg import register_vector

from database.init_db import get_database_url
from ingestion.ast_parser import CodeChunk, parse_file

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_BATCH_SIZE = 16
MAX_CONCURRENT_BATCHES = 4

INSERT_CODE_CHUNKS_SQL = '''
INSERT INTO code_chunks (
    file_path,
    chunk_type,
    chunk_name,
    code_content,
    embedding
)
VALUES ($1, $2, $3, $4, $5)
'''


class ChunkRecord(TypedDict):
    type: str
    name: str
    code: str
    file_path: str


async def _init_connection(connection: asyncpg.Connection) -> None:
    await register_vector(connection)


def _to_chunk_record(file_path: Path, chunk: CodeChunk) -> ChunkRecord:
    return ChunkRecord(
        type=chunk.kind,
        name=chunk.name,
        code=chunk.code,
        file_path=str(file_path),
    )


def _batch_records(records: Sequence[ChunkRecord]) -> list[list[ChunkRecord]]:
    return [
        list(records[index : index + EMBEDDING_BATCH_SIZE])
        for index in range(0, len(records), EMBEDDING_BATCH_SIZE)
    ]


def _collect_chunk_records(repo_path: Path) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []

    for file_path in sorted(repo_path.rglob('*.py')):
        if not file_path.is_file():
            continue

        try:
            chunks = parse_file(str(file_path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            logger.exception('Failed to parse %s', file_path)
            continue

        for chunk in chunks:
            records.append(_to_chunk_record(file_path, chunk))

    return records


async def _embed_batch(
    client: AsyncClient,
    batch_index: int,
    batch: Sequence[ChunkRecord],
) -> list[tuple[str, str, str, str, list[float]]]:
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[record['code'] for record in batch],
        )
    except Exception:
        logger.exception('Embedding request failed for batch %d', batch_index)
        raise

    embeddings = [item.embedding for item in response.data]
    return [
        (
            record['file_path'],
            record['type'],
            record['name'],
            record['code'],
            embedding,
        )
        for record, embedding in zip(batch, embeddings, strict=True)
    ]


async def ingest_repository(repo_path: str) -> None:
    repo_root = Path(repo_path).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise FileNotFoundError(f'Repository path does not exist: {repo_root}')

    logger.info('Scanning repository for Python files: %s', repo_root)
    records = _collect_chunk_records(repo_root)

    if not records:
        logger.info('No Python chunks found under %s', repo_root)
        return

    logger.info('Collected %d chunks from %s', len(records), repo_root)
    batches = _batch_records(records)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

    async with AsyncClient() as client:
        async with asyncpg.create_pool(
            dsn=get_database_url(),
            min_size=1,
            max_size=5,
            init=_init_connection,
        ) as pool:
            async def _process_batch(
                batch_index: int,
                batch: Sequence[ChunkRecord],
            ) -> list[tuple[str, str, str, str, list[float]]]:
                async with semaphore:
                    return await _embed_batch(client, batch_index, batch)

            batch_results = await asyncio.gather(
                *(
                    _process_batch(batch_index, batch)
                    for batch_index, batch in enumerate(batches, start=1)
                )
            )

            rows = [row for batch_rows in batch_results for row in batch_rows]

            try:
                async with pool.acquire() as connection:
                    async with connection.transaction():
                        await connection.executemany(INSERT_CODE_CHUNKS_SQL, rows)
            except Exception:
                logger.exception('Failed to insert %d chunks into Postgres', len(rows))
                raise

    logger.info('Ingested %d chunks from %s', len(records), repo_root)


async def main() -> None:
    import os

    await ingest_repository(os.getcwd())


if __name__ == '__main__':
    asyncio.run(main())