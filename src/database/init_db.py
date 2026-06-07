from __future__ import annotations

import os

import asyncpg

DEFAULT_DATABASE_URL = 'postgresql://postgres:postgres@localhost:5432/postgres'

CREATE_VECTOR_EXTENSION_SQL = 'CREATE EXTENSION IF NOT EXISTS vector;'

CREATE_CODE_CHUNKS_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS code_chunks (
    id SERIAL PRIMARY KEY,
    file_path TEXT,
    chunk_type TEXT,
    chunk_name TEXT,
    code_content TEXT,
    embedding vector(1536)
);
'''

CREATE_CODE_CHUNKS_INDEX_SQL = '''
CREATE INDEX IF NOT EXISTS code_chunks_embedding_hnsw_idx
    ON code_chunks
    USING hnsw (embedding vector_cosine_ops);
'''


def get_database_url() -> str:
    return os.getenv('DATABASE_URL', DEFAULT_DATABASE_URL)


async def init_db() -> None:
    pool = await asyncpg.create_pool(dsn=get_database_url(), min_size=1, max_size=5)

    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(CREATE_VECTOR_EXTENSION_SQL)
                await connection.execute(CREATE_CODE_CHUNKS_TABLE_SQL)
                await connection.execute(CREATE_CODE_CHUNKS_INDEX_SQL)
    finally:
        await pool.close()


def main() -> None:
    import asyncio

    asyncio.run(init_db())


if __name__ == '__main__':
    main()