from __future__ import annotations

import logging
import os

import asyncpg

DATABASE_URL = 'postgresql://admin:admin@localhost:5432/vector_db'

logger = logging.getLogger(__name__)

CREATE_VECTOR_EXTENSION_SQL = 'CREATE EXTENSION IF NOT EXISTS vector;'

CREATE_CODE_CHUNKS_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS code_chunks (
    id SERIAL PRIMARY KEY,
    file_path TEXT,
    chunk_type TEXT,
    chunk_name TEXT,
    code_content TEXT,
    embedding vector(384)
);
'''

CREATE_CODE_CHUNKS_INDEX_SQL = '''
CREATE INDEX IF NOT EXISTS code_chunks_embedding_hnsw_idx
    ON code_chunks
    USING hnsw (embedding vector_cosine_ops);
'''


def get_database_url() -> str:
    return os.getenv('DATABASE_URL', DATABASE_URL)


async def _create_pool(database_url: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=5,
    )


async def init_db() -> None:
    pool = None
    database_url = get_database_url()

    try:
        logger.info('Connecting to PostgreSQL at %s', database_url)
        pool = await _create_pool(database_url)
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(CREATE_VECTOR_EXTENSION_SQL)
                await connection.execute(CREATE_CODE_CHUNKS_TABLE_SQL)
                await connection.execute(CREATE_CODE_CHUNKS_INDEX_SQL)
        logger.info('Database initialized successfully')
    except (OSError, asyncpg.PostgresError) as error:
        logger.error('Could not initialize database at %s: %s', database_url, error)
        raise RuntimeError(
            'Unable to connect to PostgreSQL. Start the local database with '
            '`docker compose up -d postgres` or set DATABASE_URL to a '
            'reachable PostgreSQL instance.'
        ) from error
    finally:
        if pool is not None:
            await pool.close()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    try:
        asyncio.run(init_db())
    except RuntimeError as error:
        logger.error('%s', error)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()