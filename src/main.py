from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if __package__ is None or __package__ == '':
    sys.path.append(str(Path(__file__).resolve().parent))

from agents.reviewer import review_code

PROMPT = 'Code-Lens> '
EXIT_COMMANDS = {'exit', 'quit'}


async def run_cli() -> None:
    while True:
        try:
            query = await asyncio.to_thread(input, PROMPT)
        except (EOFError, KeyboardInterrupt):
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in EXIT_COMMANDS:
            break

        print(await review_code(query))


def main() -> None:
    asyncio.run(run_cli())


if __name__ == '__main__':
    main()