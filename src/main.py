from __future__ import annotations

import asyncio

from agents.reviewer import build_agent, close_reviewer_deps, create_reviewer_deps

PROMPT = 'Code-Lens> '
EXIT_COMMANDS = {'exit', 'quit'}


async def run_cli() -> None:
    agent = build_agent()
    deps = await create_reviewer_deps()
    try:
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

            result = await agent.run(query, deps=deps)
            print(result.output)
    finally:
        await close_reviewer_deps(deps)


def main() -> None:
    asyncio.run(run_cli())


if __name__ == '__main__':
    main()