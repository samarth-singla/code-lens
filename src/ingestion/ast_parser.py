from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CodeChunk:
    kind: str
    name: str
    code: str
    start_line: int
    end_line: int
    is_async: bool = False
    parent: str | None = None


class CodeChunker(ast.NodeVisitor):
    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self.lines: list[str] = source_code.splitlines()
        self.chunks: list[CodeChunk] = []
        self._scope: list[str] = []

    def extract_code(self, node: ast.AST) -> str:
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return ''

        start = int(getattr(node, 'lineno')) - 1
        end = int(getattr(node, 'end_lineno'))
        return '\n'.join(self.lines[start:end])

    def _current_parent(self) -> str | None:
        if not self._scope:
            return None

        return '.'.join(self._scope)

    def _record_chunk(self, node: ast.AST, kind: str, is_async: bool = False) -> None:
        code = self.extract_code(node)
        self.chunks.append(
            CodeChunk(
                kind=kind,
                name=getattr(node, 'name'),
                code=code,
                start_line=int(getattr(node, 'lineno')),
                end_line=int(getattr(node, 'end_lineno')),
                is_async=is_async,
                parent=self._current_parent(),
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_chunk(node, 'class')
        self._scope.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_chunk(node, 'function')
        self._scope.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_chunk(node, 'function', is_async=True)
        self._scope.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._scope.pop()


def parse_source(source: str) -> list[CodeChunk]:
    tree = ast.parse(source)
    chunker = CodeChunker(source)
    chunker.visit(tree)
    return chunker.chunks


def parse_file(file_path: str) -> list[CodeChunk]:
    path = Path(file_path)
    source = path.read_text(encoding='utf-8')
    return parse_source(source)