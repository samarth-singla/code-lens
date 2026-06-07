import ast
from pathlib import Path

class CodeChunker(ast.NodeVisitor):
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.chunks = []

    def extract_code(self, node):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""
        start = node.lineno - 1
        end = node.end_lineno
        return "\n".join(self.lines[start:end])

    def visit_ClassDef(self, node):
        class_code = self.extract_code(node)
        self.chunks.append({
            "type": "class",
            "name": node.name,
            "code": class_code
        })
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        func_code = self.extract_code(node)
        self.chunks.append({
            "type": "function",
            "name": node.name,
            "code": func_code
        })
        self.generic_visit(node)

def parse_file(file_path: str) -> list:
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    
    chunker = CodeChunker(source)
    chunker.visit(tree)
    return chunker.chunks