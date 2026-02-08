import os
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None


DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_USER = "neo4j"
DEFAULT_DATABASE = "neo4j"


def _get_neo4j_config() -> Tuple[str, str, str, str]:
    uri = os.getenv("NEO4J_URI", DEFAULT_URI)
    user = os.getenv("NEO4J_USER", DEFAULT_USER)
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", DEFAULT_DATABASE)
    return uri, user, password, database


def _has_neo4j_config() -> bool:
    _uri, _user, password, _database = _get_neo4j_config()
    return bool(password)


def get_neo4j_config() -> Tuple[str, str, str, str]:
    return _get_neo4j_config()


def has_neo4j_config() -> bool:
    return _has_neo4j_config()


def _is_safe_read_cypher(cypher: str) -> bool:
    lowered = cypher.strip().lower()
    if not (lowered.startswith("match") or lowered.startswith("call")):
        return False
    forbidden = ["create ", "merge ", "delete ", "set ", "drop ", "remove ", "load csv", "apoc.create"]
    return not any(token in lowered for token in forbidden)


def run_read_cypher(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if GraphDatabase is None:
        return []
    if not _has_neo4j_config():
        return []
    if not _is_safe_read_cypher(cypher):
        return []

    uri, user, password, database = _get_neo4j_config()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]
    finally:
        driver.close()


def _format_node(node: Any) -> str:
    if node is None:
        return ""
    try:
        labels = list(node.labels)
        props = dict(node)
    except Exception:
        return str(node)
    name = props.get("name") or props.get("title") or props.get("source") or ""
    label = labels[0] if labels else "Node"
    return f"{label}({name})" if name else label


def _format_path(path: Any) -> str:
    try:
        nodes = list(path.nodes)
        rels = list(path.relationships)
    except Exception:
        return str(path)

    if not nodes:
        return ""
    parts = [_format_node(nodes[0])]
    for idx, rel in enumerate(rels):
        rel_type = getattr(rel, "type", None)
        rel_name = rel_type() if callable(rel_type) else str(rel_type)
        parts.append(f"-[{rel_name}]->")
        parts.append(_format_node(nodes[idx + 1]))
    return " ".join(parts)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        return _format_path(value)
    if hasattr(value, "labels"):
        return _format_node(value)
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _format_rows(rows: List[Dict[str, Any]], limit: int = 20) -> str:
    if not rows:
        return ""
    lines = []
    for row in rows[:limit]:
        parts = []
        for key, value in row.items():
            parts.append(f"{key}={_format_value(value)}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def _collect_paths(rows: List[Dict[str, Any]], limit: int = 10) -> List[str]:
    paths: List[str] = []
    for row in rows:
        for value in row.values():
            if hasattr(value, "nodes") and hasattr(value, "relationships"):
                text = _format_path(value)
                if text:
                    paths.append(text)
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, "nodes") and hasattr(item, "relationships"):
                        text = _format_path(item)
                        if text:
                            paths.append(text)
        if len(paths) >= limit:
            break
    return paths[:limit]


def generate_cypher_with_llm(question: str, llm: Any) -> Optional[Dict[str, Any]]:
    prompt = (
        "你是Neo4j图数据库专家。根据问题生成只读Cypher查询。\n"
        "图谱模式:\n"
        "- (:Person {name})\n"
        "- (:Place {name})\n"
        "- (:Event {name, source})\n"
        "- (:Chapter {title, index, source})\n"
        "关系:\n"
        "- (:Person)-[:RELATES {type}]->(:Person)\n"
        "- (:Person)-[:APPEARS_IN]->(:Chapter)\n"
        "- (:Person)-[:IN_EVENT]->(:Event)\n"
        "- (:Event)-[:HAPPENS_IN]->(:Chapter)\n"
        "- (:Event)-[:AT]->(:Place)\n"
        "- (:Chapter)-[:NEXT]->(:Chapter)\n"
        "人物别名存放在 Person.aliases 列表中。查询人名时优先使用:\n"
        "WHERE p.name = $name OR $name IN p.aliases\n"
        "要求返回路径，优先使用 MATCH p=... RETURN p LIMIT 20\n"
        "只输出JSON，格式: {\"cypher\":\"...\", \"params\":{}}\n"
        "不要输出解释文字。\n"
        f"问题: {question}"
    )

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        data = json.loads(text)
        if not isinstance(data, dict) or "cypher" not in data:
            return None
        return data
    except Exception:
        return None


def graph_search_with_llm(question: str, llm: Any) -> str:
    if not _has_neo4j_config():
        return ""
    payload = generate_cypher_with_llm(question, llm)
    if not payload:
        return ""

    cypher = payload.get("cypher", "").strip()
    params = payload.get("params", {}) if isinstance(payload.get("params"), dict) else {}
    if not _is_safe_read_cypher(cypher):
        return ""

    rows = run_read_cypher(cypher, params)
    if not rows:
        return ""

    formatted = _format_rows(rows)
    path_lines = _collect_paths(rows)
    path_block = "\n".join(path_lines)
    if path_block:
        return f"Cypher: {cypher}\nPath:\n{path_block}\n结果:\n{formatted}"
    return f"Cypher: {cypher}\n结果:\n{formatted}"