import json
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_ollama import ChatOllama

from graph_utils import get_neo4j_config, has_neo4j_config

try:
    from neo4j import GraphDatabase
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 neo4j 驱动: pip install neo4j") from exc


CLEAN_NOVELS_DIR = "./clean_novels"
MAX_CHAPTERS_PER_BOOK = None
MAX_CHARS_PER_CHAPTER = None


def split_text_by_chapters(text: str) -> List[Dict[str, Any]]:
    chapter_pattern = r"(第[0-9一二三四五六七八九十百千]+[章卷][^\n]*)"
    parts = re.split(chapter_pattern, text)

    chapters: List[Dict[str, Any]] = []
    current_title = "序章/前言"

    if parts[0].strip():
        chapters.append({"title": current_title, "content": parts[0].strip()})

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if not content:
            continue
        chapters.append({"title": title, "content": content})

    return chapters


def _unique_strings(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_persons(raw_items: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    persons: List[Dict[str, Any]] = []
    alias_map: Dict[str, str] = {}
    for item in raw_items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("person") or item.get("entity")
            aliases = item.get("aliases") or []
        else:
            name = str(item).strip()
            aliases = []
        if not name:
            continue
        aliases = [str(a).strip() for a in aliases if str(a).strip()]
        persons.append({"name": name, "aliases": aliases})
        for alias in aliases:
            alias_map[alias] = name
    return persons, alias_map


def _normalize_name(name: str, alias_map: Dict[str, str]) -> str:
    return alias_map.get(name, name)


def extract_graph_data(
    llm: ChatOllama,
    book_name: str,
    chapter_title: str,
    chapter_index: int,
    text: str,
) -> Dict[str, Any]:
    snippet = text if MAX_CHARS_PER_CHAPTER is None else text[:MAX_CHARS_PER_CHAPTER]
    prompt = (
        "你是信息抽取系统。请从小说片段中抽取实体与关系。\n"
        "输出JSON，格式如下：\n"
        "{\n"
        "  \"persons\": [\n"
        "    {\"name\": \"人名1\", \"aliases\": [\"别名1\", \"别名2\"]}\n"
        "  ],\n"
        "  \"places\": [\"地点1\"],\n"
        "  \"events\": [\n"
        "    {\"name\": \"事件名\", \"participants\": [\"人名1\"], \"place\": \"地点1\"}\n"
        "  ],\n"
        "  \"relations\": [\n"
        "    {\"source\": \"人名1\", \"target\": \"人名2\", \"type\": \"关系类型\", \"evidence\": \"证据短语\"}\n"
        "  ]\n"
        "}\n"
        "要求：\n"
        "- 只输出JSON，不要多余文字。\n"
        "- 如果没有则输出空数组。\n"
        "- 实体名尽量精简，persons使用主要称呼，其它称呼放 aliases。\n"
        f"书名: {book_name}\n"
        f"章节: {chapter_title}\n"
        "片段: \n"
        f"{snippet}"
    )

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        data = json.loads(text)
    except Exception:
        data = {"persons": [], "places": [], "events": [], "relations": []}

    persons, alias_map = _normalize_persons(data.get("persons", []))
    data["persons"] = persons
    data["places"] = _unique_strings(data.get("places", []))
    data["events"] = data.get("events", [])
    data["relations"] = data.get("relations", [])
    data["chapter_title"] = chapter_title
    data["chapter_index"] = chapter_index

    for event in data["events"]:
        participants = event.get("participants", []) if isinstance(event, dict) else []
        event["participants"] = [_normalize_name(p, alias_map) for p in participants]

    for rel in data["relations"]:
        if not isinstance(rel, dict):
            continue
        rel["source"] = _normalize_name(rel.get("source", ""), alias_map)
        rel["target"] = _normalize_name(rel.get("target", ""), alias_map)

    return data


def ingest_chapter(session, book_name: str, chapter: Dict[str, Any], payload: Dict[str, Any]):
    chapter_title = payload["chapter_title"]
    chapter_index = payload["chapter_index"]

    session.run(
        "MERGE (c:Chapter {title: $title, source: $source}) "
        "SET c.index = $index",
        {"title": chapter_title, "source": book_name, "index": chapter_index},
    )

    persons = payload.get("persons", [])
    if persons:
        session.run(
            "UNWIND $rows AS row "
            "MERGE (p:Person {name: row.name}) "
            "SET p.aliases = coalesce(p.aliases, []) + row.aliases "
            "WITH p "
            "MATCH (c:Chapter {title: $title, source: $source}) "
            "MERGE (p)-[:APPEARS_IN]->(c)",
            {
                "rows": persons,
                "title": chapter_title,
                "source": book_name,
            },
        )

    places = payload.get("places", [])
    if places:
        session.run(
            "UNWIND $rows AS row "
            "MERGE (pl:Place {name: row.name})",
            {"rows": [{"name": name} for name in places]},
        )

    events = payload.get("events", [])
    if events:
        session.run(
            "UNWIND $rows AS row "
            "MERGE (e:Event {name: row.name, source: $source}) "
            "WITH e, row "
            "MATCH (c:Chapter {title: $title, source: $source}) "
            "MERGE (e)-[:HAPPENS_IN]->(c) "
            "FOREACH (pname IN row.participants | "
            "  MERGE (p:Person {name: pname}) "
            "  MERGE (p)-[:IN_EVENT]->(e) "
            ") "
            "FOREACH (plname IN CASE WHEN row.place IS NULL OR row.place = '' THEN [] ELSE [row.place] END | "
            "  MERGE (pl:Place {name: plname}) "
            "  MERGE (e)-[:AT]->(pl) "
            ")",
            {
                "rows": events,
                "title": chapter_title,
                "source": book_name,
            },
        )

    relations = payload.get("relations", [])
    if relations:
        session.run(
            "UNWIND $rows AS row "
            "MERGE (a:Person {name: row.source}) "
            "MERGE (b:Person {name: row.target}) "
            "MERGE (a)-[r:RELATES {type: row.type}]->(b) "
            "SET r.evidence = row.evidence",
            {"rows": relations},
        )


def main():
    if not has_neo4j_config():
        raise RuntimeError("请先设置 NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD/NEO4J_DATABASE 环境变量")

    uri, user, password, database = get_neo4j_config()
    driver = GraphDatabase.driver(uri, auth=(user, password))

    llm = ChatOllama(model="qwen2.5:7b", temperature=0, timeout=120.0)

    with driver.session(database=database) as session:
        for filename in os.listdir(CLEAN_NOVELS_DIR):
            if not filename.endswith(".txt"):
                continue
            book_name = filename.replace(".txt", "")
            file_path = os.path.join(CLEAN_NOVELS_DIR, filename)
            print(f"处理书籍: {book_name}")

            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            chapters = split_text_by_chapters(full_text)
            chapter_list = chapters
            if MAX_CHAPTERS_PER_BOOK is not None:
                chapter_list = chapters[:MAX_CHAPTERS_PER_BOOK]

            for idx, chapter in enumerate(chapter_list, start=1):
                print(f"  - 抽取章节: {chapter['title']}")
                payload = extract_graph_data(
                    llm, book_name, chapter["title"], idx, chapter["content"]
                )
                ingest_chapter(session, book_name, chapter, payload)

            for idx in range(1, len(chapter_list)):
                title_prev = chapter_list[idx - 1]["title"]
                title_next = chapter_list[idx]["title"]
                session.run(
                    "MATCH (a:Chapter {title: $title_a, source: $source}) "
                    "MATCH (b:Chapter {title: $title_b, source: $source}) "
                    "MERGE (a)-[:NEXT]->(b)",
                    {"title_a": title_prev, "title_b": title_next, "source": book_name},
                )

    driver.close()
    print("图谱抽取完成。")


if __name__ == "__main__":
    main()
