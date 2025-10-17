import asyncio
import os
from pathlib import Path


def load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned)


load_env()

from app.core.knowledge_search import queries

USER_ID = "255e34cc-3de9-4475-8dc1-a928084c398d"
ORG_ID = "1"
DOCUMENT_ID_PDF = "1_255e34cc-3de9-4475-8dc1-a928084c398d_open-webui_8b900bd9-9c4a-4711-933d-85ba4224a45a_python.pdf"
DOCUMENT_ID_IMAGE = "1_255e34cc-3de9-4475-8dc1-a928084c398d_open-webui_85130152-f2b5-4a47-ab5d-b93b94fa93e4_SSN 2024-12-16 21_13_07.jpeg"
DOCUMENT_ID_CSV = "1_255e34cc-3de9-4475-8dc1-a928084c398d_open-webui_f9688c15-2a4a-456a-9003-0e3b0c34bb78_bedrock-long-term-api-key.csv"
PAGE_NUMBER = 24
ENTITY_ID = "IPv4"
COLUMN_NAME = "API key"
CALLS = [
    ("execute_search_documents", queries.execute_search_documents, (), {"filename_pattern": None, "file_type": None, "category": None, "source": None, "limit": 5}),
    ("execute_get_document_details", queries.execute_get_document_details, (DOCUMENT_ID_PDF,), {}),
    ("execute_search_pages_by_content", queries.execute_search_pages_by_content, ("Python programming",), {"similarity_threshold": 0.6, "limit": 5}),
    ("execute_search_pages_in_document", queries.execute_search_pages_in_document, (DOCUMENT_ID_PDF,), {"page_number": PAGE_NUMBER, "is_tabular": None}),
    ("execute_get_page_details", queries.execute_get_page_details, (DOCUMENT_ID_PDF, PAGE_NUMBER), {}),
    ("execute_search_entities_by_semantic", queries.execute_search_entities_by_semantic, ("network protocol",), {"entity_type": "protocol", "similarity_threshold": 0.6, "limit": 5}),
    ("execute_search_entities_by_type", queries.execute_search_entities_by_type, ("protocol",), {"limit": 20}),
    ("execute_search_entities_by_text", queries.execute_search_entities_by_text, ("IPv4",), {"entity_type": "protocol", "limit": 20}),
    ("execute_get_entity_details", queries.execute_get_entity_details, (ENTITY_ID, DOCUMENT_ID_PDF), {}),
    ("execute_find_entity_relationships", queries.execute_find_entity_relationships, (ENTITY_ID, DOCUMENT_ID_PDF), {"direction": "both", "limit": 20}),
    ("execute_search_relationships_by_type", queries.execute_search_relationships_by_type, ("related_to",), {"limit": 20}),
    ("execute_search_relationships_semantic", queries.execute_search_relationships_semantic, ("network connection",), {"similarity_threshold": 0.6, "limit": 5}),
    ("execute_traverse_entity_graph", queries.execute_traverse_entity_graph, (ENTITY_ID, DOCUMENT_ID_PDF), {"max_hops": 2, "limit": 20}),
    ("execute_search_columns", queries.execute_search_columns, (), {"column_name_pattern": "API", "semantic_query": None, "document_id": DOCUMENT_ID_CSV, "limit": 10}),
    ("execute_get_column_values", queries.execute_get_column_values, (COLUMN_NAME, DOCUMENT_ID_CSV), {"limit": 50}),
    ("execute_search_row_values", queries.execute_search_row_values, ("ABSK",), {"column_name": COLUMN_NAME, "document_id": DOCUMENT_ID_CSV, "limit": 20}),
    ("execute_query_tabular_data", queries.execute_query_tabular_data, (DOCUMENT_ID_CSV,), {"sheet_name": None, "row_index": 0}),
    ("execute_hybrid_search", queries.execute_hybrid_search, ("network security", ["Page", "Entity", "Column"]), {"include_relationships": True, "limit": 10}),
    ("execute_breadth_first_search", queries.execute_breadth_first_search, ("Document", DOCUMENT_ID_PDF), {"max_depth": 2, "relationship_types": None, "limit": 20}),
    ("execute_get_entity_context", queries.execute_get_entity_context, (ENTITY_ID, DOCUMENT_ID_PDF), {"include_pages": True, "include_related_entities": True, "include_document": True}),
]


def prune(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if "embedding" in key.lower():
                result[key] = "<omitted>"
            else:
                result[key] = prune(item)
        return result
    if isinstance(value, list):
        return [prune(value[0])] if value else []
    return value


def summarize(value):
    if isinstance(value, list):
        return {"count": len(value), "sample": prune(value[0]) if value else None}
    return prune(value)


async def main():
    for name, func, args, kwargs in CALLS:
        try:
            result = await func(USER_ID, ORG_ID, *args, **kwargs)
            print(name, summarize(result))
        except Exception as exc:
            print(name, {"error": str(exc)})


asyncio.run(main())
