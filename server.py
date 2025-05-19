from typing import Dict, List, Optional
import chromadb
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
import argparse
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import datetime

# Initialize FastMCP server
mcp = FastMCP("history")

# Global variables
_chroma_client = None


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="FastMCP server for Chroma DB")
    parser.add_argument(
        "--client-type",
        choices=["http", "cloud", "persistent", "ephemeral"],
        default=os.getenv("CHROMA_CLIENT_TYPE", "ephemeral"),
        help="Type of Chroma client to use",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("CHROMA_DATA_DIR"),
        help="Directory for persistent client data (only used with persistent client)",
    )
    parser.add_argument(
        "--host",
        help="Chroma host (required for http client)",
        default=os.getenv("CHROMA_HOST"),
    )
    parser.add_argument(
        "--port",
        help="Chroma port (optional for http client)",
        default=os.getenv("CHROMA_PORT"),
    )
    parser.add_argument(
        "--custom-auth-credentials",
        help="Custom auth credentials (optional for http client)",
        default=os.getenv("CHROMA_CUSTOM_AUTH_CREDENTIALS"),
    )
    parser.add_argument(
        "--tenant",
        help="Chroma tenant (optional for http client)",
        default=os.getenv("CHROMA_TENANT"),
    )
    parser.add_argument(
        "--database",
        help="Chroma database (required if tenant is provided)",
        default=os.getenv("CHROMA_DATABASE"),
    )
    parser.add_argument(
        "--api-key",
        help="Chroma API key (required if tenant is provided)",
        default=os.getenv("CHROMA_API_KEY"),
    )
    parser.add_argument(
        "--ssl",
        help="Use SSL (optional for http client)",
        type=lambda x: x.lower() in ["true", "yes", "1", "t", "y"],
        default=os.getenv("CHROMA_SSL", "true").lower()
        in ["true", "yes", "1", "t", "y"],
    )
    parser.add_argument(
        "--dotenv-path",
        help="Path to .env file",
        default=os.getenv("CHROMA_DOTENV_PATH", ".chroma_env"),
    )
    return parser


def get_chroma_client(args=None):
    """Get or create the global Chroma client instance. Always returns a persistent client."""
    global _chroma_client
    if _chroma_client is None:
        if args is None:
            parser = create_parser()
            args = parser.parse_args()
        load_dotenv(dotenv_path=args.dotenv_path)
        data_dir = args.data_dir or os.getenv("CHROMA_DATA_DIR")
        if not data_dir:
            raise ValueError(
                "Data directory must be provided via --data-dir flag or CHROMA_DATA_DIR environment variable."
            )
        _chroma_client = chromadb.PersistentClient(path=data_dir)
    return _chroma_client


def _validate_non_empty_str(val, name):
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _validate_non_empty_list(val, name):
    if (
        not isinstance(val, list)
        or not val
        or not all(isinstance(x, str) and x.strip() for x in val)
    ):
        raise ValueError(f"{name} must be a non-empty list of strings.")


def _validate_positive_int(val, name):
    if val is not None and (not isinstance(val, int) or val < 1):
        raise ValueError(f"{name} must be a positive integer if provided.")


def _validate_iso8601(val, name):
    try:
        datetime.datetime.fromisoformat(val.replace("Z", "+00:00"))
    except Exception:
        raise ValueError(f"{name} must be a valid ISO 8601 date string.")


def is_persistent_client(client) -> bool:
    return client.__class__.__name__ == "PersistentClient"


##### Query and Listing Tools #####


@mcp.tool()
async def get_history_info() -> Dict:
    """Get information about conversation history. Use this to inspect the size and sample contents of all conversations before querying in detail."""
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection("conversation_history")
        count = collection.count()
        peek_results = collection.peek(limit=3)
        # Remove embeddings from peek_results
        if isinstance(peek_results, list):
            cleaned_peek = [
                {k: v for k, v in doc.items() if k != "embeddings"}
                if isinstance(doc, dict)
                else doc
                for doc in peek_results
            ]
        elif isinstance(peek_results, dict):
            cleaned_peek = [
                {k: v for k, v in peek_results.items() if k != "embeddings"}
            ]
        else:
            cleaned_peek = [peek_results]
        return {"count": count, "sample_documents": cleaned_peek}
    except Exception as e:
        raise Exception(
            f"Failed to get memory info for '{memory_name}': {str(e)}"
        ) from e


@mcp.tool()
async def query_conversation_history(
    query_texts: List[str],
    n_results: int = 10,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: List[str] = ["documents", "metadatas"],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """Query conversation history with advanced filtering. This tool is ideal for retrieving relevant documents from a specific memory based on query text and optional filters. Querying by text performs a semantic search using vector embeddings.

    Args:
        query_texts: List of query strings to search for semantically relevant documents. Each string is embedded and compared to the stored documents to find the most similar ones. This enables natural language search, not just keyword matching.
            Examples:
                - ["What articles did I read about AI last week?"]
                - ["github.com", "python web scraping"]
                - ["meeting notes from yesterday"]
        n_results: Number of top results to return for each query text.
        where: (Optional) Metadata filter for advanced filtering (e.g., {"source": "wikipedia"}).
        where_document: (Optional) Document content filter for advanced filtering.
        include: List of fields to include in the result (e.g., ["documents", "metadatas"]).
        start_date: (Optional) Only include documents created after this ISO 8601 date string.
        end_date: (Optional) Only include documents created before this ISO 8601 date string.

    The query_texts parameter is used to specify the search intent in natural language or keywords. The system will return the most semantically similar documents from the memory for each query text provided.
    If start_date and/or end_date are provided, results will be filtered to only include documents within the specified date range (using the 'created_at' metadata field).
    """
    _validate_non_empty_list(query_texts, "query_texts")
    _validate_positive_int(n_results, "n_results")
    if start_date:
        _validate_iso8601(start_date, "start_date")
    if end_date:
        _validate_iso8601(end_date, "end_date")
    client = get_chroma_client()
    try:
        collection = client.get_collection("conversation_history")
        # Build date range filter if needed
        where_combined = where.copy() if where else {}
        if start_date or end_date:
            created_at_filter = {}
            if start_date:
                created_at_filter["$gte"] = datetime.datetime.fromisoformat(
                    start_date.replace("Z", "+00:00")
                ).timestamp()
            if end_date:
                created_at_filter["$lte"] = datetime.datetime.fromisoformat(
                    end_date.replace("Z", "+00:00")
                ).timestamp()
            where_combined["created_at"] = created_at_filter
        # Only pass where if it is not empty, otherwise use None
        where_arg = where_combined if where_combined else None
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where_arg,
            where_document=where_document,
            include=include,
        )
    except Exception as e:
        raise Exception(
            f"Failed to query documents from conversation history: {str(e)}"
        ) from e


@mcp.tool()
async def health_check() -> dict:
    """Check if the server can import chromadb and connect to the database. Useful for monitoring and debugging deployments. Also performs a round-trip test: add, retrieve, and delete a test document in a test collection. Ensures the client is persistent."""
    result = {
        "chromadb_imported": False,
        "db_connection": False,
        "round_trip": False,
        "persistent_client": False,
    }
    try:
        import chromadb

        result["chromadb_imported"] = True
    except Exception as e:
        result["chromadb_imported"] = False
        result["import_error"] = f"chromadb import failed: {str(e)}"
        return result
    try:
        client = get_chroma_client()
        # Check if client is persistent
        if client.__class__.__name__ == "PersistentClient":
            result["persistent_client"] = True
        else:
            result["persistent_client"] = False
            result["persistent_client_error"] = "Chroma client is not persistent."
            return result
        client.list_collections(limit=1)
        result["db_connection"] = True
    except Exception as e:
        result["db_connection"] = False
        result["db_connection_error"] = f"DB connection failed: {str(e)}"
        return result
    # Round-trip test
    try:
        test_collection_name = "health_check_test"
        test_doc = "health check test document"
        test_id = "health_check_test_id"
        test_metadata = {"created_at": datetime.datetime.now().timestamp()}
        # Create collection and add document
        collection = client.get_or_create_collection(test_collection_name)
        collection.add(documents=[test_doc], metadatas=[test_metadata], ids=[test_id])
        # Retrieve document
        retrieved = collection.get(ids=[test_id])
        docs = retrieved.get("documents", [])
        ids = retrieved.get("ids", [])
        if docs and ids and docs[0] == test_doc and ids[0] == test_id:
            result["round_trip"] = True
        else:
            result["round_trip"] = False
            result["round_trip_error"] = (
                f"Document round-trip failed: got docs={docs}, ids={ids}"
            )
        # Clean up: delete the test collection
        client.delete_collection(test_collection_name)
    except Exception as e:
        result["round_trip"] = False
        result["round_trip_error"] = f"Round-trip test failed: {str(e)}"
    return result


@mcp.tool()
async def remember_this_conversation(
    conversation_content: str,
) -> str:
    """Remember current conversation by storing it in the database so that it can be retrieved later.

    Args:
        conversation_content: Content of the conversation, either summarized or raw.

    Example queries that should trigger this tool:
        - "Memorize this conversation"
        - "Store our current chat"
        - "Remember this chat"
        - "Save our discussion"
        - "Log this conversation"
        - "Archive this exchange"
        - "Store this session for later"
    """
    client = get_chroma_client()
    try:
        created_at_epoch = datetime.datetime.now().timestamp()
        collection = client.get_or_create_collection("conversation_history")
        doc_id = f"conversation_history_{created_at_epoch}"
        # Store created_at as epoch
        collection.add(
            documents=[conversation_content],
            metadatas=[{"created_at": created_at_epoch}],
            ids=[doc_id],
        )
        return f"Successfully memorized 1 conversation in history"
    except Exception as e:
        raise Exception(f"Failed to memorize conversation in history: {str(e)}") from e


@mcp.tool()
async def delete_all_large_entries() -> str:
    """Delete the oldest n_entries from the conversation history."""
    try:
        return remove_large_documents_from_history(30 * 1024)
    except Exception as e:
        raise Exception(
            f"Failed to delete all large entries from conversation history: {str(e)}"
        ) from e


def _sort_and_limit_docs(docs: dict, n_results: int) -> dict:
    if (
        isinstance(docs, dict)
        and "metadatas" in docs
        and isinstance(docs["metadatas"], list)
    ):
        metadatas = docs["metadatas"]
        if all(isinstance(md, dict) and "created_at" in md for md in metadatas):
            sort_indices = sorted(
                range(len(metadatas)),
                key=lambda i: metadatas[i]["created_at"],
                reverse=True,
            )
            for k in docs:
                if isinstance(docs[k], list) and len(docs[k]) == len(sort_indices):
                    docs[k] = [docs[k][i] for i in sort_indices]
            for k in docs:
                if isinstance(docs[k], list):
                    docs[k] = docs[k][:n_results]
    return docs


@mcp.tool()
async def recall_recent_conversations(
    n_results: int = 1,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> dict:
    """Recall recent browser history and chats by retrieving them from the database.

    Args:
        n_results: Number of recent conversations to recall.
        start_time: (Optional) Only recall conversations after this ISO 8601 time.
        end_time: (Optional) Only recall conversations before this ISO 8601 time.

    If no time range is specified, the tool will automatically expand the search window: it will first try since yesterday, then since a week ago, then a month ago, then a year ago, until results are found or all windows are exhausted.

    Returns:
        A dict of recalled browser history, sorted by most recent first (may be empty).
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection("conversation_history")
        if not collection:
            return {"error": f"No memory found for 'conversation_history'"}
        now = datetime.datetime.now(datetime.timezone.utc)
        # If time range is specified, use it
        if start_time or end_time:
            start_epoch = None
            end_epoch = None
            if start_time:
                _validate_iso8601(start_time, "start_time")
                start_epoch = datetime.datetime.fromisoformat(
                    start_time.replace("Z", "+00:00")
                ).timestamp()
            if end_time:
                _validate_iso8601(end_time, "end_time")
                end_epoch = datetime.datetime.fromisoformat(
                    end_time.replace("Z", "+00:00")
                ).timestamp()
            where = {
                "$and": [
                    {"created_at": {"$gte": start_epoch}},
                    {"created_at": {"$lte": end_epoch}},
                ]
            }
            docs = collection.get(where=where)
            return _sort_and_limit_docs(docs, n_results)
        # Exponential backoff: 1 day, 1 week, 1 month, 1 year
        windows = [
            datetime.timedelta(days=1),
            datetime.timedelta(weeks=1),
            datetime.timedelta(days=30),
            datetime.timedelta(days=365),
        ]
        last_docs = None
        for window in windows:
            start_dt = now - window
            start_epoch = start_dt.timestamp()
            where = {"created_at": {"$gte": start_epoch}}
            docs = collection.get(where=where)
            last_docs = docs
            docs_sorted = _sort_and_limit_docs(docs, n_results)
            if docs_sorted.get("documents"):
                return docs_sorted
        # If all windows are empty, return the last (widest) result, limited
        return _sort_and_limit_docs(
            last_docs if last_docs is not None else {}, n_results
        )
    except Exception as e:
        return {
            "error": f"Failed to recall recent conversations from conversation history: {str(e)}"
        }


def remove_large_documents_from_history(
    max_size_bytes: int = 100 * 1024,
) -> str:
    """
    Go through conversation history and remove all documents larger than max_size_bytes.
    """
    client = get_chroma_client()
    collection = client.get_collection("conversation_history")
    count = 0
    total_deleted = 0
    offset = 0
    batch_size = 100
    while True:
        batch = collection.get(include=["documents"], limit=batch_size, offset=offset)
        docs = batch.get("documents", [])
        ids = batch.get("ids", [])
        if not docs or not ids:
            break
        to_delete = []
        for doc, doc_id in zip(docs, ids):
            if (
                doc
                and isinstance(doc, str)
                and len(doc.encode("utf-8")) > max_size_bytes
            ):
                to_delete.append(doc_id)
        if to_delete:
            collection.delete(ids=to_delete)
            total_deleted += len(to_delete)
            count += total_deleted
        if len(docs) < batch_size:
            break
        offset += batch_size
    return f"Total deleted from conversation history: {count}"


def main():
    """Entry point for the Chroma MCP server."""
    parser = create_parser()
    args = parser.parse_args()
    if args.dotenv_path:
        load_dotenv(dotenv_path=args.dotenv_path)
        parser = create_parser()
        args = parser.parse_args()
    # Validate required arguments based on client type
    if args.client_type == "http":
        if not args.host:
            parser.error(
                "Host must be provided via --host flag or CHROMA_HOST environment variable when using HTTP client"
            )
    elif args.client_type == "cloud":
        if not args.tenant:
            parser.error(
                "Tenant must be provided via --tenant flag or CHROMA_TENANT environment variable when using cloud client"
            )
        if not args.database:
            parser.error(
                "Database must be provided via --database flag or CHROMA_DATABASE environment variable when using cloud client"
            )
        if not args.api_key:
            parser.error(
                "API key must be provided via --api-key flag or CHROMA_API_KEY environment variable when using cloud client"
            )
    # Initialize client with parsed args
    try:
        get_chroma_client(args)
        print("Successfully initialized Chroma client")
    except Exception as e:
        print(f"Failed to initialize Chroma client: {str(e)}")
        raise
    # Initialize and run the server
    print("Starting MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
