from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Any

import aiosqlite
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore, SearchItem
from langgraph.store.sqlite.aio import AsyncSqliteStore
from typing_extensions import NotRequired, Required

from app.config import settings

logger = logging.getLogger(__name__)


def _patch_typing_for_langmem() -> None:
    """Allow current LangMem releases to import cleanly on Python 3.10."""
    if not hasattr(typing, "NotRequired"):
        typing.NotRequired = NotRequired  # type: ignore[attr-defined]
    if not hasattr(typing, "Required"):
        typing.Required = Required  # type: ignore[attr-defined]


_patch_typing_for_langmem()

try:
    from langmem import create_memory_store_manager  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised when optional dep is absent
    class MemoryStoreManager:
        def __init__(self, *args: Any, **kwargs: Any):
            self.args = args
            self.kwargs = kwargs

        async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None):
            logger.warning("langmem is not installed; skipping memory extraction")
            return None

    def create_memory_store_manager(*args: Any, **kwargs: Any) -> MemoryStoreManager:
        return MemoryStoreManager(*args, **kwargs)


MEMORY_NAMESPACE_PREFIX = "memories"


async def build_memory_store() -> AsyncSqliteStore:
    path = _memory_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(path, isolation_level=None)
    store = AsyncSqliteStore(conn)
    await store.setup()
    return store


def build_memory_manager(store: BaseStore):
    """Build the LangMem manager.

    LangMem only does text extraction, so it always hits an OpenAI-compatible
    chat-completions endpoint. By default we reuse the VLM provider (Doubao
    and MiMo both expose /chat/completions); override LANGMEM_API_* to point at
    a cheaper text-only model.
    """
    model_name = settings.langmem_model_name or settings.vlm_model_name
    base_url = (
        settings.langmem_api_base_url
        or settings.vlm_api_base_url
    ).rstrip("/")
    api_key = settings.langmem_api_key or settings.vlm_api_key or "EMPTY"
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout=settings.vlm_api_timeout,
        temperature=0,
        max_retries=1,
    )
    return create_memory_store_manager(
        model,
        namespace=(MEMORY_NAMESPACE_PREFIX, "{langgraph_user_id}"),
        store=store,
        query_limit=settings.langmem_query_limit,
    )


async def memory_context(
    store: BaseStore,
    user_id: str,
    query: str | None = None,
    limit: int | None = None,
) -> str:
    items = await search_memories(store, user_id, query, limit)
    lines = [_format_memory_item(item) for item in items]
    return "\n".join(f"- {line}" for line in lines if line)


async def search_memories(
    store: BaseStore,
    user_id: str,
    query: str | None = None,
    limit: int | None = None,
) -> list[SearchItem]:
    namespace = (MEMORY_NAMESPACE_PREFIX, user_id)
    actual_limit = max(1, limit or settings.langmem_query_limit)
    try:
        return await store.asearch(namespace, query=query or None, limit=actual_limit)
    except Exception:
        logger.exception("LangMem search failed; retrying without query")
        return await store.asearch(namespace, limit=actual_limit)


async def write_memories(
    manager: Any,
    messages: list[AnyMessage],
    user_id: str,
) -> None:
    if not messages:
        return
    config = {"configurable": {"langgraph_user_id": user_id}}
    try:
        await manager.ainvoke({"messages": messages}, config=config)
    except Exception:
        logger.exception("LangMem memory write failed")


def _memory_store_path() -> Path:
    return settings.langmem_store_path or (settings.data_dir / "langmem_store.sqlite3")


def _format_memory_item(item: SearchItem) -> str:
    value = item.value
    content = value.get("content", value)
    if isinstance(content, dict):
        for key in ("content", "memory", "text", "data"):
            nested = content.get(key)
            if nested:
                return str(nested)
        return ", ".join(f"{key}: {value}" for key, value in content.items())
    return str(content)
