import asyncio
import typing

from app.memory import build_memory_manager, memory_context


def test_langmem_import_patches_python_310_typing():
    assert hasattr(typing, "NotRequired")


def test_memory_context_reads_langgraph_store():
    from langgraph.store.memory import InMemoryStore

    async def _run():
        store = InMemoryStore()
        store.put(
            ("memories", "user-1"),
            "pref",
            {
                "kind": "Memory",
                "content": {"content": "User prefers concise Chinese answers."},
            },
        )

        context = await memory_context(store, "user-1", "Chinese", 3)

        assert "User prefers concise Chinese answers." in context
        assert build_memory_manager(store).__class__.__name__ == "MemoryStoreManager"

    asyncio.run(_run())
