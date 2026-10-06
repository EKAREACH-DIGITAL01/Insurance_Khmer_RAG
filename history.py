"""Compatibility imports. New code should use khmer_rag.storage."""

from khmer_rag.storage import (
    DB_PATH,
    TITLE_MAX_CHARS,
    append_message,
    create_conversation,
    delete_conversation,
    derive_title,
    get_db,
    init_db,
    list_conversations,
    load_message,
    load_messages,
    record_feedback,
    rename_conversation,
)

__all__ = [
    "DB_PATH",
    "TITLE_MAX_CHARS",
    "get_db",
    "init_db",
    "derive_title",
    "list_conversations",
    "create_conversation",
    "rename_conversation",
    "delete_conversation",
    "load_messages",
    "load_message",
    "append_message",
    "record_feedback",
]
