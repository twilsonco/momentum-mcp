"""
Memory Module — Sam's Always-Learning Brain.

Three ChromaDB collections that grow over time:
  1. conversation_memory — stores user interactions, topics, and preferences
  2. technical_snapshots — stores ticker analysis results for learning/backtesting  
  3. market_context — stores market regime snapshots for pattern recognition

Sam gets smarter with every conversation.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any

from mcp_server.knowledge import _get_chroma, _embed_texts, _embed_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

def _get_conversation_collection():
    return _get_chroma().get_or_create_collection(
        name="conversation_memory",
        metadata={"hnsw:space": "cosine"},
    )


def _get_technical_collection():
    return _get_chroma().get_or_create_collection(
        name="technical_snapshots",
        metadata={"hnsw:space": "cosine"},
    )


def _get_market_collection():
    return _get_chroma().get_or_create_collection(
        name="market_context",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# 1. Conversation Memory — remember what users care about
# ---------------------------------------------------------------------------

def remember_conversation(
    user_message: str,
    assistant_response: str,
    tickers_mentioned: list[str] | None = None,
    tools_used: list[str] | None = None,
) -> None:
    """Store a conversation turn for future context recall."""
    try:
        collection = _get_conversation_collection()
        now = datetime.now(timezone.utc)
        
        # Create a summary of the interaction
        summary = f"User asked: {user_message[:200]}\nSam responded about: {assistant_response[:300]}"
        if tickers_mentioned:
            summary += f"\nTickers: {', '.join(tickers_mentioned)}"
        
        chunk_id = hashlib.md5(
            f"conv:{now.isoformat()}:{user_message[:50]}".encode()
        ).hexdigest()
        
        embedding = _embed_texts([summary])[0]
        
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[{
                "timestamp": now.isoformat(),
                "tickers": ",".join(tickers_mentioned or []),
                "tools": ",".join(tools_used or []),
                "type": "conversation",
            }],
        )
        logger.debug("Memory: stored conversation turn (%d chars)", len(summary))
    except Exception as e:
        logger.warning("Memory store error: %s", e)


def recall_relevant_context(query: str, top_k: int = 3) -> list[dict]:
    """Recall past conversations relevant to current query."""
    try:
        collection = _get_conversation_collection()
        if collection.count() == 0:
            return []
        
        query_embedding = _embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        
        memories = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                score = 1 - results["distances"][0][i]
                if score < 0.4:  # only recall genuinely relevant memories
                    continue
                memories.append({
                    "text": results["documents"][0][i],
                    "timestamp": results["metadatas"][0][i].get("timestamp", ""),
                    "tickers": results["metadatas"][0][i].get("tickers", ""),
                    "relevance": round(score, 3),
                })
        return memories
    except Exception as e:
        logger.warning("Memory recall error: %s", e)
        return []


# ---------------------------------------------------------------------------
# 2. Technical Snapshots — learn from every analysis
# ---------------------------------------------------------------------------

def store_technical_snapshot(
    ticker: str,
    data: dict[str, Any],
    source: str = "analyze_technicals",
) -> None:
    """Store a technical analysis snapshot for learning/backtesting.
    
    Stores ALL available indicator data — the full EMA stack, SMAs,
    oscillators, trend indicators, and TV consensus.
    """
    try:
        collection = _get_technical_collection()
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        
        # Extract values robustly (handle nested dicts from old format)
        def _val(key: str) -> str:
            v = data.get(key, None)
            if isinstance(v, dict):
                return str(v.get("value", v))
            return str(v) if v is not None else "?"
        
        price = _val("close") if _val("close") != "?" else _val("latest_close")
        rsi = _val("rsi_14") if _val("rsi_14") != "?" else _val("rsi")
        
        # Build a comprehensive searchable text
        text_parts = [
            f"{ticker} technical snapshot ({date_str}): Price=${price}",
        ]
        
        # Core oscillators
        if _val("rsi_14") != "?":
            text_parts.append(f"RSI(14)={_val('rsi_14')}")
        if _val("macd") != "?":
            text_parts.append(f"MACD={_val('macd')} Signal={_val('macd_signal')} Hist={_val('macd_histogram')}")
        
        # EMA Stack (8/21/34/55/89)
        ema_vals = {k: _val(k) for k in ["ema_8", "ema_21", "ema_34", "ema_55", "ema_89"] if _val(k) != "?"}
        if ema_vals:
            text_parts.append(f"EMA Stack: {', '.join(f'{k}={v}' for k, v in ema_vals.items())}")
        
        # SMAs
        sma_vals = {k: _val(k) for k in ["sma_50", "sma_100", "sma_200"] if _val(k) != "?"}
        if sma_vals:
            text_parts.append(f"SMAs: {', '.join(f'{k}={v}' for k, v in sma_vals.items())}")
        
        # Trend & Volatility
        for key, label in [("adx_14", "ADX"), ("atr_14", "ATR"), ("williams_r_14", "Williams%R"),
                           ("cci_20", "CCI"), ("stoch_k", "Stoch%K"), ("stoch_d", "Stoch%D")]:
            if _val(key) != "?":
                text_parts.append(f"{label}={_val(key)}")
        
        # Bollinger Bands
        if _val("bb_upper") != "?":
            text_parts.append(f"BB: {_val('bb_lower')}-{_val('bb_mid')}-{_val('bb_upper')}")
        
        # TV Analysis consensus
        if "recommendation" in data:
            text_parts.append(f"TV Consensus={data['recommendation']}")
            if "summary" in data:
                s = data["summary"]
                text_parts.append(f"(Buy:{s.get('buy',0)} Sell:{s.get('sell',0)} Neutral:{s.get('neutral',0)})")
        
        text = ", ".join(text_parts) + f". Source: {source}"
        
        chunk_id = hashlib.md5(f"tech:{ticker}:{date_str}:{source}".encode()).hexdigest()
        embedding = _embed_texts([text])[0]
        
        # Store ALL numeric values in metadata for filtering/backtesting
        metadata: dict[str, str] = {
            "ticker": ticker,
            "date": date_str,
            "timestamp": now.isoformat(),
            "source": source,
            "type": "technical_snapshot",
            "price": str(price),
        }
        
        # Store every numeric field we have
        numeric_keys = [
            "rsi_14", "macd", "macd_signal", "macd_histogram",
            "ema_8", "ema_21", "ema_34", "ema_55", "ema_89",
            "sma_50", "sma_100", "sma_200",
            "adx_14", "atr_14", "williams_r_14",
            "stoch_k", "stoch_d", "cci_20",
            "bb_upper", "bb_mid", "bb_lower",
        ]
        for key in numeric_keys:
            val = data.get(key)
            if val is not None:
                metadata[key] = str(val)
        
        # TV consensus fields
        if "recommendation" in data:
            metadata["tv_recommendation"] = str(data["recommendation"])
        
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        logger.debug("Memory: stored technical snapshot for %s (%d fields)", ticker, len(metadata))
    except Exception as e:
        logger.warning("Technical snapshot store error: %s", e)


def get_ticker_history(ticker: str, top_k: int = 10) -> list[dict]:
    """Get historical technical snapshots for a ticker."""
    try:
        collection = _get_technical_collection()
        if collection.count() == 0:
            return []
        
        # Search by ticker name
        query_embedding = _embed_query(f"{ticker} technical analysis")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
            where={"ticker": ticker},
        )
        
        history = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                history.append({
                    "text": results["documents"][0][i],
                    "date": results["metadatas"][0][i].get("date", ""),
                    "rsi": results["metadatas"][0][i].get("rsi", ""),
                    "price": results["metadatas"][0][i].get("price", ""),
                })
        return history
    except Exception as e:
        logger.warning("Ticker history error: %s", e)
        return []


# ---------------------------------------------------------------------------
# 3. Market Context — remember market conditions
# ---------------------------------------------------------------------------

def store_market_snapshot(data: dict[str, Any], source: str = "market_pulse") -> None:
    """Store a market condition snapshot for pattern recognition."""
    try:
        collection = _get_market_collection()
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d %H:%M")
        
        # Build text summary from market data
        text_parts = [f"Market snapshot ({date_str}):"]
        
        if "summary" in data:
            text_parts.append(f"Pulse: {data['summary'][:200]}")
        if "flow_score" in data:
            text_parts.append(f"Flow Score: {data['flow_score']}")
        if "sentiment" in data:
            text_parts.append(f"Sentiment: {data['sentiment']}")
        
        text = " | ".join(text_parts)
        
        chunk_id = hashlib.md5(f"market:{date_str}:{source}".encode()).hexdigest()
        embedding = _embed_texts([text])[0]
        
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "date": date_str,
                "timestamp": now.isoformat(),
                "source": source,
                "type": "market_snapshot",
            }],
        )
        logger.debug("Memory: stored market snapshot")
    except Exception as e:
        logger.warning("Market snapshot store error: %s", e)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_memory_stats() -> dict:
    """Get stats for all memory collections."""
    try:
        return {
            "conversations": _get_conversation_collection().count(),
            "technical_snapshots": _get_technical_collection().count(),
            "market_snapshots": _get_market_collection().count(),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Memory Consolidation — prune old entries, keep recent + high-value
# ---------------------------------------------------------------------------

def consolidate_memories(
    retention_days: int = 7,
    max_per_collection: int = 500,
) -> dict[str, int]:
    """
    Prune old memories to prevent unbounded growth.
    
    Strategy:
      - Delete entries older than retention_days
      - If still over max_per_collection, keep the most recent ones
      - Technical snapshots are kept longer (30 days) since they're used for backtesting
    
    Returns dict of {collection: entries_pruned}
    """
    from datetime import timedelta
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    tech_cutoff = datetime.now(timezone.utc) - timedelta(days=30)  # longer retention for technicals
    cutoff_iso = cutoff.isoformat()
    tech_cutoff_iso = tech_cutoff.isoformat()
    
    pruned = {}
    
    for collection_fn, name, older_than in [
        (_get_conversation_collection, "conversations", cutoff_iso),
        (_get_technical_collection, "technical_snapshots", tech_cutoff_iso),
        (_get_market_collection, "market_snapshots", cutoff_iso),
    ]:
        try:
            collection = collection_fn()
            count_before = collection.count()
            
            if count_before == 0:
                pruned[name] = 0
                continue
            
            # Get all entries to check timestamps
            all_data = collection.get(
                include=["metadatas"],
                limit=count_before,
            )
            
            if not all_data["ids"]:
                pruned[name] = 0
                continue
            
            # Find IDs to delete (older than cutoff)
            ids_to_delete = []
            entries_with_time = []
            
            for i, doc_id in enumerate(all_data["ids"]):
                ts = all_data["metadatas"][i].get("timestamp", "")
                if ts and ts < older_than:
                    ids_to_delete.append(doc_id)
                else:
                    entries_with_time.append((doc_id, ts))
            
            # If still too many after time-based pruning, trim the oldest
            remaining_after = len(entries_with_time)
            if remaining_after > max_per_collection:
                entries_with_time.sort(key=lambda x: x[1])
                excess = remaining_after - max_per_collection
                ids_to_delete.extend([eid for eid, _ in entries_with_time[:excess]])
            
            # Batch delete
            if ids_to_delete:
                # ChromaDB delete in batches of 100
                for batch_start in range(0, len(ids_to_delete), 100):
                    batch = ids_to_delete[batch_start:batch_start + 100]
                    collection.delete(ids=batch)
            
            pruned[name] = len(ids_to_delete)
            logger.info(
                "Memory consolidation: %s — pruned %d/%d entries",
                name, len(ids_to_delete), count_before,
            )
        except Exception as e:
            logger.warning("Consolidation error for %s: %s", name, e)
            pruned[name] = -1
    
    return pruned


# ---------------------------------------------------------------------------
# Training Data Export — for future fine-tuning
# ---------------------------------------------------------------------------

def export_training_data(output_path: str = "data/training_export.jsonl") -> int:
    """
    Export conversation memory as JSONL for fine-tuning.
    
    Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Returns number of examples exported.
    """
    import json
    
    try:
        collection = _get_conversation_collection()
        if collection.count() == 0:
            return 0
        
        all_data = collection.get(
            include=["documents", "metadatas"],
            limit=collection.count(),
        )
        
        examples = []
        for i, doc in enumerate(all_data["documents"]):
            # Parse the stored format: "User asked: X\nSam responded about: Y"
            parts = doc.split("\nSam responded about: ")
            if len(parts) == 2:
                user_part = parts[0].replace("User asked: ", "")
                assistant_part = parts[1].split("\nTickers:")[0]  # strip metadata
                examples.append({
                    "messages": [
                        {"role": "user", "content": user_part},
                        {"role": "assistant", "content": assistant_part},
                    ]
                })
        
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        
        logger.info("Exported %d training examples to %s", len(examples), output_path)
        return len(examples)
    except Exception as e:
        logger.warning("Training export error: %s", e)
        return 0


# ---------------------------------------------------------------------------
# 4. Persistent User Notes — "remember I'm short TSLA at $180"
# ---------------------------------------------------------------------------

def _get_notes_collection():
    return _get_chroma().get_or_create_collection(
        name="user_notes",
        metadata={"hnsw:space": "cosine"},
    )


def save_user_note(note: str) -> dict:
    """Save a persistent user note for cross-session recall.

    Args:
        note: Free-text note (e.g. "I'm short TSLA at $180, stop at $190")

    Returns:
        Confirmation dict.
    """
    try:
        collection = _get_notes_collection()
        now = datetime.now(timezone.utc)

        chunk_id = hashlib.md5(
            f"note:{now.isoformat()}:{note[:50]}".encode()
        ).hexdigest()

        embedding = _embed_texts([note])[0]

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[note],
            metadatas=[{
                "timestamp": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "type": "user_note",
            }],
        )
        logger.info("User note saved: %s", note[:80])
        return {
            "status": "saved",
            "note": note,
            "message": f"Got it — I'll remember that. ({collection.count()} notes stored)",
        }
    except Exception as e:
        logger.warning("Note save error: %s", e)
        return {"error": str(e)}


def recall_user_notes(query: str, top_k: int = 5) -> list[dict]:
    """Recall user notes relevant to a query.

    Args:
        query: Search text (user's current message)
        top_k: Max notes to return

    Returns:
        List of {text, timestamp, relevance} dicts.
    """
    try:
        collection = _get_notes_collection()
        if collection.count() == 0:
            return []

        query_embedding = _embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        notes = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                score = 1 - results["distances"][0][i]
                if score < 0.35:  # only recall genuinely relevant notes
                    continue
                notes.append({
                    "text": results["documents"][0][i],
                    "timestamp": results["metadatas"][0][i].get("timestamp", ""),
                    "relevance": round(score, 3),
                })
        return notes
    except Exception as e:
        logger.warning("Note recall error: %s", e)
        return []


def delete_user_note(note_text: str) -> dict:
    """Delete a user note by approximate text match."""
    try:
        collection = _get_notes_collection()
        if collection.count() == 0:
            return {"status": "empty", "message": "No notes to delete."}

        # Find the closest match
        embedding = _embed_query(note_text)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "distances"],
        )

        if results["ids"] and results["ids"][0]:
            score = 1 - results["distances"][0][0]
            if score > 0.5:
                deleted_text = results["documents"][0][0]
                collection.delete(ids=[results["ids"][0][0]])
                return {
                    "status": "deleted",
                    "note": deleted_text,
                    "message": f"Deleted note: \"{deleted_text[:80]}...\"",
                }
        return {"status": "not_found", "message": "No matching note found."}
    except Exception as e:
        logger.warning("Note delete error: %s", e)
        return {"error": str(e)}

