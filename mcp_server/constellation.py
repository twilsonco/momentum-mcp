"""
constellation.py — Parallel AI Board Orchestration for momentum-chat.

The Constellation is a permanent, reusable multi-model synthesis loop that acts as
a Chief Orchestrator: it fans out a prompt to multiple LLM "Board Members" in parallel,
synthesizes their responses, optionally red-teams the synthesis, and returns a
unified, high-confidence answer.

Architecture:
  Phase 1 — Context: Gather relevant codebase / tool context
  Phase 2 — Fan-Out: Send prompt + context to ALL board members concurrently
  Phase 3 — Synthesis: Orchestrator reads all responses, extracts best elements
  Phase 4 — Red-Team: Board critiques the synthesis for flaws/gaps (optional)
  Phase 5 — Final: Lock in the synthesized answer

Board Members (configured via .env):
  With OPENROUTER_API_KEY: Claude, GPT-4o, Grok, DeepSeek, Gemini (full board)
  Without (default): Multiple Gemini model variants (reduced board)

Usage as MCP tool:
  Sam calls `run_constellation(question, mode)` to invoke a Board decision.

Usage as standalone script:
  python -m mcp_server.constellation --question "..." --mode full
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from dotenv import load_dotenv

from mcp_server.schema import SignalResult

load_dotenv()
logger = logging.getLogger(__name__)

# ── Board Member Definitions ──────────────────────────────────────────────────

@dataclass
class BoardMember:
    name: str           # Human-readable label (e.g. "Claude Opus")
    model_id: str       # Provider-specific model ID
    provider: str       # "openrouter" | "gemini"
    role: str           # Role description for self-awareness in prompts
    temperature: float = 0.4
    max_tokens: int = 2048


# Full board requires OPENROUTER_API_KEY
FULL_BOARD: list[BoardMember] = [
    BoardMember(
        name="Claude Sonnet (Rigorous Analyst)",
        model_id="anthropic/claude-sonnet-4.6",
        provider="openrouter",
        role="You are a rigorous, detail-oriented analyst. Prioritize correctness, edge-case coverage, and type safety. Flag any potential runtime errors or race conditions. Be brutally honest about weaknesses.",
        temperature=0.3,
    ),
    BoardMember(
        name="GPT-4o (Clean Coder)",
        model_id="openai/gpt-4o",
        provider="openrouter",
        role="You are an expert software engineer focused on clean, idiomatic Python. Prioritize readability, DRY principles, and elegant structure. Suggest the most Pythonic solutions.",
        temperature=0.4,
    ),
    BoardMember(
        name="Grok 3 (Contrarian)",
        model_id="x-ai/grok-3",
        provider="openrouter",
        role="You are a contrarian red-team analyst. Your job is to find what everyone else missed. Aggressively identify hidden assumptions, bottlenecks, logic flaws, and failure modes. Don't just agree with obvious solutions.",
        temperature=0.6,
    ),
    BoardMember(
        name="DeepSeek R1 (Systems Thinker)",
        model_id="deepseek/deepseek-r1",
        provider="openrouter",
        role="You are a systems architect focused on performance, scalability, and long-term maintainability. Think about async patterns, concurrency safety, and downstream effects of code changes.",
        temperature=0.3,
    ),
    BoardMember(
        name="Qwen 2.5 (Speed Scout)",
        model_id="qwen/qwen-2.5-72b-instruct",
        provider="openrouter",
        role="You are a fast pragmatist. Identify the highest-impact, lowest-effort improvements. Cut through complexity to the essential action items.",
        temperature=0.5,
    ),
]

# Fallback board — only needs GEMINI_API_KEY (always available)
GEMINI_BOARD: list[BoardMember] = [
    BoardMember(
        name="Gemini 2.5 Pro (Lead Analyst)",
        model_id="gemini/gemini-2.5-pro-preview-05-06",
        provider="gemini",
        role="You are a rigorous lead analyst and architect. Prioritize correctness, edge cases, type safety, and async concurrency patterns. Be specific and unambiguous.",
        temperature=0.3,
    ),
    BoardMember(
        name="Gemini 2.5 Flash (Speed Scout)",
        model_id="gemini/gemini-2.5-flash",
        provider="gemini",
        role="You are a pragmatic speed-optimizer. Identify the highest-impact, lowest-complexity improvements. Ruthlessly prioritize what actually matters.",
        temperature=0.5,
    ),
    BoardMember(
        name="Gemini 2.0 Flash (Contrarian)",
        model_id="gemini/gemini-2.0-flash",
        provider="gemini",
        role="You are a contrarian red-team reviewer. Find what others missed: hidden assumptions, race conditions, silent failures, untested edge cases. Don't hold back.",
        temperature=0.6,
    ),
]

# ── LLM API Calls ─────────────────────────────────────────────────────────────

async def _call_openrouter(member: BoardMember, messages: list[dict], api_key: str) -> str:
    """Call a board member via OpenRouter API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://ghost.mphinance.com",
                "X-Title": "momentum-chat-constellation",
                "Content-Type": "application/json",
            },
            json={
                "model": member.model_id,
                "messages": messages,
                "temperature": member.temperature,
                "max_tokens": member.max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _call_gemini(member: BoardMember, messages: list[dict]) -> str:
    """Call a board member via LiteLLM (Gemini native)."""
    from litellm import acompletion
    api_key = os.getenv("GEMINI_API_KEY", "")
    resp = await acompletion(
        model=member.model_id,
        messages=messages,
        temperature=member.temperature,
        max_tokens=member.max_tokens,
        api_key=api_key,
    )
    return resp.choices[0].message.content


async def _query_member(member: BoardMember, messages: list[dict], api_key_or: str | None) -> dict:
    """Query a single board member and return structured result."""
    start = time.perf_counter()
    try:
        if member.provider == "openrouter" and api_key_or:
            content = await _call_openrouter(member, messages, api_key_or)
        elif member.provider == "gemini":
            content = await _call_gemini(member, messages)
        else:
            return {
                "member": member.name,
                "status": "skipped",
                "content": "No API key configured for this provider.",
                "latency_ms": 0,
            }
        return {
            "member": member.name,
            "role": member.role,
            "status": "success",
            "content": content,
            "latency_ms": round((time.perf_counter() - start) * 1000),
        }
    except Exception as e:
        logger.warning(f"Constellation member {member.name} failed: {e}")
        return {
            "member": member.name,
            "status": "error",
            "content": f"Error: {e}",
            "latency_ms": round((time.perf_counter() - start) * 1000),
        }


# ── Orchestrator Phases ───────────────────────────────────────────────────────

def _build_system_prompt(member_role: str) -> dict:
    return {
        "role": "system",
        "content": (
            f"{member_role}\n\n"
            "You are part of a multi-model AI Board called the Constellation. "
            "The Chief Orchestrator will synthesize all board responses into a final decision. "
            "Be direct, structured, and specific. Use code blocks where relevant. "
            "Avoid padding, filler, or restating the question."
        )
    }


async def _fan_out(
    board: list[BoardMember],
    user_messages: list[dict],
    api_key_or: str | None
) -> list[dict]:
    """Phase 2: Send prompt to all board members concurrently."""
    tasks = [
        _query_member(member, [_build_system_prompt(member.role)] + user_messages, api_key_or)
        for member in board
    ]
    return await asyncio.gather(*tasks)


def _build_synthesis_prompt(question: str, responses: list[dict]) -> list[dict]:
    """Phase 3: Build the orchestrator's synthesis request."""
    board_text = ""
    for r in responses:
        if r["status"] == "success":
            board_text += f"\n\n--- [{r['member']}] ---\n{r['content']}"

    return [
        {
            "role": "system",
            "content": (
                "You are the Chief Orchestrator of the AI Constellation Board. "
                "Your job is to synthesize multiple independent expert analyses into one optimal, unified answer. "
                "Rules:\n"
                "1. Extract the most rigorous edge-case handling from any lane\n"
                "2. Adopt the cleanest code structure from the strongest coder\n"
                "3. Use the best type annotations from the most structured response\n"
                "4. Discard bloated formatting, hallucinations, or contradictory advice\n"
                "5. Resolve disagreements by choosing the most defensible position\n"
                "6. Output a single, clear, actionable synthesis. Use markdown headers.\n"
                "Be the best possible version of all board members combined."
            ),
        },
        {
            "role": "user",
            "content": (
                f"**Original Question:**\n{question}\n\n"
                f"**Board Responses:**{board_text}\n\n"
                "Please synthesize the above into a single optimal answer."
            ),
        }
    ]


def _build_redteam_prompt(question: str, synthesis: str, responses: list[dict]) -> list[dict]:
    """Phase 4: Red-team critique of the synthesis."""
    board_text = ""
    for r in responses:
        if r["status"] == "success":
            board_text += f"\n\n--- [{r['member']}] ---\n{r['content']}"

    return [
        {
            "role": "system",
            "content": (
                "You are an aggressive red-team reviewer on the AI Constellation Board. "
                "You've seen the original question, all board analyses, and the Chief Orchestrator's synthesis. "
                "Your ONLY job is to find flaws. Look for:\n"
                "- Logic errors or incorrect assumptions\n"
                "- Missing edge cases (None values, empty lists, network failures, concurrency bugs)\n"
                "- Performance bottlenecks or inefficient patterns\n"
                "- Security issues or data integrity problems\n"
                "- Things the synthesizer missed or got wrong\n"
                "If the synthesis is correct, say so briefly. Do NOT restate what's right. Focus only on gaps."
            ),
        },
        {
            "role": "user",
            "content": (
                f"**Original Question:**\n{question}\n\n"
                f"**All Board Analyses:**{board_text}\n\n"
                f"**Chief Orchestrator Synthesis:**\n{synthesis}\n\n"
                "What did the synthesis miss, get wrong, or leave vulnerable?"
            ),
        }
    ]


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def run_constellation(
    question: str,
    mode: Literal["full", "gemini_only", "fast"] = "full",
    context: str | None = None,
    red_team: bool = True,
) -> SignalResult:
    """
    Run the full Constellation synthesis loop.

    Args:
        question: The question or task to send to the board.
        mode: "full" uses OpenRouter board if key available, falls back to Gemini.
               "gemini_only" forces the Gemini board regardless.
               "fast" skips red-team phase for speed.
        context: Optional additional context (code snippets, data, etc.) to inject.
        red_team: Whether to run Phase 4 red-team critique.
    
    Returns:
        SignalResult with synthesis, red-team critique, and full board transcript.
    """
    start_total = time.perf_counter()
    api_key_or = os.getenv("OPENROUTER_API_KEY")

    # Select board
    if mode == "gemini_only" or not api_key_or:
        board = GEMINI_BOARD
        board_mode = "Gemini Board (3 models)"
    else:
        board = FULL_BOARD
        board_mode = "Full OpenRouter Board (5 models)"

    if mode == "fast":
        red_team = False

    logger.info(f"Constellation starting | board={board_mode} | red_team={red_team}")

    # Build user messages
    question_with_ctx = question
    if context:
        question_with_ctx = f"**Context:**\n{context}\n\n**Question:**\n{question}"

    user_messages = [{"role": "user", "content": question_with_ctx}]

    # Phase 2: Fan-out
    logger.info("Phase 2: Fan-out to board members...")
    board_responses = await _fan_out(board, user_messages, api_key_or)

    successful = [r for r in board_responses if r["status"] == "success"]
    if not successful:
        return SignalResult.error_msg("All Constellation board members failed to respond.")

    # Phase 3: Synthesis
    logger.info("Phase 3: Synthesizing board responses...")
    synthesis_messages = _build_synthesis_prompt(question, board_responses)

    # Use the lead model for synthesis
    synth_member = GEMINI_BOARD[0] if (mode == "gemini_only" or not api_key_or) else FULL_BOARD[0]
    synth_result = await _query_member(synth_member, synthesis_messages, api_key_or)
    synthesis = synth_result.get("content", "Synthesis failed.")

    # Phase 4: Red-Team (optional)
    redteam_result = None
    if red_team and len(successful) >= 2:
        logger.info("Phase 4: Red-teaming synthesis...")
        rt_messages = _build_redteam_prompt(question, synthesis, board_responses)
        # Use the contrarian model for red-teaming
        rt_member = next(
            (m for m in board if "Contrarian" in m.role or "contrarian" in m.role),
            board[-1]
        )
        rt_result = await _query_member(rt_member, rt_messages, api_key_or)
        redteam_result = rt_result.get("content")

    total_ms = round((time.perf_counter() - start_total) * 1000)
    logger.info(f"Constellation complete in {total_ms}ms")

    return SignalResult.success(
        data={
            "synthesis": synthesis,
            "redteam_critique": redteam_result,
            "board_transcript": board_responses,
            "board_mode": board_mode,
            "total_latency_ms": total_ms,
            "members_responded": len(successful),
            "members_total": len(board),
            "summary": (
                f"Constellation ({board_mode}) completed in {total_ms}ms. "
                f"{len(successful)}/{len(board)} board members responded."
            ),
        },
        metadata={
            "question": question[:200],
            "mode": mode,
            "red_team": red_team,
        }
    )


# ── CLI Entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Constellation AI Board.")
    parser.add_argument("--question", "-q", required=True, help="Question or task for the board.")
    parser.add_argument(
        "--mode", "-m",
        choices=["full", "gemini_only", "fast"],
        default="full",
        help="Board mode: full (OpenRouter+Gemini), gemini_only, or fast (no red-team)."
    )
    parser.add_argument("--context", "-c", default=None, help="Optional context string.")
    parser.add_argument("--no-redteam", action="store_true", help="Skip red-team phase.")
    args = parser.parse_args()

    async def _main():
        result = await run_constellation(
            question=args.question,
            mode=args.mode,
            context=args.context,
            red_team=not args.no_redteam,
        )
        import json
        if result.status == "success":
            d = result.data
            print(f"\n{'='*60}")
            print(f"CONSTELLATION RESULT — {d['board_mode']}")
            print(f"{'='*60}")
            print(f"\n## SYNTHESIS\n{d['synthesis']}")
            if d.get("redteam_critique"):
                print(f"\n## RED-TEAM CRITIQUE\n{d['redteam_critique']}")
            print(f"\n{'='*60}")
            print(f"Board: {d['members_responded']}/{d['members_total']} responded | {d['total_latency_ms']}ms")
        else:
            print(f"ERROR: {result.error}")

    asyncio.run(_main())
