"""Explicit paid provider probes using only public test inputs."""

import json
from datetime import UTC, datetime

from ai_status_report.model.client import DeepSeekClient, ProviderError
from ai_status_report.rag.glm import GLMClient

OCR_SAMPLE = "https://cdn.bigmodel.cn/static/logo/introduction.png"


def safe_usage(result: dict) -> dict:
    usage = result.get("usage") or {}
    return {
        k: usage[k]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if type(usage.get(k)) is int and usage[k] >= 0
    }


def message(result: dict, finish="stop") -> dict:
    try:
        choice = result["choices"][0]
        if choice["finish_reason"] != finish:
            raise ProviderError("unexpected_finish_reason")
        return choice["message"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError("invalid_completion_schema") from None


def run_checks(settings, service="all", *, ledger=None) -> dict:
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "small public samples; no production quality claim",
        "checks": {},
    }

    def check(name, operation):
        try:
            details = operation()
            report["checks"][name] = {"status": "passed", **details}
        except ProviderError as exc:
            report["checks"][name] = {"status": "failed", "reason": str(exc)}
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            report["checks"][name] = {"status": "failed", "reason": "response_validation"}
        print(name + ": " + report["checks"][name]["status"], flush=True)

    if service in {"all", "deepseek"}:
        with DeepSeekClient(settings.generation, settings.timeout, ledger=ledger) as client:

            def text_check():
                result = client.chat([{"role": "user", "content": "Reply with the word READY."}])
                if "READY" not in (message(result).get("content") or ""):
                    raise ProviderError("text_expectation")
                return {"usage": safe_usage(result)}

            def json_check():
                # https://api-docs.deepseek.com/guides/json_mode/
                result = client.chat(
                    [
                        {
                            "role": "user",
                            "content": 'Return exactly this json object: {"ready": true}',
                        }
                    ],
                    response_format={"type": "json_object"},
                )
                if json.loads(message(result)["content"]) != {"ready": True}:
                    raise ProviderError("json_expectation")
                return {"usage": safe_usage(result)}

            def tool_check():
                # Local deterministic probe, no MCP/A2A success claim.
                # https://api-docs.deepseek.com/guides/tool_calls/
                messages = [{"role": "user", "content": "Call add with a=2,b=3. Then reply 5."}]
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "description": "Add integers",
                            "parameters": {
                                "type": "object",
                                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                "required": ["a", "b"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
                first = client.chat(
                    messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "add"}},
                )
                assistant = message(first, "tool_calls")
                calls = assistant["tool_calls"]
                if len(calls) != 1:
                    raise ProviderError("unexpected_tool_count")
                call = calls[0]
                args = json.loads(call["function"]["arguments"])
                if call["function"]["name"] != "add" or args != {"a": 2, "b": 3}:
                    raise ProviderError("tool_arguments")
                if not all(type(v) is int for v in args.values()):
                    raise ProviderError("tool_argument_types")
                messages.extend(
                    [
                        assistant,
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": str(args["a"] + args["b"]),
                        },
                    ]
                )
                second = client.chat(messages, tools=tools, tool_choice="none")
                if "5" not in message(second)["content"]:
                    raise ProviderError("tool_result_consumption")
                return {"rounds": 2, "usage": [safe_usage(first), safe_usage(second)]}

            check("deepseek_text", text_check)
            check("deepseek_json", json_check)
            check("deepseek_tool_roundtrip", tool_check)

            def streaming():
                result = client.stream_probe([{"role": "user", "content": "Reply READY."}])
                return {
                    "chunks": result["chunks"],
                    "characters": result["characters"],
                    "usage": safe_usage(result),
                }

            check("deepseek_stream", streaming)

    if service in {"all", "ocr"}:
        with GLMClient(settings.ocr, settings.timeout, ledger=ledger) as client:

            def ocr_check():
                result = client.parse_document(OCR_SAMPLE)
                text = result.get("md_results")
                pages = result.get("layout_details")
                if (
                    not isinstance(text, str)
                    or not text.strip()
                    or not isinstance(pages, list)
                    or not pages
                ):
                    raise ProviderError("ocr_empty_text_or_layout")
                blocks = [block for page in pages for block in page]
                if not blocks or not all(
                    isinstance(b, dict) and "bbox_2d" in b and len(b["bbox_2d"]) == 4
                    for b in blocks
                ):
                    raise ProviderError("ocr_layout_schema")
                return {
                    "characters": len(text),
                    "pages": len(pages),
                    "blocks": len(blocks),
                    "usage": safe_usage(result),
                    "quality": "basic text/layout only; manual corpus review pending",
                }

            check("glm_ocr", ocr_check)

    if service in {"all", "embedding"}:
        with GLMClient(settings.embedding, settings.timeout, ledger=ledger) as client:

            def embedding_check():
                result = client.embed(
                    ["人工智能行业技术进展", "AI research and applications"], settings.dimensions
                )
                return {
                    "count": len(result["data"]),
                    "dimensions": settings.dimensions,
                    "finite_values": True,
                    "usage": safe_usage(result),
                }

            check("glm_embedding", embedding_check)
    return report
