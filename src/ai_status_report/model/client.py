"""Bounded provider HTTP transport; https://www.python-httpx.org/quickstart/ ."""

import json
import time

import httpx

from ai_status_report.settings import Profile
from ai_status_report.token_budget.allocator import RUN, SEARCH_REQUESTS, TokenLedger
from ai_status_report.token_budget.estimator import estimate_messages, estimate_tokens
from ai_status_report.token_budget.usage import parse_usage, usage_tokens


class ProviderError(RuntimeError):
    """Only fixed error codes and numeric HTTP status may leave this boundary."""


class ProviderClient:
    def __init__(self, profile: Profile, timeout: float, *, transport=None, ledger: TokenLedger | None = None):
        self.profile = profile
        self.timeout = timeout
        # Every provider client owns an accounting boundary. Production callers
        # pass the run-scoped ledger; standalone diagnostics get an isolated
        # ledger instead of an unaccounted request.
        self.ledger = ledger or TokenLedger(f"unscoped-{profile.model}")
        self.http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(10, timeout)),
            follow_redirects=False,
            transport=transport,
            headers={"Authorization": f"Bearer {profile.api_key}"},
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.http.close()

    def post(self, path: str, payload: dict) -> dict:
        if path not in {
            "/chat/completions",
            "/responses",
            "/layout_parsing",
            "/embeddings",
            "/rerank",
        }:
            raise ProviderError("unapproved_api_path")
        try:
            response = self.http.post(self.profile.base_url + path, json=payload)
            if response.status_code != 200:
                raise ProviderError(f"http_{response.status_code}")
            data = response.json()
            if not isinstance(data, dict) or data.get("error"):
                raise ProviderError("invalid_provider_response")
            return data
        except httpx.TimeoutException:
            raise ProviderError("timeout") from None
        except httpx.HTTPError:
            raise ProviderError("transport_error") from None
        except ValueError:
            raise ProviderError("invalid_json") from None

    def accounted_post(self, path: str, payload: dict, *, estimate: int) -> dict:
        """Reserve before one billed request and settle from Provider usage."""

        reservation = self.ledger.reserve(RUN, estimate)
        try:
            response = self.post(path, payload)
        except Exception:
            self.ledger.settle(reservation, None)
            raise
        usage = parse_usage(response)
        self.ledger.settle(reservation, usage_tokens(usage) if usage else None)
        return response


class DeepSeekClient(ProviderClient):
    def payload(self, messages: list[dict], **options) -> dict:
        # The connectivity probe uses non-thinking mode; thinking support needs its own tests.
        # https://api-docs.deepseek.com/guides/thinking_mode/
        return {
            "model": self.profile.model,
            "messages": messages,
            "max_tokens": 128,
            "thinking": {"type": "disabled"},
            **options,
        }

    def chat(self, messages: list[dict], **options) -> dict:
        payload = self.payload(messages, **options)
        return self.accounted_post(
            "/chat/completions",
            payload,
            estimate=max(estimate_messages(messages) + int(payload.get("max_tokens", 0)), 0),
        )

    def web_search(self, query: str, *, max_output_tokens: int = 65536) -> dict:
        """Use DeepSeek's native Responses API web_search tool."""
        if not 1024 <= max_output_tokens <= 384000:
            raise ProviderError("invalid_web_search_budget")
        payload = {
            "model": self.profile.model,
            "input": query,
            "instructions": (
                "使用 web_search 工具检索真实来源。最终只输出 JSON，不输出 Markdown。"
                "JSON 必须包含 summary 和 sources。sources 是数组，每项必须包含 title、url、"
                "snippet；url 必须是实际检索到的来源 URL，不能编造。没有可靠来源时返回空数组。"
            ),
            "reasoning": {"effort": "none"},
            "tools": [{"type": "web_search"}],
            "tool_choice": {"type": "web_search"},
            "max_output_tokens": max_output_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        self.ledger.meter(SEARCH_REQUESTS)
        return self.accounted_post(
            "/responses",
            payload,
            estimate=max(estimate_tokens(query) + max_output_tokens, 0),
        )

    def stream_probe(self, messages: list[dict]) -> dict:
        payload = self.payload(messages, stream=True, stream_options={"include_usage": True})
        chunks = 0
        characters = 0
        usage = None
        finish = None
        started = time.monotonic()
        reservation = self.ledger.reserve(
            RUN,
            max(estimate_messages(messages) + int(payload.get("max_tokens", 0)), 0),
        )
        try:
            with self.http.stream(
                "POST", self.profile.base_url + "/chat/completions", json=payload
            ) as response:
                if response.status_code != 200:
                    raise ProviderError(f"http_{response.status_code}")
                for line in response.iter_lines():
                    if time.monotonic() - started > self.timeout:
                        raise ProviderError("stream_deadline")
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        if characters == 0 or finish != "stop":
                            raise ProviderError("incomplete_stream")
                        self.ledger.settle(
                            reservation,
                            usage_tokens(parse_usage({"usage": usage})) if usage else None,
                        )
                        return {"chunks": chunks, "characters": characters, "usage": usage}
                    event = json.loads(body)
                    if "error" in event:
                        raise ProviderError("stream_provider_error")
                    chunks += 1
                    if chunks > 1024:
                        raise ProviderError("stream_event_limit")
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        characters += len(choice.get("delta", {}).get("content") or "")
                        finish = choice.get("finish_reason") or finish
        except ProviderError:
            self.ledger.settle(reservation, None)
            raise
        except httpx.TimeoutException:
            self.ledger.settle(reservation, None)
            raise ProviderError("timeout") from None
        except httpx.HTTPError:
            self.ledger.settle(reservation, None)
            raise ProviderError("transport_error") from None
        except (ValueError, TypeError, AttributeError):
            self.ledger.settle(reservation, None)
            raise ProviderError("invalid_stream") from None
        self.ledger.settle(reservation, None)
        raise ProviderError("stream_missing_done")
