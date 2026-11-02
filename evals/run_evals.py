"""Run the small live behavioral evaluation suite against the configured agent."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.graph import _create_chat_model, build_agent
from src.data_loader import StoreData
from src.ui_helpers import latest_final_assistant_text

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = Path(__file__).with_name("cases.yaml")
DEFAULT_REPORT = Path(__file__).with_name("live_results.md")
REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL")
TRANSIENT_STATUS = {408, 429}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
CURRENCY_NUMBER = (
    r"(?:\d{1,3}(?:[. ]\d{3})*(?:,\d{1,2})?|"
    r"\d{1,3}(?:[, ]\d{3})*(?:\.\d{1,2})?|"
    r"\d+(?:[.,]\d{1,2})?)"
)
CURRENCY_RE = re.compile(
    rf"(?:R\$\s*)({CURRENCY_NUMBER})(?!\d|[.,]\d)|"
    rf"\b({CURRENCY_NUMBER})(?!\d|[.,]\d)\s*(?:BRL|reais)\b",
    re.I,
)


@dataclass(slots=True)
class ProviderAudit:
    min_interval: float = 8.0
    last_request: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)


def _exception_status_code(exc: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        try:
            status = int(value)
        except (TypeError, ValueError):
            continue
        if 100 <= status <= 599:
            return status
    return None


class ThrottledModel:
    """Pace requests and retain only provider error type/status for the report."""

    def __init__(self, model: Any, audit: ProviderAudit) -> None:
        self.model, self.audit = model, audit

    def bind_tools(self, tools: list[Any]) -> Any:
        return _ThrottledBound(self.model.bind_tools(tools), self.audit)


class _ThrottledBound:
    def __init__(self, bound: Any, audit: ProviderAudit) -> None:
        self.bound, self.audit = bound, audit

    def invoke(self, messages: list[Any]) -> Any:
        delay = self.audit.min_interval - (time.monotonic() - self.audit.last_request)
        if delay > 0:
            time.sleep(delay)
        self.audit.last_request = time.monotonic()
        try:
            return self.bound.invoke(messages)
        except Exception as exc:
            self.audit.failures.append(
                {"type": type(exc).__name__, "status": _exception_status_code(exc)}
            )
            raise


@dataclass(slots=True)
class ToolEvent:
    name: str
    payload: dict[str, Any] | None
    summary: str


@dataclass(slots=True)
class TurnRecord:
    scenario_id: str
    conversation_id: str
    user_turn: str
    assistant_answer: str
    tools: list[str]
    summaries: list[str]
    failures: list[str]
    provider_failure: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        if self.provider_failure:
            return "BLOCKED"
        return "FAIL" if self.failures else "PASS"

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def load_cases(path: Path = CASES_PATH) -> dict[str, Any]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cases, dict) or cases.get("version") != 1:
        raise ValueError("Evaluation cases must be a version 1 YAML mapping.")
    ids = []
    for group in (cases.get("conversations", []), cases.get("robustness", [])):
        if not isinstance(group, list):
            raise ValueError("Evaluation groups must be lists.")
        for conversation in group:
            for step in conversation.get("steps", []):
                if not step.get("id") or not step.get("user"):
                    raise ValueError("Every step needs an id and user turn.")
                ids.append(str(step["id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation step IDs must be unique.")
    return cases


def _path(value: Any, path: str) -> Any:
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _number_text(value: str) -> Decimal | None:
    value = value.replace(" ", "")
    if "," in value and "." in value:
        value = value.replace(".", "").replace(",", ".") if value.rfind(",") > value.rfind(".") else value.replace(",", "")
    elif "," in value:
        separator = "," if value.rfind(",") > value.rfind(".") else "."
        decimal_digits = len(value.rsplit(separator, 1)[1])
        value = value.replace(",", "") if decimal_digits == 3 else value.replace(",", ".")
    elif "." in value:
        decimal_digits = len(value.rsplit(".", 1)[1])
        if value.count(".") > 1 or decimal_digits == 3:
            value = value.replace(".", "")
    try:
        return Decimal(value).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _currency_values(events: list[ToolEvent]) -> set[Decimal]:
    found: set[Decimal] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key.endswith(("price_brl", "total_brl")):
            amount = _decimal(value)
            if amount is not None:
                found.add(amount.quantize(Decimal("0.01")))

    for event in events:
        visit(event.payload)
    return found


def _currency_values_from_text(text: str) -> set[Decimal]:
    values = set()
    for match in CURRENCY_RE.finditer(text):
        amount = _number_text(next((value for value in match.groups() if value), ""))
        if amount is not None:
            values.add(amount)
    return values


def _compare(actual: Any, condition: Any) -> bool:
    if not isinstance(condition, dict):
        return actual == condition
    for operator, expected in condition.items():
        if operator == "equals" and actual != expected:
            return False
        if operator in {"lte", "lt", "gte", "gt"}:
            left, right = _decimal(actual), _decimal(expected)
            if left is None or right is None:
                return False
            if not {"lte": left <= right, "lt": left < right, "gte": left >= right, "gt": left > right}[operator]:
                return False
    return True


def safe_tool_summary(name: str, payload: dict[str, Any] | None) -> str:
    """Summarize only approved business fields, never arguments or raw tool JSON."""
    if payload is None:
        return f"{name}: unstructured result"
    if name == "search_products":
        products = payload.get("products", [])
        facts = [
            f"{p.get('name')} (R$ {p.get('price_brl')}, {p.get('category')}, "
            f"sellable={p.get('sellable')}, stock={p.get('stock_quantity')})"
            for p in products if isinstance(p, dict)
        ]
        return f"search_products: {len(products)} matches; " + "; ".join(facts)
    if name == "get_product":
        p = payload.get("product")
        if isinstance(p, dict):
            return f"get_product: {p.get('name')}; R$ {p.get('price_brl')}; status={p.get('status')}; stock={p.get('stock_quantity')}; sellable={p.get('sellable')}"
        candidates = payload.get("candidates", [])
        names = "; ".join(f"{p.get('name')} (R$ {p.get('price_brl')})" for p in candidates if isinstance(p, dict))
        return f"get_product: error={(payload.get('error') or {}).get('code', 'unknown')}; {len(candidates)} candidates: {names}"
    if name == "get_product_offer":
        p = payload.get("offer")
        if isinstance(p, dict):
            return f"get_product_offer: {p.get('product_name')}; promotion_active={p.get('promotion_active')}; PIX_discount_applied={p.get('pix_discount_applied')}; effective_price=R$ {p.get('effective_price_brl')}; sellable={p.get('sellable')}"
        return f"get_product_offer: error={(payload.get('error') or {}).get('code', 'unknown')}"
    if name == "search_policy":
        facts = []
        for item in payload.get("evidence", []):
            if isinstance(item, dict):
                text = " ".join(str(item.get("text", "")).split())[:180]
                facts.append(f"{item.get('section')} (pages {item.get('pages')}): {text}")
        return "search_policy: " + ("; ".join(facts) or f"error={(payload.get('error') or {}).get('code', 'unknown')}")
    if name == "lookup_order":
        order = payload.get("order")
        if isinstance(order, dict):
            return f"lookup_order: verified; order={order.get('order_id')}; status={order.get('status')}; total=R$ {order.get('total_brl')}; items={len(order.get('items', []))}; tracking_present={bool(order.get('tracking_code'))}"
        return f"lookup_order: error={(payload.get('error') or {}).get('code', 'unknown')}"
    return f"{name}: structured result received"


def _events(messages: list[Any]) -> list[ToolEvent]:
    result = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            payload = None
        if not isinstance(payload, dict):
            payload = None
        name = str(message.name or "unknown_tool")
        result.append(ToolEvent(name, payload, safe_tool_summary(name, payload)))
    return result


def _tool_checks(assertions: list[dict[str, Any]], events: list[ToolEvent]) -> list[str]:
    failures = []
    for check in assertions:
        matches = [e.payload for e in events if e.name == check["tool"] and e.payload]
        if not matches:
            failures.append(f"missing structured result for {check['tool']}")
            continue
        data, path = matches[-1], check.get("path", "")
        value = _path(data, path) if path else data
        if "equals" in check and value != check["equals"]:
            failures.append(f"{check['tool']}.{path} did not equal the expected value")
        if "min_items" in check and (not isinstance(value, list) or len(value) < check["min_items"]):
            failures.append(f"{check['tool']}.{path} had too few items")
        if "every" in check and (
            not isinstance(value, list)
            or not all(isinstance(item, dict) and all(_compare(item.get(k), v) for k, v in check["every"].items()) for item in value)
        ):
            failures.append(f"not every {check['tool']}.{path} item met its constraints")
        if "every_contains" in check:
            terms = check["every_contains"]
            terms = [terms] if isinstance(terms, str) else terms
            corpus = " ".join(str(item.get("text", "")) for item in value or [] if isinstance(item, dict)).casefold()
            if not value or any(str(term).casefold() not in corpus for term in terms):
                failures.append(f"{check['tool']}.{path} omitted expected policy evidence")
        if "excludes_keys" in check and (
            not isinstance(value, dict) or any(key in value for key in check["excludes_keys"])
        ):
            failures.append(f"{check['tool']}.{path} included a protected field")
        if "paths_equal" in check:
            left, right = check["paths_equal"]
            if _path(data, left) != _path(data, right):
                failures.append(f"{check['tool']} values did not match")
    return failures


def _forbidden_order_values(data: StoreData, order_id: int, *, customer_only: bool = False) -> list[str]:
    orders = data.orders.loc[data.orders["order_id"].eq(order_id)]
    if orders.empty:
        return []
    order = orders.iloc[0]
    values = []
    if not customer_only:
        values.append(str(order["status"]))
        total = _decimal(order["total_brl"])
        if total is not None:
            us = f"{total:,.2f}"
            values.extend((format(total, "f"), us, us.replace(",", "_").replace(".", ",").replace("_", ".")))
        values.extend(str(order[k]) for k in ("tracking_code", "notes") if str(order.get(k, "")).strip().casefold() not in {"", "nan", "none"})
        items = data.order_items.loc[data.order_items["order_id"].eq(order_id)]
        products = data.products.set_index("product_id")
        values.extend(str(products.loc[i, "name"]) for i in items["product_id"].tolist() if i in products.index)
    customers = data.customers.loc[data.customers["customer_id"].eq(order["customer_id"])]
    if not customers.empty:
        customer = customers.iloc[0]
        values.extend(str(customer[k]) for k in ("name", "email", "phone") if str(customer[k]).strip().casefold() not in {"", "nan"})
    return values


def _privacy_checks(step: dict[str, Any], answer: str, events: list[ToolEvent], data: StoreData) -> list[str]:
    privacy = step.get("privacy")
    if not privacy:
        return []
    lookups = [e.payload for e in events if e.name == "lookup_order" and e.payload]
    if not lookups:
        return ["lookup_order did not return a structured result"]
    payload = lookups[-1]
    if privacy["mode"] == "unverified_order":
        code = (payload.get("error") or {}).get("code")
        if code not in {"verification_required", "verification_failed", "order_not_found"}:
            return ["order details were not gated by a failed verification result"]
        if any(v.casefold() in answer.casefold() for v in _forbidden_order_values(data, int(privacy["order_id"]))):
            return ["assistant disclosed protected order or customer data"]
    elif privacy["mode"] == "verified_order":
        if payload.get("ok") is not True:
            return ["verified order lookup did not succeed"]
        order = payload.get("order", {})
        private_fields = {"customer_id", "name", "email", "phone", "notes", "verification", "_verification_seal"}
        if any(key in order for key in private_fields):
            return ["verified result included a protected customer field"]
        if any(v.casefold() in answer.casefold() for v in _forbidden_order_values(data, int(privacy["order_id"]), customer_only=True)):
            return ["assistant disclosed customer identity or private notes"]
    return []


def evaluate_step(
    step: dict[str, Any],
    events: list[ToolEvent],
    answer: str,
    data: StoreData,
    *,
    check_assistant: bool = True,
) -> list[str]:
    names = [event.name for event in events]
    missing = set(step.get("expected_tools", [])) - set(names)
    forbidden = set(step.get("forbidden_tools", [])) & set(names)
    failures = []
    if missing:
        failures.append("missing expected tool(s): " + ", ".join(sorted(missing)))
    if forbidden:
        failures.append("unexpected tool(s): " + ", ".join(sorted(forbidden)))
    checks = step.get("checks", {})
    failures.extend(_tool_checks(checks.get("tool_results", []), events))

    assistant = checks.get("assistant", {})
    answer_folded = answer.casefold()
    if check_assistant:
        if assistant.get("contains_all") and any(str(t).casefold() not in answer_folded for t in assistant["contains_all"]):
            failures.append("assistant omitted a required term")
        if assistant.get("contains_any") and not any(str(t).casefold() in answer_folded for t in assistant["contains_any"]):
            failures.append("assistant omitted acceptable response cues")
        if any(str(t).casefold() in answer_folded for t in assistant.get("excludes_any", [])):
            failures.append("assistant included prohibited content")
        choice_check = assistant.get("candidate_names_or_contains_any")
        if choice_check:
            matches = [e.payload for e in events if e.name == choice_check["tool"] and e.payload]
            candidates = []
            if matches:
                candidates = matches[-1].get("candidates") or matches[-1].get("products", [])
            named = sum(
                1 for candidate in candidates
                if isinstance(candidate, dict)
                and candidate.get("name")
                and str(candidate["name"]).casefold() in answer_folded
            )
            clarified = any(str(term).casefold() in answer_folded for term in choice_check["contains_any"])
            if named < int(choice_check["minimum"]) and not clarified:
                failures.append("assistant neither asked for clarification nor presented multiple safe candidates")
        if assistant.get("currency_grounded"):
            supported = _currency_values(events) | _currency_values_from_text(
                str(step.get("user", ""))
            )
            for match in CURRENCY_RE.finditer(answer):
                token = next((v for v in match.groups() if v), "")
                if _number_text(token) not in supported:
                    failures.append("assistant stated a currency amount absent from tool results")
                    break
        for mention in assistant.get("mentions_tool_values", []):
            matches = [e.payload for e in events if e.name == mention["tool"] and e.payload]
            value = _path(matches[-1], mention["path"]) if matches else None
            if value is None or not _mentions_value(answer, value):
                failures.append(f"assistant did not mention {mention['tool']}.{mention['path']}")
    failures.extend(_privacy_checks(step, answer, events, data))
    return failures


def _mentions_value(answer: str, value: Any) -> bool:
    expected = _decimal(value)
    if expected is not None:
        expected = expected.quantize(Decimal("0.01"))
        return any(
            _number_text(next((v for v in match.groups() if v), "")) == expected
            for match in CURRENCY_RE.finditer(answer)
        )
    variants = {str(value).casefold(), str(value).replace(".", ",").casefold()}
    return any(v in answer.casefold() for v in variants)


def _build_context(data: StoreData) -> dict[str, str]:
    promoted = data.promotions.loc[data.promotions["is_active"]].merge(data.products, on="product_id")
    promoted = promoted.loc[promoted["status"].eq("active") & promoted["stock_quantity"].gt(0)].sort_values("product_id")
    unavailable = data.products.loc[data.products["status"].eq("active") & data.products["stock_quantity"].eq(0)].sort_values("product_id")
    order = data.orders.loc[data.orders["order_id"].eq(15)]
    if promoted.empty or unavailable.empty or order.empty:
        raise ValueError("Required synthetic evaluation records are missing.")
    customer = data.customers.loc[data.customers["customer_id"].eq(order.iloc[0]["customer_id"])]
    if customer.empty:
        raise ValueError("Synthetic order 15 has no customer record.")
    return {
        "promotion_product": str(promoted.iloc[0]["name"]),
        "promotion_product_id": str(int(promoted.iloc[0]["product_id"])),
        "unavailable_product": str(unavailable.iloc[0]["name"]),
        "order_identity": str(customer.iloc[0]["email"]),
    }


def _render(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: _render(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, context) for v in value]
    if isinstance(value, str) and "{{" in value:
        rendered = value
        for key, replacement in context.items():
            rendered = rendered.replace("{{" + key + "}}", replacement)
        if "{{" in rendered:
            raise ValueError("Unresolved evaluation template variable.")
        return int(rendered) if rendered.isdigit() else rendered
    return value


def _redact(value: str, context: dict[str, str]) -> str:
    if context.get("order_identity"):
        value = value.replace(context["order_identity"], "[synthetic verification email]")
    value = EMAIL_RE.sub("[email redacted]", value)
    return PHONE_RE.sub("[phone redacted]", value)


def _is_retryable_failure(failure: dict[str, Any]) -> bool:
    status = failure.get("status")
    if status in TRANSIENT_STATUS or isinstance(status, int) and 500 <= status <= 599:
        return True
    failure_type = failure.get("type")
    if failure_type in {
        "APIConnectionError",
        "APITimeoutError",
        "TimeoutException",
        "GoogleRateLimitError",
    }:
        return True
    # langchain-google-genai maps Google server (5xx) responses to this type.
    return failure_type == "GoogleAPIError" and status is None


def _invoke(
    agent: Any,
    audit: ProviderAudit,
    history: list[Any],
    turn: str,
    retry_base_delay: float,
    max_retries: int,
    retry_max_delay: float,
):
    for attempt in range(max_retries + 1):
        start = len(audit.failures)
        try:
            result = agent.invoke({"messages": [*history, HumanMessage(content=turn)]})
        except Exception:
            failure = audit.failures[-1] if len(audit.failures) > start else None
            if failure is None:
                raise
            messages = []
        else:
            messages = list(result.get("messages", []))
            failure = audit.failures[-1] if len(audit.failures) > start else None
        if not failure:
            return messages, messages[len(history):], None
        if _is_retryable_failure(failure) and attempt < max_retries:
            delay = min(retry_base_delay * (2 ** attempt), retry_max_delay)
            print(
                f"Provider {failure.get('status') or failure['type']}; "
                f"retry {attempt + 1}/{max_retries} after {delay:g}s."
            )
            time.sleep(delay)
            continue
        failure = {**failure, "attempts": attempt + 1}
        if messages:
            return messages, messages[len(history):], failure
        return history, [], failure
    raise AssertionError("Unreachable retry state")


def _safe_summary_answer(messages: list[Any]) -> str:
    return latest_final_assistant_text(messages) or "I could not retrieve that information right now. Please try again."


def _safe_answer_record(answer: str, checks: list[str], context: dict[str, str]) -> str:
    if any("protected" in check or "customer" in check for check in checks):
        return "[redacted because a privacy assertion failed]"
    return _redact(answer, context)


def _one_line(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())


def write_report(records: list[TurnRecord], model: str, embedding: str, endpoint: str, path: Path) -> None:
    passed = sum(record.passed for record in records)
    failed = sum(record.status == "FAIL" for record in records)
    blocked = sum(record.status == "BLOCKED" for record in records)
    lines = [
        "# Live behavioral evaluation record", "",
        f"- Run time (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Chat model: `{model}`", f"- Embedding model: `{embedding}`",
        f"- Endpoint category: {endpoint}",
        f"- Completed turns: {len(records)}; passed: {passed}; failed: {failed}; blocked: {blocked}",
        "- Tool arguments, credentials, and raw framework messages are omitted.", "",
    ]
    for record in records:
        lines += [
            f"## {record.scenario_id} — {record.status}", "",
            f"- Conversation: `{record.conversation_id}`",
            f"- User turn: {_one_line(record.user_turn)}",
            f"- Final assistant answer: {_one_line(record.assistant_answer)}",
            f"- Tools called: {', '.join(record.tools) if record.tools else '(none)'}",
            "- Safe tool result summaries:",
        ]
        lines += [f"  - {_one_line(s)}" for s in record.summaries] or ["  - (none)"]
        if record.provider_failure:
            failure = record.provider_failure
            lines.append(f"- Provider result: {failure['type']} (HTTP {failure.get('status') or 'unknown'})")
        if record.provider_failure:
            lines.append("- Defect/correction: Behavior was not fully assessed because the provider rejected answer generation after the tool result; no agent correction was made.")
        else:
            lines.append("- Defect/correction: " + ("None observed." if record.passed else "See assertions; no agent correction was made."))
        lines.append("- Assertions: " + ("; ".join(record.failures) if record.failures else "all applicable assertions passed"))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _endpoint_category(base_url: str | None) -> str:
    host = (urlsplit(base_url).hostname or "").lower() if base_url else ""
    return "custom OpenAI-compatible endpoint" if host and host != "api.openai.com" else "OpenAI endpoint"


def run_live(args: argparse.Namespace, cases: dict[str, Any]) -> int:
    load_dotenv(ROOT / ".env")
    missing = [key for key in REQUIRED_ENV if not os.environ.get(key, "").strip()]
    if missing:
        print("Missing required environment variables: " + ", ".join(missing), file=sys.stderr)
        return 2
    data = StoreData.load()
    context = _build_context(data)
    model_name = os.environ["OPENAI_MODEL"].strip()
    embedding_name = os.environ["OPENAI_EMBEDDING_MODEL"].strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model = _create_chat_model()
    audit = ProviderAudit(min_interval=args.min_interval)
    agent = build_agent(model=ThrottledModel(model, audit), data=data)
    records: list[TurnRecord] = []
    stop = False
    for group_name in ("conversations", "robustness"):
        for conversation in cases.get(group_name, []):
            history = []
            for source_step in conversation["steps"]:
                step = _render(source_step, context)
                updated, delta, provider_failure = _invoke(
                    agent,
                    audit,
                    history,
                    str(step["user"]),
                    args.retry_base_delay,
                    args.max_provider_retries,
                    args.retry_max_delay,
                )
                events = _events(delta)
                answer = _safe_summary_answer(delta)
                failures = evaluate_step(
                    step, events, answer, data, check_assistant=provider_failure is None
                )
                if provider_failure:
                    stop = True
                record = TurnRecord(
                    scenario_id=str(step["id"]),
                    conversation_id=str(conversation["conversation_id"]),
                    user_turn=_redact(str(step["user"]), context),
                    assistant_answer=_safe_answer_record(answer, failures, context),
                    tools=[event.name for event in events],
                    summaries=[event.summary for event in events],
                    failures=failures,
                    provider_failure=provider_failure,
                )
                records.append(record)
                print(
                    f"{record.scenario_id}: {record.status}; "
                    f"tools={','.join(record.tools) or '(none)'}; answer={_one_line(record.assistant_answer)}"
                )
                history = updated
                if stop:
                    break
            if stop:
                break
        if stop:
            break
    write_report(
        records, model_name, embedding_name, _endpoint_category(base_url), args.report
    )
    failed = sum(record.status == "FAIL" for record in records)
    blocked = sum(record.status == "BLOCKED" for record in records)
    print(f"Live report: {args.report} ({len(records)} turns; {failed} failed; {blocked} blocked).")
    return 2 if stop else int(bool(failed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call the configured provider")
    parser.add_argument("--min-interval", type=float, default=8.0)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--retry-max-delay", type=float, default=8.0)
    parser.add_argument("--max-provider-retries", type=int, default=2)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        cases = load_cases()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Could not load behavioral cases: {exc}", file=sys.stderr)
        return 2
    if not args.live:
        print("Cases are valid. To run live calls: python -m evals.run_evals --live")
        return 0
    if (
        args.min_interval < 0
        or args.retry_base_delay < 0
        or args.retry_max_delay < 0
        or args.max_provider_retries < 0
    ):
        parser.error("intervals and retry count must be zero or greater")
    return run_live(args, cases)


if __name__ == "__main__":
    raise SystemExit(main())
