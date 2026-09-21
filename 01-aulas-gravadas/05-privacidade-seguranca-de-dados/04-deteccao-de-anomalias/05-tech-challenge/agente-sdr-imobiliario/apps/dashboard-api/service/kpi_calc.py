from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

PIPELINE_STATES = (
    "greeting",
    "elicitation",
    "intent",
    "qualification",
    "recommendation",
    "scheduling",
    "handoff",
    "followup",
)
UNKNOWN_STATE_LABEL = "outros"
QUALIFIED_STATES = ("recommendation", "scheduling", "handoff")
SCHEDULED_STATES = ("scheduling", "handoff")
QUALIFIED_STATUSES = ("qualified",)
ROUTE_ROTATION_LABEL = "consultores"
ROUTE_SPECIALIST_LABEL = "diretor"
AREA_THRESHOLD_M2 = 500
ROUTE_OVERRIDE_FIELDS = ("route", "route_target", "assigned_to")
_HOUR = 60 * 60


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def count_new_leads(
    profiles: dict[str, dict[str, Any]], lower: datetime, now: datetime
) -> int:
    count = 0
    for profile in profiles.values():
        created = parse_iso8601(profile.get("created_at"))
        if created is not None and lower <= created <= now:
            count += 1
    return count


def start_of_today(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def days_ago(now: datetime, days: int) -> datetime:
    return now - timedelta(days=days)


def latest_conversation_by_lead(conversations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for conversation in conversations:
        lead_id = conversation.get("lead_id")
        if not lead_id:
            continue
        created = parse_iso8601(conversation.get("created_at"))
        key = (created or datetime.min.replace(tzinfo=timezone.utc), str(conversation.get("session_id") or ""))
        current = latest.get(str(lead_id))
        if current is None:
            latest[str(lead_id)] = conversation
            continue
        current_created = parse_iso8601(current.get("created_at"))
        current_key = (
            current_created or datetime.min.replace(tzinfo=timezone.utc),
            str(current.get("session_id") or ""),
        )
        if key > current_key:
            latest[str(lead_id)] = conversation
    return latest


def is_qualified(profile: dict[str, Any], conversation: dict[str, Any] | None) -> bool:
    if str(profile.get("status") or "") in QUALIFIED_STATUSES:
        return True
    if conversation is None:
        return False
    if conversation.get("context", {}).get("lead_qualified") is True:
        return True
    return str(conversation.get("current_state") or "") in QUALIFIED_STATES


def qualification_rate(
    profiles: dict[str, dict[str, Any]], latest_by_lead: dict[str, dict[str, Any]]
) -> float:
    total = len(profiles)
    if total == 0:
        return 0.0
    qualified = sum(
        1 for lead_id, profile in profiles.items() if is_qualified(profile, latest_by_lead.get(lead_id))
    )
    return round(qualified / total, 4)


def state_funnel(latest_by_lead: dict[str, dict[str, Any]]) -> dict[str, int]:
    funnel = {state: 0 for state in PIPELINE_STATES}
    funnel[UNKNOWN_STATE_LABEL] = 0
    for conversation in latest_by_lead.values():
        state = str(conversation.get("current_state") or "")
        if state in funnel:
            funnel[state] += 1
        else:
            funnel[UNKNOWN_STATE_LABEL] += 1
    return funnel


def scheduled_visits_count(latest_by_lead: dict[str, dict[str, Any]]) -> int:
    return sum(
        1
        for conversation in latest_by_lead.values()
        if str(conversation.get("current_state") or "") in SCHEDULED_STATES
    )


def intent_volume(profiles: dict[str, dict[str, Any]]) -> dict[str, int]:
    volume: dict[str, int] = {}
    for profile in profiles.values():
        intent = profile.get("intent")
        if not intent:
            continue
        label = str(intent)
        volume[label] = volume.get(label, 0) + 1
    return volume


def parse_area_m2(value: Any) -> float | None:
    """Converte `area` em m² (texto livre). Formatos aceitos: "800 m²" →
    800.0; "1.200 m²" (ponto = milhar, padrão u1) → 1200.0; "12,5 m²"
    (vírgula = decimal, PT-BR) → 12.5; "1.234,56" → 1234.56; "12.5" →
    12.5 (ponto decimal com 1–2 casas). Sem dígitos → None."""
    if value is None:
        return None
    match = re.search(r"\d+(?:[.,]\d+)*", str(value))
    if not match:
        return None
    raw = match.group(0)
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    else:
        parts = raw.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            raw = "".join(parts)
    return float(raw)


def route_for_profile(profile: dict[str, Any]) -> str:
    for field in ROUTE_OVERRIDE_FIELDS:
        route = profile.get(field)
        if route:
            return str(route)
    area = parse_area_m2(profile.get("area"))
    if area is not None and area > AREA_THRESHOLD_M2:
        return ROUTE_SPECIALIST_LABEL
    return ROUTE_ROTATION_LABEL


def route_distribution(profiles: dict[str, dict[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for profile in profiles.values():
        label = route_for_profile(profile)
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def alerts_last_24h(
    alerts: list[dict[str, Any]], now: datetime, hours: int = 24
) -> list[dict[str, Any]]:
    lower = now - timedelta(hours=hours)
    window: list[tuple[datetime, dict[str, Any]]] = []
    for alert in alerts:
        detected = parse_iso8601(alert.get("detected_at"))
        if detected is None or not lower <= detected <= now:
            continue
        window.append((detected, alert))
    window.sort(key=lambda pair: pair[0], reverse=True)
    return [alert for _, alert in window]
