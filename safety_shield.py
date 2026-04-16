from __future__ import annotations

import re
from typing import Any


US_CRISIS_RESOURCES = [
    {
        "name": "988 Suicide & Crisis Lifeline",
        "contact": "Call or text 988",
        "url": "https://988lifeline.org",
    },
    {
        "name": "Crisis Text Line",
        "contact": "Text HOME to 741741",
        "url": "https://www.crisistextline.org",
    },
    {
        "name": "Emergency services",
        "contact": "Call 911 or go to the nearest emergency department",
        "url": "",
    },
]

INFORMATIONAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*what\s+is\s+",
        r"^\s*what\s+are\s+",
        r"^\s*define\s+",
        r"^\s*explain\s+",
        r"^\s*tell\s+me\s+about\s+",
        r"^\s*how\s+common\s+is\s+",
        r"^\s*what\s+are\s+the\s+(symptoms|criteria|risk factors|treatments)\s+of\s+",
    ]
]

FIRST_PERSON_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bi\b",
        r"\bme\b",
        r"\bmy\b",
        r"\bmyself\b",
        r"\bi'm\b",
        r"\bi am\b",
        r"\bi've\b",
        r"\bi have\b",
    ]
]

CRISIS_SIGNAL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "explicit_self_harm_intent": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\bkill(?:ing)? myself\b",
            r"\bend(?:ing)? my life\b",
            r"\bcommit(?:ting)? suicide\b",
            r"\bsuicidal\b",
            r"\bsuicide\b",
            r"\bharm(?:ing)? myself\b",
            r"\bhurt(?:ing)? myself\b",
            r"\bself[- ]?harm\b",
            r"\bdo not want to live\b",
            r"\bdont want to live\b",
            r"\bdon't want to live\b",
            r"\bcan'?t go on\b",
            r"\bcannot go on\b",
            r"\bno way? ?to live\b",
            r"\bwant(?:ing)? to die\b",
            r"\bwish(?:ed)? i were dead\b",
            r"\bbetter off dead\b",
            r"\bend it all\b",
        ]
    ],
    "imminence_or_plan": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\btonight\b",
            r"\bright now\b",
            r"\btoday\b",
            r"\bthis evening\b",
            r"\bplan to\b",
            r"\bplanning to\b",
            r"\bgoing to\b",
            r"\boverdose\b",
            r"\bpills\b",
            r"\bgun\b",
            r"\brope\b",
            r"\bknife\b",
            r"\bjump off\b",
        ]
    ],
    "severe_distress": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\bcan'?t go on\b",
            r"\bcannot go on\b",
            r"\bno reason to live\b",
            r"\bnot safe\b",
            r"\babout to\b",
            r"\bneed help now\b",
            r"\bhit rock bottom\b",
            r"\bworthless\b",
            r"\bhopeless\b",
            r"\bempty\b",
            r"\bgive up\b",
            r"\bno point\b",
            r"\bnot worth\b",
        ]
    ],
}


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _matched_indicators(text: str) -> list[str]:
    matched: list[str] = []
    for label, patterns in CRISIS_SIGNAL_PATTERNS.items():
        if _matches_any(text, patterns):
            matched.append(label)
    return matched


def build_crisis_response(resources: list[dict[str, str]] | None = None) -> str:
    resources = resources or US_CRISIS_RESOURCES
    lines = [
        "I’m concerned that you may be in immediate distress.",
        "Please pause the GraphRAG flow and reach out for urgent help right now:",
    ]
    for resource in resources:
        line = f"- {resource['name']}: {resource['contact']}"
        if resource.get("url"):
            line += f" ({resource['url']})"
        lines.append(line)
    return "\n".join(lines)


def assess_safety_risk(text: str) -> dict[str, Any]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return {
            "is_crisis": False,
            "matched_indicators": [],
            "resources": [],
            "response": "",
        }

    informational = _matches_any(normalized, INFORMATIONAL_PATTERNS)
    first_person = _matches_any(normalized, FIRST_PERSON_PATTERNS)
    indicators = _matched_indicators(normalized)

    is_crisis = False
    if indicators:
        if informational and not first_person and "imminence_or_plan" not in indicators and "severe_distress" not in indicators:
            is_crisis = False
        elif first_person or "imminence_or_plan" in indicators or "severe_distress" in indicators:
            is_crisis = True
        elif not informational:
            is_crisis = True

    resources = US_CRISIS_RESOURCES if is_crisis else []
    response = build_crisis_response(resources) if is_crisis else ""
    return {
        "is_crisis": is_crisis,
        "matched_indicators": indicators,
        "resources": resources,
        "response": response,
    }
