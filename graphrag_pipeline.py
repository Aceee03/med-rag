from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import pickle
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings(
    "ignore",
    message="The 'validate_default' attribute with value True was provided to the `Field\\(\\)` function.*",
)

import networkx as nx
import numpy as np
from graspologic.partition import leiden
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.llms import ChatMessage

from pipeline_helpers import citation_from_metadata
from progress_utils import ProgressPrinter
from safety_shield import assess_safety_risk


CHECKPOINT_VERSION = 1
GRAPH_PROMPT_VERSION = "v6"
GRAPH_REPAIR_PROMPT_VERSION = "v1"
COMMUNITY_SUMMARY_PROMPT_VERSION = "v2"
COMMUNITY_SUMMARY_FILE = "community_summaries.json"
COMMUNITIES_FILE = "communities.json"
GRAPH_META_FILE = "graph_meta.json"
GRAPH_PROGRESS_FILE = "graph_progress.jsonl"
PARTIAL_STATE_FILE = "graph_partial_state.pkl"
PARTIAL_META_FILE = "graph_partial_meta.json"

GRAPH_REQUIRED_FILES = (
    "custom_entities.pkl",
    "entity_id_to_node.pkl",
    "entity_descriptions.pkl",
    "entity_sources.pkl",
    "custom_relations.pkl",
    "relation_metadata.pkl",
    "relation_index.pkl",
    "chunk_citation_map.pkl",
)
GRAPH_OPTIONAL_FILES = (
    "chunk_payload_map.pkl",
)

ALLOWED_TYPES = {
    "CONDITION",
    "SYMPTOM",
    "TREATMENT",
    "MEDICATION",
    "SAFETY_RESOURCE",
    "DEMOGRAPHIC",
    "DIAGNOSTIC_CRITERION",
    "SPECIFIER",
    "RISK_FACTOR",
    "COURSE_FEATURE",
    "PREVALENCE_STATEMENT",
}

ALLOWED_RELATIONS = {
    "HAS_SYMPTOM",
    "TREATED_BY",
    "PRESCRIBES",
    "SUITABLE_FOR",
    "CONTRAINDICATED_FOR",
    "URGENT_ACTION",
    "HAS_DIAGNOSTIC_CRITERION",
    "HAS_SPECIFIER",
    "HAS_RISK_FACTOR",
    "DIFFERENTIAL_DIAGNOSIS",
    "COMORBID_WITH",
    "HAS_COURSE",
    "HAS_PREVALENCE",
}

ABBREVIATION_MAP = {
    "CBT": "COGNITIVE BEHAVIORAL THERAPY",
    "DBT": "DIALECTICAL BEHAVIOR THERAPY",
    "ACT": "ACCEPTANCE AND COMMITMENT THERAPY",
    "EMDR": "EYE MOVEMENT DESENSITIZATION AND REPROCESSING",
    "ECT": "ELECTROCONVULSIVE THERAPY",
    "TMS": "TRANSCRANIAL MAGNETIC STIMULATION",
    "SSRI": "SELECTIVE SEROTONIN REUPTAKE INHIBITOR",
    "SNRI": "SEROTONIN NOREPINEPHRINE REUPTAKE INHIBITOR",
    "TCA": "TRICYCLIC ANTIDEPRESSANT",
    "MAOI": "MONOAMINE OXIDASE INHIBITOR",
    "MDD": "MAJOR DEPRESSIVE DISORDER",
    "GAD": "GENERALIZED ANXIETY DISORDER",
    "PTSD": "POST TRAUMATIC STRESS DISORDER",
    "OCD": "OBSESSIVE COMPULSIVE DISORDER",
    "BPD": "BORDERLINE PERSONALITY DISORDER",
    "ADHD": "ATTENTION DEFICIT HYPERACTIVITY DISORDER",
    "SAD": "SEASONAL AFFECTIVE DISORDER",
    "BDD": "BODY DYSMORPHIC DISORDER",
    "ASD": "AUTISM SPECTRUM DISORDER",
}

LAST_WORD_SINGULAR = {
    "SYMPTOMS": "SYMPTOM",
    "CONDITIONS": "CONDITION",
    "CRITERIA": "CRITERION",
    "SPECIFIERS": "SPECIFIER",
    "TREATMENTS": "TREATMENT",
    "MEDICATIONS": "MEDICATION",
    "THERAPIES": "THERAPY",
    "DISORDERS": "DISORDER",
    "EPISODES": "EPISODE",
    "BEHAVIORS": "BEHAVIOR",
    "BEHAVIOURS": "BEHAVIOUR",
    "THOUGHTS": "THOUGHT",
    "CHANGES": "CHANGE",
    "PROBLEMS": "PROBLEM",
    "DIFFICULTIES": "DIFFICULTY",
    "DISTURBANCES": "DISTURBANCE",
    "STABILIZERS": "STABILIZER",
    "INHIBITORS": "INHIBITOR",
    "ANTIDEPRESSANTS": "ANTIDEPRESSANT",
    "ANTIPSYCHOTICS": "ANTIPSYCHOTIC",
    "RESOURCES": "RESOURCE",
    "HALLUCINATIONS": "HALLUCINATION",
    "DELUSIONS": "DELUSION",
    "FLASHBACKS": "FLASHBACK",
    "NIGHTMARES": "NIGHTMARE",
    "ATTACKS": "ATTACK",
    "FEELINGS": "FEELING",
    "EXPERIENCES": "EXPERIENCE",
    "ISSUES": "ISSUE",
    "EVENTS": "EVENT",
    "TECHNIQUES": "TECHNIQUE",
    "APPROACHES": "APPROACH",
    "FEATURES": "FEATURE",
    "RATES": "RATE",
}

TYPE_PRIORITY = {
    "CONDITION": 10,
    "MEDICATION": 9,
    "TREATMENT": 8,
    "SYMPTOM": 7,
    "DIAGNOSTIC_CRITERION": 6,
    "SPECIFIER": 5,
    "RISK_FACTOR": 4,
    "COURSE_FEATURE": 3,
    "PREVALENCE_STATEMENT": 2,
    "DEMOGRAPHIC": 1,
    "SAFETY_RESOURCE": 0,
}

GARBAGE_FRAGMENTS = {
    "COMBINATION",
    "VARIOUS",
    "MULTIPLE",
    "SEVERAL",
    "LIFELONG",
    "ONGOING",
    "CONTINUED",
    "ADDITIONAL",
    "APPROACH",
    "OPTION",
    "STRATEGY",
    "PROGRAM",
    "INTERVENTION",
    "MANAGEMENT",
    "SUPPORT",
    "SERVICES",
    "CARE",
    "HELP",
    "FACTOR",
    "FACTORS",
    "ISSUE",
    "ISSUES",
    "SYMPTOM",
    "SYMPTOMS",
    "CONDITION",
    "CONDITIONS",
    "TREATMENT",
    "TREATMENTS",
    "MEDICATION",
    "MEDICATIONS",
    "INFORMATION",
    "TOPIC",
    "ASPECT",
    "THING",
    "THINGS",
    "LEVEL",
    "LEVELS",
}

GRAPH_STOP_WORDS = {"OF", "AND", "OR", "THE", "A", "AN", "WITH", "FOR", "IN", "TO"}

VALID_TRIPLET_TYPES = {
    ("CONDITION", "HAS_SYMPTOM", "SYMPTOM"),
    ("CONDITION", "TREATED_BY", "TREATMENT"),
    ("CONDITION", "TREATED_BY", "MEDICATION"),
    ("CONDITION", "PRESCRIBES", "MEDICATION"),
    ("CONDITION", "SUITABLE_FOR", "DEMOGRAPHIC"),
    ("CONDITION", "SUITABLE_FOR", "CONDITION"),
    ("MEDICATION", "SUITABLE_FOR", "DEMOGRAPHIC"),
    ("MEDICATION", "SUITABLE_FOR", "CONDITION"),
    ("TREATMENT", "SUITABLE_FOR", "DEMOGRAPHIC"),
    ("TREATMENT", "SUITABLE_FOR", "CONDITION"),
    ("CONDITION", "CONTRAINDICATED_FOR", "DEMOGRAPHIC"),
    ("MEDICATION", "CONTRAINDICATED_FOR", "DEMOGRAPHIC"),
    ("TREATMENT", "CONTRAINDICATED_FOR", "DEMOGRAPHIC"),
    ("CONDITION", "URGENT_ACTION", "SAFETY_RESOURCE"),
    ("SYMPTOM", "URGENT_ACTION", "SAFETY_RESOURCE"),
    ("SAFETY_RESOURCE", "URGENT_ACTION", "SAFETY_RESOURCE"),
    ("CONDITION", "HAS_DIAGNOSTIC_CRITERION", "DIAGNOSTIC_CRITERION"),
    ("CONDITION", "HAS_SPECIFIER", "SPECIFIER"),
    ("CONDITION", "HAS_RISK_FACTOR", "RISK_FACTOR"),
    ("CONDITION", "HAS_RISK_FACTOR", "DEMOGRAPHIC"),
    ("CONDITION", "HAS_RISK_FACTOR", "CONDITION"),
    ("CONDITION", "DIFFERENTIAL_DIAGNOSIS", "CONDITION"),
    ("CONDITION", "COMORBID_WITH", "CONDITION"),
    ("CONDITION", "HAS_COURSE", "COURSE_FEATURE"),
    ("CONDITION", "HAS_PREVALENCE", "PREVALENCE_STATEMENT"),
}

STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
}

QUERY_STOPWORDS = STOPWORDS | {
    "symptom",
    "symptoms",
    "condition",
    "conditions",
    "disorder",
    "disorders",
    "could",
    "might",
    "may",
    "would",
    "treated",
    "treatment",
}

MENTAL_HEALTH_KEYWORDS = {
    "depression",
    "anxiety",
    "ptsd",
    "trauma",
    "bipolar",
    "stress",
    "autism",
    "asd",
    "adhd",
    "mood",
    "disorder",
    "mental",
    "health",
    "therapy",
    "treatment",
    "symptom",
    "criteria",
    "criterion",
    "specifier",
    "specifiers",
    "risk",
    "factor",
    "factors",
    "differential",
    "diagnosis",
    "comorbidity",
    "comorbid",
    "course",
    "prevalence",
    "epidemiology",
    "panic",
    "ocd",
    "schizophrenia",
    "psychosis",
    "phobia",
    "sad",
    "seasonal",
    "affective",
    "perinatal",
    "postpartum",
    "fatigue",
    "insomnia",
    "hypersomnia",
    "hallucination",
    "delusion",
}

HUMAN_SYMPTOM_PATTERNS = [
    "i feel",
    "i am feeling",
    "i've been feeling",
    "i have",
    "i've had",
    "i am having",
    "i'm having",
    "i'm experiencing",
    "i have been experiencing",
    "i'm suffering",
    "i have been suffering",
    "what could this be",
    "what could this possibly be",
    "what might this be",
    "what could it be",
    "could this be",
    "does this sound like",
]

FOLLOW_UP_PATTERNS = [
    "and i",
    "also",
    "plus",
    "in addition",
    "another symptom",
    "more recently",
    "lately",
    "as well",
]

HUMAN_SYMPTOM_CUE_WORDS = {
    "appetite",
    "concentrate",
    "concentrating",
    "eat",
    "eating",
    "energy",
    "exhausted",
    "fatigue",
    "hopeless",
    "hungry",
    "irritable",
    "lost",
    "numb",
    "panic",
    "sad",
    "sadness",
    "sleep",
    "sleeping",
    "tired",
    "tiredness",
    "worry",
    "worried",
}

SYMPTOM_PHRASE_ALIASES = {
    "TIRED": ["FATIGUE", "TIREDNESS", "EASILY FATIGUED"],
    "TIRED ALL THE TIME": ["FATIGUE", "TIREDNESS", "EASILY FATIGUED"],
    "LOW ENERGY": ["FATIGUE", "TIREDNESS"],
    "EXHAUSTED": ["FATIGUE", "TIREDNESS", "EASILY FATIGUED"],
    "LOST MY APPETITE": ["LOSS OF APPETITE", "POOR APPETITE"],
    "LOST APPETITE": ["LOSS OF APPETITE", "POOR APPETITE"],
    "NO APPETITE": ["LOSS OF APPETITE", "POOR APPETITE"],
    "LOSS OF APPETITE": ["LOSS OF APPETITE", "POOR APPETITE"],
    "POOR APPETITE": ["POOR APPETITE", "LOSS OF APPETITE"],
    "CANT SLEEP": ["DISTURBED SLEEP", "SLEEP PROBLEM", "INSOMNIA"],
    "CAN T SLEEP": ["DISTURBED SLEEP", "SLEEP PROBLEM", "INSOMNIA"],
    "CAN'T SLEEP": ["DISTURBED SLEEP", "SLEEP PROBLEM", "INSOMNIA"],
    "TROUBLE SLEEPING": ["DISTURBED SLEEP", "SLEEP PROBLEM", "INSOMNIA"],
    "CANNOT SLEEP": ["DISTURBED SLEEP", "SLEEP PROBLEM", "INSOMNIA"],
    "TROUBLE CONCENTRATING": ["TROUBLE CONCENTRATING", "DIFFICULTY CONCENTRATING"],
    "CAN'T CONCENTRATE": ["TROUBLE CONCENTRATING", "DIFFICULTY CONCENTRATING"],
    "CANT CONCENTRATE": ["TROUBLE CONCENTRATING", "DIFFICULTY CONCENTRATING"],
    "LOST INTEREST": ["LOSS OF INTEREST"],
    "NO INTEREST": ["LOSS OF INTEREST"],
    "FEELING SAD": ["SADNESS", "DEPRESSED MOOD"],
    "FEEL SAD": ["SADNESS", "DEPRESSED MOOD"],
    "DOWN ALL THE TIME": ["DEPRESSED MOOD", "SADNESS"],
}

OUT_OF_SCOPE_RESPONSE = (
    "I can help with mental health conditions, symptoms, treatments, and medications. "
    "Please ask a question in that area."
)

FORBIDDEN_COMMUNITY_NAMES = {
    "SYMTOM",
    "SYMPOM",
    "DEMOCRAPHIC",
    "SUBJECT",
    "OBJECT",
    "CONDITION",
    "SYMPTOM",
    "TREATMENT",
    "DEMOGRAPHIC",
    "ENTITY",
    "BEHAVIOR",
    "DEVELOPMENT",
    "SUPPORT",
    "SERVICES",
    "TREATMENTS",
    "SYMPTOMS",
    "MEDICATION",
    "SAFETY_RESOURCE",
}

EXTRACTION_PROMPT = """
You are a medical knowledge graph extractor for mental health literature.
Extract only relations that the text explicitly states. Do not infer or generalize.

The text begins with a [CONTEXT: ...] tag that identifies the topic. Use that cue.

Return JSON with:
  - entities: {id, name, type, description}
  - relations: {source, target, relation, description, strength}

Valid entity types:
  CONDITION, SYMPTOM, TREATMENT, MEDICATION, SAFETY_RESOURCE, DEMOGRAPHIC,
  DIAGNOSTIC_CRITERION, SPECIFIER, RISK_FACTOR, COURSE_FEATURE, PREVALENCE_STATEMENT

Valid relations:
  HAS_SYMPTOM, TREATED_BY, PRESCRIBES, SUITABLE_FOR, CONTRAINDICATED_FOR, URGENT_ACTION,
  HAS_DIAGNOSTIC_CRITERION, HAS_SPECIFIER, HAS_RISK_FACTOR, DIFFERENTIAL_DIAGNOSIS,
  COMORBID_WITH, HAS_COURSE, HAS_PREVALENCE

Use these meanings:
  HAS_DIAGNOSTIC_CRITERION:
    CONDITION -> DIAGNOSTIC_CRITERION when the text states formal DSM-style diagnostic rules,
    thresholds, durations, or required symptom groupings.
  HAS_SPECIFIER:
    CONDITION -> SPECIFIER for named specifiers or subtype qualifiers.
  HAS_RISK_FACTOR:
    CONDITION -> RISK_FACTOR, DEMOGRAPHIC, or CONDITION when the text explicitly names risk factors.
  DIFFERENTIAL_DIAGNOSIS:
    CONDITION -> CONDITION when the text explicitly says another disorder should be considered,
    ruled out, or differentiated.
  COMORBID_WITH:
    CONDITION -> CONDITION when the text explicitly states co-occurrence/comorbidity.
  HAS_COURSE:
    CONDITION -> COURSE_FEATURE for onset, duration, recurrence, remission, chronicity,
    episode pattern, or progression statements.
  HAS_PREVALENCE:
    CONDITION -> PREVALENCE_STATEMENT for prevalence, incidence, sex ratio, age distribution,
    or population frequency statements.

Preferred routing by section heading:
  - "Diagnostic Criteria" -> prefer HAS_DIAGNOSTIC_CRITERION
  - "Specifiers" -> prefer HAS_SPECIFIER
  - "Risk and Prognostic Factors" -> prefer HAS_RISK_FACTOR
  - "Differential Diagnosis" -> prefer DIFFERENTIAL_DIAGNOSIS
  - "Comorbidity" -> prefer COMORBID_WITH
  - "Development and Course" or "Course" -> prefer HAS_COURSE
  - "Prevalence" -> prefer HAS_PREVALENCE

Examples:
  CONDITION -> HAS_COURSE -> ONSET IN THE DEVELOPMENTAL PERIOD
  CONDITION -> HAS_COURSE -> CHRONIC AND FLUCTUATING COURSE
  CONDITION -> COMORBID_WITH -> SPECIFIC LEARNING DISORDER
  CONDITION -> HAS_PREVALENCE -> OVERALL PREVALENCE ABOUT 10 PER 1,000

Important rules:
  - Entity names must be specific and UPPERCASE.
  - HAS_SYMPTOM targets must be SYMPTOM only.
  - Do not use HAS_SYMPTOM for co-occurring disorders, specifiers, prevalence, or course facts.
  - TREATMENT and MEDICATION are never the source of HAS_SYMPTOM.
  - Use DIAGNOSTIC_CRITERION for diagnostic rule phrases, not for every symptom name.
  - Use PREVALENCE_STATEMENT for quantitative or epidemiologic statements, not DEMOGRAPHIC alone.
  - Only use DIFFERENTIAL_DIAGNOSIS and COMORBID_WITH when the passage says so explicitly.
  - If nothing meaningful is present, return {"entities": [], "relations": []}.
  - Output valid JSON only, with no markdown fences.
"""

COMMUNITY_SUMMARY_PROMPT = """
You are summarizing a community inside a mental health knowledge graph.
Write 3-5 factual sentences that explain:
1. The main disorder or topic in the community.
2. The key symptoms, treatments, medications, or demographics represented.
3. How the entities relate to each other.
Do not invent facts and do not mention graph mechanics.
"""


@dataclass
class GraphArtifacts:
    custom_entities: dict[str, EntityNode]
    entity_id_to_node: dict[str, EntityNode]
    custom_relations: list[Relation]
    relation_metadata: dict[tuple[str, str, str], dict[str, Any]]
    entity_descriptions: dict[str, str]
    entity_sources: dict[str, set[str]]
    relation_index: dict[str, dict[str, list[tuple[str, str]]]]
    chunk_citation_map: dict[str, str]
    chunk_payload_map: dict[str, dict[str, Any]]

    @property
    def entity_count(self) -> int:
        return len(self.custom_entities)

    @property
    def relation_count(self) -> int:
        return len(self.custom_relations)


@dataclass
class GraphExtractionStats:
    skipped_chunks: int = 0
    skipped_triplets: int = 0
    type_conflicts: int = 0
    garbage_rejected: int = 0
    semantic_rejected: int = 0
    json_parse_errors: int = 0


GENERIC_ENTITY_NAMES = {
    "DISORDER",
    "CONDITION",
    "CRITERION",
    "SPECIFIER",
    "DIAGNOSTIC CRITERION",
    "PREVALENCE",
    "PREVALENCE STATEMENT",
    "GLOBAL PREVALENCE",
    "NEURODEVELOPMENTAL DISORDER",
    "COMMUNICATION DISORDER",
    "MOOD DISORDER",
    "MENTAL DISORDER",
    "AFFECTIVE DISORDER",
    "COMORBID DISORDER",
    "ANOTHER DISORDER",
    "GENERALIZED DISORDER",
    "UNSPECIFIED DISORDER",
    "SEXUAL DISORDER",
    "LEARNING DISORDER",
    "NEUROLOGICAL DISORDER",
    "NEUROCOGNITIVE DISORDER",
}

LOW_SIGNAL_ENTITY_NAMES = {
    "ABILITY TO FUNCTION",
    "ADVERSE CIRCUMSTANCES",
    "BRAIN STRUCTURE AND FUNCTION",
    "COMMUNICATION SKILLS",
    "DIAGNOSTIC EVALUATION",
    "DISABILITY",
    "GUIDANCE ON COMMUNITY MENTAL HEALTH SERVICES",
    "LIFE SKILLS",
    "MEDICINES",
    "MHGAP HUMANITARIAN INTERVENTION GUIDE",
    "MHGAP PROGRAMME",
    "QUALITYRIGHTS INITIATIVE",
    "SCREENING FOR AUTISM",
    "SOCIAL SKILLS",
    "STIGMA",
    "STRENGTHS",
    "VIRUSES",
    "WHO SPECIAL INITIATIVE FOR MENTAL HEALTH",
}

LOW_SIGNAL_ENTITY_SUBSTRINGS = {
    "MENTAL HEALTH GAP ACTION PROGRAMME",
}

RELATION_SECTION_HINTS = {
    "HAS_SYMPTOM": [
        "diagnostic criteria",
        "diagnostic features",
        "associated features",
        "symptoms",
        "symptom",
        "clinical features",
    ],
    "HAS_DIAGNOSTIC_CRITERION": [
        "diagnostic criteria",
        "criterion",
        "criteria",
    ],
    "HAS_SPECIFIER": [
        "specifier",
        "specifiers",
        "severity",
        "with ",
        "without ",
    ],
    "HAS_RISK_FACTOR": [
        "risk and prognostic factors",
        "risk factor",
        "risk factors",
        "temperamental",
        "genetic and physiological",
        "environmental",
    ],
    "DIFFERENTIAL_DIAGNOSIS": [
        "differential diagnosis",
        "differentiate",
        "distinguish",
        "rule out",
    ],
    "COMORBID_WITH": [
        "comorbidity",
        "co-occur",
        "co occur",
        "comorbid",
        "associated with",
    ],
    "HAS_COURSE": [
        "development and course",
        "course",
        "onset",
        "progression",
        "remission",
        "chronic",
    ],
    "HAS_PREVALENCE": [
        "prevalence",
        "incidence",
        "lifetime prevalence",
        "12-month prevalence",
        "sex ratio",
        "how common",
    ],
}

DSM_REPAIR_TARGETS = [
    {
        "condition": "POST TRAUMATIC STRESS DISORDER",
        "relations": ["HAS_SYMPTOM", "HAS_DIAGNOSTIC_CRITERION", "HAS_PREVALENCE"],
        "preferred_node_ids": [
            "4644ad04-a9fd-4a3e-9b50-55ea4bba261b",
            "6cc5e8d8-a91e-4975-89b8-ac540c2ac02d",
            "3c14ec6e-3424-4bf3-a75d-151b2b05c22e",
            "86b25115-3fc2-4ecd-ac3b-32988a8310a2",
            "bc35f4d9-51fc-4497-967f-044710b722ac",
            "b530983c-1d73-4e1a-b59f-3e178b7bbba8",
        ],
    },
    {
        "condition": "INTELLECTUAL DEVELOPMENTAL DISORDER",
        "relations": ["HAS_PREVALENCE"],
    },
    {
        "condition": "INTELLECTUAL DEVELOPMENTAL DISORDER",
        "relations": ["DIFFERENTIAL_DIAGNOSIS"],
    },
    {
        "condition": "SPEECH SOUND DISORDER",
        "relations": ["DIFFERENTIAL_DIAGNOSIS", "HAS_DIAGNOSTIC_CRITERION", "HAS_COURSE"],
    },
]

CLINICAL_REPAIR_TARGETS = [
    {
        "condition": "DEPRESSION",
        "relations": ["HAS_PREVALENCE"],
    },
    {
        "condition": "GENERALIZED ANXIETY DISORDER",
        "relations": ["HAS_PREVALENCE", "DIFFERENTIAL_DIAGNOSIS"],
    },
    {
        "condition": "POST TRAUMATIC STRESS DISORDER",
        "relations": ["DIFFERENTIAL_DIAGNOSIS"],
    },
    {
        "condition": "AUTISM SPECTRUM DISORDER",
        "relations": ["TREATED_BY"],
    },
]

DEFAULT_REPAIR_TARGETS = [*DSM_REPAIR_TARGETS, *CLINICAL_REPAIR_TARGETS]


def _ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _pickle_dump(path: Path, obj: Any) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def _pickle_load(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_model_name(llm_client: Any) -> str:
    if llm_client is None:
        return "none"
    metadata = getattr(llm_client, "metadata", None)
    if metadata is not None:
        model_name = getattr(metadata, "model_name", None)
        if model_name:
            return str(model_name)
    for attr in ("model", "model_name"):
        value = getattr(llm_client, attr, None)
        if value:
            return str(value)
    return "unknown-model"


def _build_nodes_hash(nodes: list[Any]) -> str:
    digester = hashlib.sha256()
    for node in nodes:
        text = getattr(node, "text", "") or ""
        metadata = getattr(node, "metadata", {}) or {}
        original_text = metadata.get("original_text", text)
        payload = {
            "node_id": getattr(node, "node_id", None),
            "file_name": metadata.get("file_name"),
            "file_path": metadata.get("file_path"),
            "header_path": metadata.get("header_path"),
            "text_sha": hashlib.sha256(str(original_text).encode("utf-8")).hexdigest(),
        }
        digester.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digester.hexdigest()


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "message") and getattr(response.message, "content", None):
        return str(response.message.content)
    if hasattr(response, "text"):
        return str(response.text)
    return str(response)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _load_json_payload(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(text)
    if not cleaned:
        return {"entities": [], "relations": []}
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            return json.loads(candidate)
        raise


def normalize_lookup_text(text: str) -> str:
    text = text.upper()
    text = ABBREVIATION_MAP.get(text, text)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return " ".join(text.split())


def canonicalize(name: str) -> str:
    name = normalize_lookup_text(name)
    name = ABBREVIATION_MAP.get(name, name)
    words = name.split()
    if words:
        words[-1] = LAST_WORD_SINGULAR.get(words[-1], words[-1])
    return " ".join(words)


def derive_entity_name(raw_name: str, raw_type: str, description: str) -> str:
    canon = canonicalize(raw_name)
    normalized_description = " ".join(str(description or "").split())
    if raw_type == "PREVALENCE_STATEMENT":
        if is_placeholder_entity_name(canon, raw_type) and normalized_description:
            description_name = normalize_lookup_text(normalized_description)
            if description_name:
                return description_name[:140].strip()
    if raw_type == "COURSE_FEATURE":
        if canon in {"COURSE FEATURE", "COURSE", "DEVELOPMENT AND COURSE"} and normalized_description:
            description_name = normalize_lookup_text(normalized_description)
            if description_name:
                return description_name[:140].strip()
    return canon


def resolve_type(existing_type: str, new_type: str) -> str:
    if TYPE_PRIORITY.get(existing_type, 0) >= TYPE_PRIORITY.get(new_type, 0):
        return existing_type
    return new_type


def is_garbage_entity(name: str) -> bool:
    words = set(name.upper().split()) - GRAPH_STOP_WORDS
    if not words:
        return True
    if len(name) <= 2:
        return True
    if words.issubset(GARBAGE_FRAGMENTS):
        return True
    if len(words) <= 2 and words & GARBAGE_FRAGMENTS:
        return True
    return False


def is_valid_triplet(subj_type: str, relation: str, obj_type: str) -> bool:
    return (subj_type, relation, obj_type) in VALID_TRIPLET_TYPES


def _parse_strength(value: Any) -> int:
    if isinstance(value, bool):
        return 5
    if isinstance(value, int):
        return max(1, min(10, value))
    if isinstance(value, float):
        return max(1, min(10, int(round(value))))

    normalized = str(value or "").strip().lower()
    if not normalized:
        return 5
    if normalized.isdigit():
        return max(1, min(10, int(normalized)))

    strength_map = {
        "very weak": 2,
        "weak": 3,
        "low": 3,
        "mild": 4,
        "medium": 5,
        "moderate": 5,
        "average": 5,
        "fairly strong": 7,
        "strong": 8,
        "high": 8,
        "very strong": 9,
        "extremely strong": 10,
    }
    if normalized in strength_map:
        return strength_map[normalized]

    match = re.search(r"\d+", normalized)
    if match:
        return max(1, min(10, int(match.group(0))))
    return 5


def build_chunk_citation_map(nodes: list[Any]) -> dict[str, str]:
    citation_map: dict[str, str] = {}
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        citation_map[getattr(node, "node_id")] = citation_from_metadata(metadata)
    return citation_map


def build_chunk_payload_map(nodes: list[Any]) -> dict[str, dict[str, Any]]:
    payload_map: dict[str, dict[str, Any]] = {}
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        chunk_id = str(getattr(node, "node_id", "") or "")
        if not chunk_id:
            continue
        original_text = str(metadata.get("original_text") or getattr(node, "text", "") or "")
        payload_map[chunk_id] = {
            "citation": citation_from_metadata(metadata),
            "text": original_text,
            "section_title": str(metadata.get("section_title", "") or ""),
            "context_tag": str(metadata.get("context_tag", "") or ""),
            "source_label": str(metadata.get("source_label", "") or ""),
            "file_path": str(metadata.get("file_path", "") or ""),
            "header_path": str(metadata.get("header_path", "") or ""),
        }
    return payload_map


def refresh_chunk_citation_map(
    artifacts: GraphArtifacts,
    enriched_nodes: list[Any],
) -> GraphArtifacts:
    refreshed_map = build_chunk_citation_map(enriched_nodes)
    refreshed_payload_map = build_chunk_payload_map(enriched_nodes)
    return GraphArtifacts(
        custom_entities=artifacts.custom_entities,
        entity_id_to_node=artifacts.entity_id_to_node,
        custom_relations=artifacts.custom_relations,
        relation_metadata=artifacts.relation_metadata,
        entity_descriptions=artifacts.entity_descriptions,
        entity_sources=artifacts.entity_sources,
        relation_index=artifacts.relation_index,
        chunk_citation_map=refreshed_map,
        chunk_payload_map=refreshed_payload_map,
    )


def graph_checkpoint_exists(checkpoint_dir: str | Path) -> bool:
    root = Path(checkpoint_dir)
    return all((root / name).exists() for name in GRAPH_REQUIRED_FILES)


def save_graph_checkpoint(
    checkpoint_dir: str | Path,
    artifacts: GraphArtifacts,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    root = _ensure_dir(checkpoint_dir)
    payload = {
        "custom_entities.pkl": artifacts.custom_entities,
        "entity_id_to_node.pkl": artifacts.entity_id_to_node,
        "entity_descriptions.pkl": artifacts.entity_descriptions,
        "entity_sources.pkl": artifacts.entity_sources,
        "custom_relations.pkl": artifacts.custom_relations,
        "relation_metadata.pkl": artifacts.relation_metadata,
        "relation_index.pkl": artifacts.relation_index,
        "chunk_citation_map.pkl": artifacts.chunk_citation_map,
    }
    for filename, obj in payload.items():
        _pickle_dump(root / filename, obj)
    if artifacts.chunk_payload_map:
        _pickle_dump(root / "chunk_payload_map.pkl", artifacts.chunk_payload_map)
    if meta is not None:
        _json_dump(root / GRAPH_META_FILE, meta)


def _normalize_loaded_graph_state(state: dict[str, Any]) -> GraphArtifacts:
    entity_sources = {
        key: set(value) if isinstance(value, list) else set(value or [])
        for key, value in (state.get("entity_sources") or {}).items()
    }
    relation_metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, value in (state.get("relation_metadata") or {}).items():
        normalized = dict(value or {})
        normalized["sources"] = set(normalized.get("sources") or [])
        relation_metadata[key] = normalized

    relation_index = state.get("relation_index") or {}
    for name, directions in relation_index.items():
        out_values = directions.get("out") or []
        in_values = directions.get("in") or []
        relation_index[name] = {
            "out": [tuple(item) for item in out_values],
            "in": [tuple(item) for item in in_values],
        }

    custom_entities = state.get("custom_entities") or {}
    entity_id_to_node = state.get("entity_id_to_node") or {}
    if not entity_id_to_node and custom_entities:
        for entity in custom_entities.values():
            entity_id_to_node[entity.id] = entity

    chunk_payload_map: dict[str, dict[str, Any]] = {}
    for chunk_id, payload in (state.get("chunk_payload_map") or {}).items():
        if isinstance(payload, dict):
            chunk_payload_map[str(chunk_id)] = dict(payload)

    return GraphArtifacts(
        custom_entities=custom_entities,
        entity_id_to_node=entity_id_to_node,
        custom_relations=state.get("custom_relations") or [],
        relation_metadata=relation_metadata,
        entity_descriptions=state.get("entity_descriptions") or {},
        entity_sources=entity_sources,
        relation_index=relation_index,
        chunk_citation_map=state.get("chunk_citation_map") or {},
        chunk_payload_map=chunk_payload_map,
    )


def load_graph_checkpoint(checkpoint_dir: str | Path) -> tuple[GraphArtifacts, dict[str, Any]]:
    root = Path(checkpoint_dir)
    if not graph_checkpoint_exists(root):
        missing = [name for name in GRAPH_REQUIRED_FILES if not (root / name).exists()]
        raise FileNotFoundError(
            f"Missing graph checkpoint files in {root}: {missing}"
        )

    state: dict[str, Any] = {}
    for filename in GRAPH_REQUIRED_FILES:
        state[filename[:-4]] = _pickle_load(root / filename)
    for filename in GRAPH_OPTIONAL_FILES:
        path = root / filename
        if path.exists():
            state[filename[:-4]] = _pickle_load(path)

    meta_path = root / GRAPH_META_FILE
    meta = _json_load(meta_path) if meta_path.exists() else {"legacy_checkpoint": True}
    return _normalize_loaded_graph_state(state), meta


def build_relation_index(
    custom_entities: dict[str, EntityNode],
    custom_relations: list[Relation],
    entity_id_to_node: dict[str, EntityNode],
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    relation_index: dict[str, dict[str, list[tuple[str, str]]]] = {
        name: {"out": [], "in": []} for name in custom_entities
    }
    for relation in custom_relations:
        source_node = entity_id_to_node.get(relation.source_id)
        target_node = entity_id_to_node.get(relation.target_id)
        if source_node is None or target_node is None:
            continue
        relation_index.setdefault(source_node.name, {"out": [], "in": []})
        relation_index.setdefault(target_node.name, {"out": [], "in": []})
        relation_index[source_node.name]["out"].append((relation.label, target_node.name))
        relation_index[target_node.name]["in"].append((relation.label, source_node.name))
    return relation_index


def compact_lookup_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", normalize_lookup_text(text))


def _preferred_entity_name(names: list[str]) -> str:
    return max(
        names,
        key=lambda name: (
            name.count(" "),
            len(name),
            -sum(1 for char in name if not char.isalnum() and char != " "),
            name,
        ),
    )


def merge_equivalent_entities(
    artifacts: GraphArtifacts,
) -> tuple[GraphArtifacts, dict[str, Any]]:
    compact_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, entity in artifacts.custom_entities.items():
        compact_groups[(entity.label, compact_lookup_text(name))].append(name)

    name_remap: dict[str, str] = {}
    merged_groups: list[dict[str, Any]] = []
    for (label, compact_key), names in compact_groups.items():
        if len(names) < 2 or not compact_key:
            continue
        primary = _preferred_entity_name(names)
        aliases = sorted(name for name in names if name != primary)
        if not aliases:
            continue
        merged_groups.append(
            {
                "label": label,
                "primary": primary,
                "aliases": aliases,
            }
        )
        for alias in aliases:
            name_remap[alias] = primary

    if not name_remap:
        return artifacts, {"merged_groups": [], "merged_entities": 0, "merged_relations": 0}

    merged_entities: dict[str, EntityNode] = {}
    merged_entity_descriptions: dict[str, str] = {}
    merged_entity_sources: dict[str, set[str]] = {}

    for original_name, entity in artifacts.custom_entities.items():
        canonical_name = name_remap.get(original_name, original_name)
        if canonical_name not in merged_entities:
            if canonical_name == original_name:
                merged_entity = copy.deepcopy(entity)
            else:
                merged_entity = EntityNode(
                    name=canonical_name,
                    label=entity.label,
                    properties=copy.deepcopy(entity.properties),
                )
            merged_entity.name = canonical_name
            merged_entities[canonical_name] = merged_entity
            merged_entity_descriptions[canonical_name] = artifacts.entity_descriptions.get(original_name, "")
            merged_entity_sources[canonical_name] = set(artifacts.entity_sources.get(original_name, set()))
            continue

        existing = merged_entities[canonical_name]
        resolved_type = resolve_type(existing.label, entity.label)
        existing.label = resolved_type
        existing_description = merged_entity_descriptions.get(canonical_name, "")
        incoming_description = artifacts.entity_descriptions.get(original_name, "")
        if len(incoming_description) > len(existing_description):
            merged_entity_descriptions[canonical_name] = incoming_description
            existing.properties["description"] = incoming_description
        merged_entity_sources[canonical_name].update(artifacts.entity_sources.get(original_name, set()))

    merged_relation_metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    merged_relations = 0
    for relation_key, metadata in artifacts.relation_metadata.items():
        subject_name, relation_label, target_name = relation_key
        merged_key = (
            name_remap.get(subject_name, subject_name),
            relation_label,
            name_remap.get(target_name, target_name),
        )
        if merged_key in merged_relation_metadata:
            merged_relations += 1
            existing = merged_relation_metadata[merged_key]
            existing["strength"] = max(existing.get("strength", 5), metadata.get("strength", 5))
            if len(metadata.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = metadata.get("description", "")
            existing["sources"].update(metadata.get("sources", set()))
        else:
            merged_relation_metadata[merged_key] = {
                "description": metadata.get("description", ""),
                "strength": metadata.get("strength", 5),
                "sources": set(metadata.get("sources", set())),
            }

    merged_entity_id_to_node: dict[str, EntityNode] = {}
    for name, entity in merged_entities.items():
        entity.properties["description"] = merged_entity_descriptions.get(name, "")
        entity.properties["sources"] = sorted(merged_entity_sources.get(name, set()))
        merged_entity_id_to_node[entity.id] = entity

    rebuilt_relations: list[Relation] = []
    for subject_name, relation_label, target_name in merged_relation_metadata:
        subject_entity = merged_entities.get(subject_name)
        target_entity = merged_entities.get(target_name)
        if subject_entity is None or target_entity is None:
            continue
        rebuilt_relations.append(
            Relation(
                source_id=subject_entity.id,
                target_id=target_entity.id,
                label=relation_label,
            )
        )

    rebuilt_index = build_relation_index(
        merged_entities,
        rebuilt_relations,
        merged_entity_id_to_node,
    )
    merged_artifacts = GraphArtifacts(
        custom_entities=merged_entities,
        entity_id_to_node=merged_entity_id_to_node,
        custom_relations=rebuilt_relations,
        relation_metadata=merged_relation_metadata,
        entity_descriptions=merged_entity_descriptions,
        entity_sources=merged_entity_sources,
        relation_index=rebuilt_index,
        chunk_citation_map=artifacts.chunk_citation_map,
        chunk_payload_map=artifacts.chunk_payload_map,
    )
    report = {
        "merged_groups": merged_groups,
        "merged_entities": len(name_remap),
        "merged_relations": merged_relations,
    }
    return merged_artifacts, report


def _is_numericish_name(name: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]+", "", normalize_lookup_text(name))
    if not normalized:
        return True
    return not any(char.isalpha() for char in normalized)


def is_placeholder_entity_name(name: str, label: str) -> bool:
    normalized = normalize_lookup_text(name)
    if normalized in GENERIC_ENTITY_NAMES:
        return True
    if normalized.startswith("DIAGNOSTIC CRITERION"):
        return True
    if label == "PREVALENCE_STATEMENT" and (
        normalized in {"PREVALENCE STATEMENT", "GLOBAL PREVALENCE", "PREVALENCE"}
        or _is_numericish_name(normalized)
    ):
        return True
    if label == "CONDITION" and normalized in {
        "DISORDER",
        "CONDITION",
        "COMORBID DISORDER",
        "MENTAL DISORDER",
        "ANOTHER DISORDER",
        "GENERALIZED DISORDER",
        "UNSPECIFIED DISORDER",
    }:
        return True
    if label == "CONDITION" and _is_numericish_name(normalized):
        return True
    return False


def is_low_signal_entity_name(name: str, label: str, description: str = "") -> bool:
    normalized = normalize_lookup_text(name)
    if normalized in LOW_SIGNAL_ENTITY_NAMES:
        return True
    if any(fragment in normalized for fragment in LOW_SIGNAL_ENTITY_SUBSTRINGS):
        return True
    if label == "TREATMENT" and normalized.startswith("SCREENING FOR "):
        return True
    if label == "TREATMENT" and normalized.endswith(" EVALUATION"):
        return True
    if label == "CONDITION" and normalized in {"VIRUSES"}:
        return True
    if label == "SYMPTOM" and normalized in {"DISABILITY", "STIGMA", "BRAIN STRUCTURE AND FUNCTION"}:
        return True
    if label == "RISK_FACTOR" and normalized in {"ABILITY TO FUNCTION", "ADVERSE CIRCUMSTANCES"}:
        return True
    return False


def _is_strictly_valid_post_cleanup_relation(
    subject_type: str,
    relation: str,
    object_type: str,
) -> bool:
    if relation == "HAS_SYMPTOM":
        return subject_type == "CONDITION" and object_type == "SYMPTOM"
    return is_valid_triplet(subject_type, relation, object_type)


def clean_graph_artifacts(
    artifacts: GraphArtifacts,
) -> tuple[GraphArtifacts, dict[str, Any]]:
    original_entity_count = artifacts.entity_count
    original_relation_count = artifacts.relation_count
    merged_artifacts, merge_report = merge_equivalent_entities(artifacts)
    artifacts = merged_artifacts

    removed_entity_names = {
        name
        for name, entity in artifacts.custom_entities.items()
        if is_placeholder_entity_name(name, entity.label)
        or is_low_signal_entity_name(
            name,
            entity.label,
            artifacts.entity_descriptions.get(name, ""),
        )
    }

    filtered_relation_metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    removed_relations = 0

    for relation_key, metadata in artifacts.relation_metadata.items():
        subject_name, relation_label, target_name = relation_key
        subject_entity = artifacts.custom_entities.get(subject_name)
        target_entity = artifacts.custom_entities.get(target_name)

        if subject_entity is None or target_entity is None:
            removed_relations += 1
            continue
        if subject_name in removed_entity_names or target_name in removed_entity_names:
            removed_relations += 1
            continue
        if not _is_strictly_valid_post_cleanup_relation(
            subject_entity.label,
            relation_label,
            target_entity.label,
        ):
            removed_relations += 1
            continue
        filtered_relation_metadata[relation_key] = {
            **metadata,
            "sources": set(metadata.get("sources", set())),
        }

    referenced_names: set[str] = set()
    for subject_name, _, target_name in filtered_relation_metadata:
        referenced_names.add(subject_name)
        referenced_names.add(target_name)

    kept_entities: dict[str, EntityNode] = {}
    kept_entity_descriptions: dict[str, str] = {}
    kept_entity_sources: dict[str, set[str]] = {}
    kept_entity_id_to_node: dict[str, EntityNode] = {}

    removed_isolated_entities = 0
    for name, entity in artifacts.custom_entities.items():
        if name in removed_entity_names:
            continue
        if name not in referenced_names:
            removed_isolated_entities += 1
            continue
        kept_entities[name] = entity
        kept_entity_descriptions[name] = artifacts.entity_descriptions.get(name, "")
        kept_entity_sources[name] = set(artifacts.entity_sources.get(name, set()))
        kept_entity_id_to_node[entity.id] = entity

    filtered_relations: list[Relation] = []
    for (subject_name, relation_label, target_name), metadata in filtered_relation_metadata.items():
        subject_entity = kept_entities.get(subject_name)
        target_entity = kept_entities.get(target_name)
        if subject_entity is None or target_entity is None:
            removed_relations += 1
            continue
        filtered_relations.append(
            Relation(
                source_id=subject_entity.id,
                target_id=target_entity.id,
                label=relation_label,
            )
        )

    rebuilt_relation_index = build_relation_index(
        kept_entities,
        filtered_relations,
        kept_entity_id_to_node,
    )

    cleaned = GraphArtifacts(
        custom_entities=kept_entities,
        entity_id_to_node=kept_entity_id_to_node,
        custom_relations=filtered_relations,
        relation_metadata=filtered_relation_metadata,
        entity_descriptions=kept_entity_descriptions,
        entity_sources=kept_entity_sources,
        relation_index=rebuilt_relation_index,
        chunk_citation_map=artifacts.chunk_citation_map,
        chunk_payload_map=artifacts.chunk_payload_map,
    )
    report = {
        "original_entities": original_entity_count,
        "cleaned_entities": cleaned.entity_count,
        "original_relations": original_relation_count,
        "cleaned_relations": cleaned.relation_count,
        "alias_merge_report": merge_report,
        "removed_entities": len(removed_entity_names),
        "removed_isolated_entities": removed_isolated_entities,
        "removed_relations": removed_relations,
    }
    return cleaned, report


def clean_graph_checkpoint(
    source_checkpoint_dir: str | Path,
    target_checkpoint_dir: str | Path,
    enriched_nodes: list[Any] | None = None,
) -> tuple[GraphArtifacts, dict[str, Any], dict[str, Any]]:
    artifacts, source_meta = load_graph_checkpoint(source_checkpoint_dir)
    cleaned, report = clean_graph_artifacts(artifacts)
    if enriched_nodes is not None:
        cleaned = refresh_chunk_citation_map(cleaned, enriched_nodes)
    meta = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_dir": str(Path(source_checkpoint_dir)),
        "source_meta": source_meta,
        "cleanup_report": report,
        "prompt_version": source_meta.get("prompt_version"),
        "nodes_hash": source_meta.get("nodes_hash"),
        "node_count": source_meta.get("node_count"),
        "model_name": source_meta.get("model_name"),
    }
    save_graph_checkpoint(target_checkpoint_dir, cleaned, meta=meta)
    return cleaned, meta, report


def _merge_chunk_citation_maps(
    primary_map: dict[str, str],
    secondary_map: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    merged = dict(primary_map)
    collisions: list[str] = []
    for chunk_id, citation in secondary_map.items():
        existing = merged.get(chunk_id)
        if existing is not None and existing != citation:
            collisions.append(chunk_id)
        merged[chunk_id] = citation
    return merged, collisions


def _merge_chunk_payload_maps(
    primary_map: dict[str, dict[str, Any]],
    secondary_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {chunk_id: dict(payload) for chunk_id, payload in primary_map.items()}
    for chunk_id, payload in secondary_map.items():
        merged[chunk_id] = dict(payload)
    return merged


def _meta_lineage_entries(
    checkpoint_dir: str | Path,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    lineage = meta.get("lineage")
    if isinstance(lineage, list) and lineage:
        return [dict(item) for item in lineage]

    entry = {"checkpoint_dir": str(Path(checkpoint_dir))}
    for key in (
        "checkpoint_version",
        "generated_at",
        "prompt_version",
        "repair_prompt_version",
        "model_name",
        "repair_model_name",
        "node_count",
        "nodes_hash",
    ):
        if key in meta:
            entry[key] = meta[key]
    return [entry]


def merge_graph_artifacts(
    primary_artifacts: GraphArtifacts,
    secondary_artifacts: GraphArtifacts,
    *,
    primary_name: str = "primary",
    secondary_name: str = "secondary",
) -> tuple[GraphArtifacts, dict[str, Any]]:
    state = _graph_state_from_artifacts(primary_artifacts)
    custom_entities = state["custom_entities"]
    entity_descriptions = state["entity_descriptions"]
    entity_sources = state["entity_sources"]

    merged_chunk_map, chunk_collisions = _merge_chunk_citation_maps(
        state["chunk_citation_map"],
        secondary_artifacts.chunk_citation_map,
    )
    state["chunk_citation_map"] = merged_chunk_map
    state["chunk_payload_map"] = _merge_chunk_payload_maps(
        state.get("chunk_payload_map", {}),
        secondary_artifacts.chunk_payload_map,
    )

    report = {
        "primary_name": primary_name,
        "secondary_name": secondary_name,
        "primary_entity_count": primary_artifacts.entity_count,
        "secondary_entity_count": secondary_artifacts.entity_count,
        "primary_relation_count": primary_artifacts.relation_count,
        "secondary_relation_count": secondary_artifacts.relation_count,
        "added_entities": 0,
        "overlapping_entities": 0,
        "type_conflicts": 0,
        "type_conflict_entities": [],
        "added_relations": 0,
        "merged_relations": 0,
        "chunk_citation_collisions": len(chunk_collisions),
        "chunk_citation_collision_ids": chunk_collisions[:25],
    }

    for name, incoming_entity in secondary_artifacts.custom_entities.items():
        incoming_description = (
            secondary_artifacts.entity_descriptions.get(name, "")
            or incoming_entity.properties.get("description", "")
        )
        incoming_sources = set(secondary_artifacts.entity_sources.get(name, set()))

        if name not in custom_entities:
            copied_entity = copy.deepcopy(incoming_entity)
            copied_entity.name = name
            copied_entity.properties = copy.deepcopy(getattr(incoming_entity, "properties", {}) or {})
            copied_entity.properties["description"] = incoming_description
            copied_entity.properties["sources"] = sorted(incoming_sources)
            custom_entities[name] = copied_entity
            state["entity_id_to_node"][copied_entity.id] = copied_entity
            entity_descriptions[name] = incoming_description
            entity_sources[name] = incoming_sources
            report["added_entities"] += 1
            continue

        report["overlapping_entities"] += 1
        existing = custom_entities[name]
        existing_description = entity_descriptions.get(name, "") or existing.properties.get("description", "")
        existing_sources = entity_sources.setdefault(name, set())
        existing_sources.update(incoming_sources)
        existing.properties["sources"] = sorted(existing_sources)

        if existing.label != incoming_entity.label:
            report["type_conflicts"] += 1
            report["type_conflict_entities"].append(
                {
                    "name": name,
                    f"{primary_name}_label": existing.label,
                    f"{secondary_name}_label": incoming_entity.label,
                }
            )
            if not existing_description and incoming_description:
                entity_descriptions[name] = incoming_description
                existing.properties["description"] = incoming_description
            continue

        if len(incoming_description) > len(existing_description):
            entity_descriptions[name] = incoming_description
            existing.properties["description"] = incoming_description

    merged_relation_metadata: dict[tuple[str, str, str], dict[str, Any]] = {
        key: {
            "description": value.get("description", ""),
            "strength": value.get("strength", 5),
            "sources": set(value.get("sources", set())),
        }
        for key, value in state["relation_metadata"].items()
    }

    for relation_key, metadata in secondary_artifacts.relation_metadata.items():
        subject_name, relation_label, target_name = relation_key
        if subject_name not in custom_entities or target_name not in custom_entities:
            continue

        incoming_meta = {
            "description": metadata.get("description", ""),
            "strength": metadata.get("strength", 5),
            "sources": set(metadata.get("sources", set())),
        }
        if relation_key in merged_relation_metadata:
            existing_meta = merged_relation_metadata[relation_key]
            existing_meta["strength"] = max(existing_meta.get("strength", 5), incoming_meta["strength"])
            if len(incoming_meta["description"]) > len(existing_meta.get("description", "")):
                existing_meta["description"] = incoming_meta["description"]
            existing_meta["sources"].update(incoming_meta["sources"])
            report["merged_relations"] += 1
            continue

        merged_relation_metadata[relation_key] = incoming_meta
        report["added_relations"] += 1

    merged_entity_id_to_node = {entity.id: entity for entity in custom_entities.values()}
    rebuilt_relations: list[Relation] = []
    for subject_name, relation_label, target_name in merged_relation_metadata:
        subject_entity = custom_entities.get(subject_name)
        target_entity = custom_entities.get(target_name)
        if subject_entity is None or target_entity is None:
            continue
        rebuilt_relations.append(
            Relation(
                source_id=subject_entity.id,
                target_id=target_entity.id,
                label=relation_label,
            )
        )

    merged_artifacts = GraphArtifacts(
        custom_entities=custom_entities,
        entity_id_to_node=merged_entity_id_to_node,
        custom_relations=rebuilt_relations,
        relation_metadata=merged_relation_metadata,
        entity_descriptions=entity_descriptions,
        entity_sources=entity_sources,
        relation_index=build_relation_index(
            custom_entities,
            rebuilt_relations,
            merged_entity_id_to_node,
        ),
        chunk_citation_map=merged_chunk_map,
        chunk_payload_map=state["chunk_payload_map"],
    )
    cleaned_artifacts, cleanup_report = clean_graph_artifacts(merged_artifacts)
    report["pre_cleanup_entities"] = merged_artifacts.entity_count
    report["pre_cleanup_relations"] = merged_artifacts.relation_count
    report["cleanup_report"] = cleanup_report
    report["final_entities"] = cleaned_artifacts.entity_count
    report["final_relations"] = cleaned_artifacts.relation_count
    return cleaned_artifacts, report


def merge_graph_checkpoints(
    primary_checkpoint_dir: str | Path,
    secondary_checkpoint_dir: str | Path,
    target_checkpoint_dir: str | Path,
) -> tuple[GraphArtifacts, dict[str, Any], dict[str, Any]]:
    primary_artifacts, primary_meta = load_graph_checkpoint(primary_checkpoint_dir)
    secondary_artifacts, secondary_meta = load_graph_checkpoint(secondary_checkpoint_dir)
    merged_artifacts, merge_report = merge_graph_artifacts(
        primary_artifacts,
        secondary_artifacts,
        primary_name="primary",
        secondary_name="secondary",
    )

    lineage = _meta_lineage_entries(primary_checkpoint_dir, primary_meta) + _meta_lineage_entries(
        secondary_checkpoint_dir,
        secondary_meta,
    )
    meta = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_checkpoint_dir": str(Path(primary_checkpoint_dir)),
        "secondary_checkpoint_dir": str(Path(secondary_checkpoint_dir)),
        "primary_meta": primary_meta,
        "secondary_meta": secondary_meta,
        "lineage": lineage,
        "merge_report": merge_report,
        "prompt_version": primary_meta.get("prompt_version") or secondary_meta.get("prompt_version"),
        "repair_prompt_version": primary_meta.get("repair_prompt_version")
        or secondary_meta.get("repair_prompt_version"),
        "model_name": primary_meta.get("model_name") or secondary_meta.get("model_name"),
        "repair_model_name": primary_meta.get("repair_model_name")
        or secondary_meta.get("repair_model_name"),
    }
    save_graph_checkpoint(target_checkpoint_dir, merged_artifacts, meta=meta)
    return merged_artifacts, meta, merge_report


def _graph_state_from_artifacts(artifacts: GraphArtifacts) -> dict[str, Any]:
    copied_entities: dict[str, EntityNode] = {}
    for name, entity in artifacts.custom_entities.items():
        copied_entity = copy.deepcopy(entity)
        copied_entity.name = name
        copied_entities[name] = copied_entity

    return {
        "custom_entities": copied_entities,
        "entity_id_to_node": {entity.id: entity for entity in copied_entities.values()},
        "entity_descriptions": {
            name: str(artifacts.entity_descriptions.get(name, ""))
            for name in copied_entities
        },
        "entity_sources": {
            name: set(artifacts.entity_sources.get(name, set()))
            for name in copied_entities
        },
        "custom_relations": [
            Relation(
                source_id=relation.source_id,
                target_id=relation.target_id,
                label=relation.label,
            )
            for relation in artifacts.custom_relations
        ],
        "relation_metadata": {
            key: {
                "description": value.get("description", ""),
                "strength": value.get("strength", 5),
                "sources": set(value.get("sources", set())),
            }
            for key, value in artifacts.relation_metadata.items()
        },
        "relation_index": {
            name: {
                "out": [tuple(item) for item in directions.get("out", [])],
                "in": [tuple(item) for item in directions.get("in", [])],
            }
            for name, directions in artifacts.relation_index.items()
        },
        "chunk_citation_map": dict(artifacts.chunk_citation_map),
        "chunk_payload_map": {
            str(chunk_id): dict(payload)
            for chunk_id, payload in artifacts.chunk_payload_map.items()
        },
        "seen_relations": set(artifacts.relation_metadata.keys()),
    }


def _graph_artifacts_from_state(state: dict[str, Any]) -> GraphArtifacts:
    return GraphArtifacts(
        custom_entities=state["custom_entities"],
        entity_id_to_node=state["entity_id_to_node"],
        custom_relations=state["custom_relations"],
        relation_metadata=state["relation_metadata"],
        entity_descriptions=state["entity_descriptions"],
        entity_sources=state["entity_sources"],
        relation_index=state["relation_index"],
        chunk_citation_map=state["chunk_citation_map"],
        chunk_payload_map=state.get("chunk_payload_map", {}),
    )


def _condition_aliases(condition_name: str, artifacts: GraphArtifacts) -> tuple[set[str], set[str]]:
    compact_name = compact_lookup_text(condition_name)
    normalized_aliases = {
        normalize_lookup_text(condition_name),
        canonicalize(condition_name),
    }
    compact_aliases = {compact_name}

    for name, entity in artifacts.custom_entities.items():
        if entity.label != "CONDITION":
            continue
        if compact_lookup_text(name) != compact_name:
            continue
        normalized_aliases.add(normalize_lookup_text(name))
        compact_aliases.add(compact_lookup_text(name))

    for abbreviation, expanded in ABBREVIATION_MAP.items():
        if compact_lookup_text(expanded) != compact_name:
            continue
        normalized_aliases.add(normalize_lookup_text(abbreviation))
        normalized_aliases.add(normalize_lookup_text(expanded))
        compact_aliases.add(compact_lookup_text(abbreviation))
        compact_aliases.add(compact_lookup_text(expanded))

    return {alias for alias in normalized_aliases if alias}, {alias for alias in compact_aliases if alias}


def _node_repair_score(
    node: Any,
    *,
    normalized_aliases: set[str],
    compact_aliases: set[str],
    relation_labels: list[str],
) -> int:
    metadata = getattr(node, "metadata", {}) or {}
    original_text = str(metadata.get("original_text") or getattr(node, "text", "") or "")
    context_tag = str(metadata.get("context_tag", "") or "")
    section_title = str(metadata.get("section_title", "") or "")

    combined_text = " ".join(part for part in [section_title, context_tag, original_text] if part)
    normalized_text = normalize_lookup_text(combined_text)
    compact_text = compact_lookup_text(combined_text)
    normalized_section = normalize_lookup_text(section_title)
    normalized_context = normalize_lookup_text(context_tag)

    score = 0
    if any(alias and alias in normalized_context for alias in normalized_aliases):
        score += 20
    if any(alias and alias in normalized_text for alias in normalized_aliases):
        score += 18
    if any(alias and alias in compact_text for alias in compact_aliases):
        score += 12

    seen_hints: set[str] = set()
    for relation_label in relation_labels:
        for hint in RELATION_SECTION_HINTS.get(relation_label, []):
            normalized_hint = normalize_lookup_text(hint)
            if not normalized_hint or normalized_hint in seen_hints:
                continue
            seen_hints.add(normalized_hint)
            if normalized_hint in normalized_section:
                score += 12
            elif normalized_hint in normalized_text:
                score += 5
    return score


def select_repair_nodes(
    condition_name: str,
    relation_labels: list[str],
    enriched_nodes: list[Any],
    artifacts: GraphArtifacts,
    *,
    preferred_node_ids: set[str] | None = None,
    top_k: int = 6,
    min_score: int = 18,
) -> list[Any]:
    if preferred_node_ids:
        preferred_nodes = [
            node
            for node in enriched_nodes
            if str(getattr(node, "node_id", "")) in preferred_node_ids
        ]
        if preferred_nodes:
            return preferred_nodes[:top_k]

    normalized_aliases, compact_aliases = _condition_aliases(condition_name, artifacts)
    anchor_indices: list[int] = []
    for index, node in enumerate(enriched_nodes):
        metadata = getattr(node, "metadata", {}) or {}
        original_text = str(metadata.get("original_text") or getattr(node, "text", "") or "")
        context_tag = str(metadata.get("context_tag", "") or "")
        section_title = str(metadata.get("section_title", "") or "")

        normalized_context = normalize_lookup_text(context_tag)
        normalized_section = normalize_lookup_text(section_title)
        normalized_text = normalize_lookup_text(original_text[:1200])
        lead_text = normalize_lookup_text(original_text[:220])

        context_match = any(
            alias and (
                normalized_context.startswith(alias)
                or alias in normalized_context[: max(len(alias) + 20, 80)]
            )
            for alias in normalized_aliases
        )
        section_match = any(alias and normalized_section.startswith(alias) for alias in normalized_aliases)
        lead_match = any(alias and alias in lead_text for alias in normalized_aliases)
        if context_match or section_match or lead_match:
            anchor_indices.append(index)

    primary_cluster: list[int] = []
    if anchor_indices:
        sorted_anchors = sorted(anchor_indices)
        current_cluster = [sorted_anchors[0]]
        clusters: list[list[int]] = []
        for anchor_index in sorted_anchors[1:]:
            if anchor_index - current_cluster[-1] <= 4:
                current_cluster.append(anchor_index)
            else:
                clusters.append(current_cluster)
                current_cluster = [anchor_index]
        clusters.append(current_cluster)
        primary_cluster = max(clusters, key=lambda cluster: (len(cluster), -cluster[0]))

    scored_nodes: list[tuple[int, int, Any]] = []
    for index, node in enumerate(enriched_nodes):
        score = _node_repair_score(
            node,
            normalized_aliases=normalized_aliases,
            compact_aliases=compact_aliases,
            relation_labels=relation_labels,
        )
        if primary_cluster:
            cluster_start = primary_cluster[0]
            cluster_end = primary_cluster[-1]
            if cluster_start <= index <= cluster_end:
                nearest_anchor = 0
            else:
                nearest_anchor = min(abs(index - cluster_start), abs(index - cluster_end))
            if nearest_anchor <= 1:
                score += 18
            elif nearest_anchor <= 3:
                score += 10
            elif nearest_anchor <= 6:
                score += 4
            elif score < (min_score + 10):
                continue
        if score < min_score:
            continue
        scored_nodes.append((score, index, node))

    scored_nodes.sort(key=lambda item: (-item[0], item[1]))
    return [node for _, _, node in scored_nodes[:top_k]]


def _build_repair_prompt(condition_name: str, relation_labels: list[str]) -> str:
    relation_text = ", ".join(relation_labels)
    return f"""
You are repairing a DSM clinical knowledge graph for one target condition.
Target condition: {condition_name}

Use only the supplied excerpts. Do not infer beyond the text.
Return JSON only with:
  - entities: {{id, name, type, description}}
  - relations: {{source, target, relation, description, strength}}

Requirements:
  - Include the target condition as entity id "c0" with type CONDITION and name "{condition_name}".
  - Every returned relation must use "c0" as the source.
  - Allowed relation types for this repair: {relation_text}
  - Only include entities needed for these relations.
  - Entity names must be specific and UPPERCASE.
  - If the excerpts do not support the requested facts, return {{"entities": [{{"id":"c0","name":"{condition_name}","type":"CONDITION","description":""}}], "relations": []}}

Routing rules:
  - HAS_SYMPTOM: extract concrete symptoms or clinical features, not comorbid disorders, prevalence statements, outcomes, or quality-of-life effects.
  - HAS_DIAGNOSTIC_CRITERION: extract formal DSM criteria or criterion bullets.
  - HAS_PREVALENCE: extract quantitative prevalence, incidence, sex ratio, or age-distribution statements as PREVALENCE_STATEMENT entities.
  - DIFFERENTIAL_DIAGNOSIS: extract disorders or conditions explicitly named in differential diagnosis.
  - HAS_COURSE: extract onset, progression, chronicity, remission, or developmental course statements as COURSE_FEATURE entities.
  - TREATED_BY: extract treatment, therapy, service, support, or medication interventions only. Do not extract screening, diagnostic evaluation, referral pathways, initiatives, or guideline/program names as treatments.

Output valid JSON only.
"""


def _apply_extracted_payload_to_state(
    state: dict[str, Any],
    extracted: dict[str, Any],
    *,
    chunk_ids: set[str],
    target_condition: str,
    allowed_relations: set[str],
) -> dict[str, int]:
    local_id_map: dict[str, str] = {}
    added_entities = 0
    added_relations = 0
    updated_relations = 0

    custom_entities = state["custom_entities"]
    entity_id_to_node = state["entity_id_to_node"]
    entity_descriptions = state["entity_descriptions"]
    entity_sources = state["entity_sources"]
    relation_index = state["relation_index"]

    for entity_data in extracted.get("entities", []):
        raw_name = str(entity_data.get("name", "")).strip()
        raw_type = str(entity_data.get("type", "")).strip().upper()
        local_id = str(entity_data.get("id", "")).strip()
        description = str(entity_data.get("description", "")).strip()
        if not local_id or not raw_name or raw_type not in ALLOWED_TYPES:
            continue

        canon = derive_entity_name(raw_name, raw_type, description)
        if compact_lookup_text(canon) == compact_lookup_text(target_condition):
            canon = target_condition
            raw_type = "CONDITION"
        if is_garbage_entity(canon):
            continue

        local_id_map[local_id] = canon
        if canon not in custom_entities:
            new_node = EntityNode(
                name=canon,
                label=raw_type,
                properties={"description": description, "sources": sorted(chunk_ids)},
            )
            custom_entities[canon] = new_node
            entity_id_to_node[new_node.id] = new_node
            entity_descriptions[canon] = description
            entity_sources[canon] = set(chunk_ids)
            relation_index[canon] = {"out": [], "in": []}
            added_entities += 1
        else:
            existing = custom_entities[canon]
            existing.label = resolve_type(existing.label, raw_type)
            if len(description) > len(entity_descriptions.get(canon, "")):
                entity_descriptions[canon] = description
                existing.properties["description"] = description
            entity_sources.setdefault(canon, set()).update(chunk_ids)
            existing.properties["sources"] = sorted(entity_sources[canon])

    if "c0" not in local_id_map:
        local_id_map["c0"] = target_condition

    for relation_data in extracted.get("relations", []):
        source_local = str(relation_data.get("source", "")).strip()
        target_local = str(relation_data.get("target", "")).strip()
        relation_label = str(relation_data.get("relation", "")).strip().upper()
        relation_desc = str(relation_data.get("description", "")).strip()
        relation_strength = _parse_strength(relation_data.get("strength", 5))

        if relation_label not in allowed_relations:
            continue

        subject = local_id_map.get(source_local)
        target = local_id_map.get(target_local)
        if not subject or not target or subject != target_condition:
            continue
        if subject not in custom_entities or target not in custom_entities:
            continue

        subject_type = custom_entities[subject].label
        target_type = custom_entities[target].label
        if not is_valid_triplet(subject_type, relation_label, target_type):
            continue

        relation_key = (subject, relation_label, target)
        if relation_key in state["seen_relations"]:
            existing_meta = state["relation_metadata"].get(relation_key)
            if existing_meta is not None:
                existing_meta["strength"] = max(existing_meta.get("strength", 5), relation_strength)
                if len(relation_desc) > len(existing_meta.get("description", "")):
                    existing_meta["description"] = relation_desc
                existing_meta["sources"].update(chunk_ids)
                updated_relations += 1
            continue

        state["seen_relations"].add(relation_key)
        relation = Relation(
            source_id=custom_entities[subject].id,
            target_id=custom_entities[target].id,
            label=relation_label,
        )
        state["custom_relations"].append(relation)
        state["relation_metadata"][relation_key] = {
            "description": relation_desc,
            "strength": relation_strength,
            "sources": set(chunk_ids),
        }
        relation_index.setdefault(subject, {"out": [], "in": []})
        relation_index.setdefault(target, {"out": [], "in": []})
        relation_index[subject]["out"].append((relation_label, target))
        relation_index[target]["in"].append((relation_label, subject))
        added_relations += 1

    return {
        "added_entities": added_entities,
        "added_relations": added_relations,
        "updated_relations": updated_relations,
    }


def repair_graph_artifacts(
    artifacts: GraphArtifacts,
    enriched_nodes: list[Any],
    llm_client: Any,
    *,
    repair_targets: list[dict[str, Any]] | None = None,
    top_k: int = 6,
    progress_every: int = 1,
) -> tuple[GraphArtifacts, dict[str, Any]]:
    repair_targets = repair_targets or DEFAULT_REPAIR_TARGETS
    state = _graph_state_from_artifacts(artifacts)
    progress = ProgressPrinter(
        label="REPAIR",
        total=len(repair_targets),
        every=max(1, progress_every),
    )
    report_targets: list[dict[str, Any]] = []
    totals = {
        "targets_attempted": len(repair_targets),
        "targets_with_nodes": 0,
        "targets_with_repairs": 0,
        "added_entities": 0,
        "added_relations": 0,
        "updated_relations": 0,
    }

    for index, target in enumerate(repair_targets, start=1):
        condition_name = resolve_entity_name(target["condition"], state["custom_entities"]) or canonicalize(target["condition"])
        relation_labels = [label for label in target.get("relations", []) if label in ALLOWED_RELATIONS]
        selected_nodes = select_repair_nodes(
            condition_name,
            relation_labels,
            enriched_nodes,
            _graph_artifacts_from_state(state),
            preferred_node_ids=set(target.get("preferred_node_ids", []) or []),
            top_k=top_k,
        )
        chunk_ids = {str(getattr(node, "node_id", "")) for node in selected_nodes if getattr(node, "node_id", "")}
        detail = {
            "condition": condition_name,
            "relations": relation_labels,
            "selected_chunks": sorted(chunk_ids),
            "selected_count": len(selected_nodes),
            "added_entities": 0,
            "added_relations": 0,
            "updated_relations": 0,
        }

        if selected_nodes:
            totals["targets_with_nodes"] += 1
            prompt = _build_repair_prompt(condition_name, relation_labels)
            excerpt_blocks = []
            for node in selected_nodes:
                metadata = getattr(node, "metadata", {}) or {}
                citation = metadata.get("citation") or state["chunk_citation_map"].get(getattr(node, "node_id"), "")
                original_text = str(metadata.get("original_text") or getattr(node, "text", "") or "").strip()
                excerpt_blocks.append(f"[{citation}]\n{original_text[:3500]}")
            user_content = (
                f"Repair the graph for {condition_name} using only these excerpts.\n\n"
                + "\n\n".join(excerpt_blocks)
            )
            try:
                messages = [
                    ChatMessage(role="system", content=prompt),
                    ChatMessage(role="user", content=user_content),
                ]
                response = llm_client.chat(messages)
                response_text = _extract_response_text(response).strip()
                try:
                    extracted = _load_json_payload(response_text)
                except json.JSONDecodeError:
                    repair_response = llm_client.chat(
                        [
                            ChatMessage(
                                role="system",
                                content=(
                                    "Convert the following content into valid JSON only. "
                                    "Preserve the same entities and relations, and do not add new facts."
                                ),
                            ),
                            ChatMessage(role="user", content=response_text),
                        ]
                    )
                    extracted = _load_json_payload(_extract_response_text(repair_response).strip())
                applied = _apply_extracted_payload_to_state(
                    state,
                    extracted,
                    chunk_ids=chunk_ids,
                    target_condition=condition_name,
                    allowed_relations=set(relation_labels),
                )
                detail.update(applied)
                totals["added_entities"] += applied["added_entities"]
                totals["added_relations"] += applied["added_relations"]
                totals["updated_relations"] += applied["updated_relations"]
                if applied["added_relations"] or applied["updated_relations"]:
                    totals["targets_with_repairs"] += 1
            except Exception as exc:
                detail["error"] = str(exc)

        report_targets.append(detail)
        progress.update(
            index,
            extra=(
                f"{condition_name} | chunks {detail['selected_count']} | "
                f"+rel {detail['added_relations']} | ~rel {detail['updated_relations']}"
            ),
        )

    repaired_artifacts = _graph_artifacts_from_state(state)
    report = {
        **totals,
        "targets": report_targets,
        "prompt_version": GRAPH_REPAIR_PROMPT_VERSION,
        "model_name": _get_model_name(llm_client),
    }
    return repaired_artifacts, report


def repair_graph_checkpoint(
    source_checkpoint_dir: str | Path,
    target_checkpoint_dir: str | Path,
    enriched_nodes: list[Any],
    llm_client: Any,
    *,
    repair_targets: list[dict[str, Any]] | None = None,
    top_k: int = 6,
    progress_every: int = 1,
) -> tuple[GraphArtifacts, dict[str, Any], dict[str, Any]]:
    artifacts, source_meta = load_graph_checkpoint(source_checkpoint_dir)
    repaired, report = repair_graph_artifacts(
        artifacts,
        enriched_nodes,
        llm_client,
        repair_targets=repair_targets,
        top_k=top_k,
        progress_every=progress_every,
    )
    repaired = refresh_chunk_citation_map(repaired, enriched_nodes)
    meta = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_dir": str(Path(source_checkpoint_dir)),
        "source_meta": source_meta,
        "repair_report": report,
        "prompt_version": source_meta.get("prompt_version"),
        "repair_prompt_version": GRAPH_REPAIR_PROMPT_VERSION,
        "nodes_hash": source_meta.get("nodes_hash"),
        "node_count": source_meta.get("node_count"),
        "model_name": source_meta.get("model_name"),
        "repair_model_name": _get_model_name(llm_client),
    }
    save_graph_checkpoint(target_checkpoint_dir, repaired, meta=meta)
    return repaired, meta, report


def _empty_graph_state(
    chunk_citation_map: dict[str, str],
    chunk_payload_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "custom_entities": {},
        "entity_id_to_node": {},
        "entity_descriptions": {},
        "entity_sources": {},
        "custom_relations": [],
        "relation_metadata": {},
        "relation_index": {},
        "chunk_citation_map": dict(chunk_citation_map),
        "chunk_payload_map": {str(chunk_id): dict(payload) for chunk_id, payload in chunk_payload_map.items()},
        "seen_relations": set(),
        "next_index": 0,
        "stats": GraphExtractionStats(),
    }


def _snapshot_graph_state(root: Path, state: dict[str, Any], meta: dict[str, Any]) -> None:
    serializable = dict(state)
    stats = serializable.get("stats")
    if isinstance(stats, GraphExtractionStats):
        serializable["stats"] = stats.__dict__
    _pickle_dump(root / PARTIAL_STATE_FILE, serializable)
    _json_dump(root / PARTIAL_META_FILE, meta)


def _load_partial_graph_state(
    checkpoint_dir: str | Path,
    *,
    nodes_hash: str,
    model_name: str,
    prompt_version: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    root = Path(checkpoint_dir)
    state_path = root / PARTIAL_STATE_FILE
    meta_path = root / PARTIAL_META_FILE
    if not state_path.exists() or not meta_path.exists():
        return None

    meta = _json_load(meta_path)
    if meta.get("checkpoint_version") != CHECKPOINT_VERSION:
        return None
    if meta.get("nodes_hash") != nodes_hash:
        return None
    if meta.get("model_name") != model_name:
        return None
    if meta.get("prompt_version") != prompt_version:
        return None

    state = _pickle_load(state_path)
    stats = state.get("stats")
    if isinstance(stats, dict):
        state["stats"] = GraphExtractionStats(**stats)
    state["entity_sources"] = {
        key: set(value) if isinstance(value, list) else set(value or [])
        for key, value in (state.get("entity_sources") or {}).items()
    }
    state["seen_relations"] = set(state.get("seen_relations") or [])
    state["chunk_payload_map"] = {
        str(chunk_id): dict(payload)
        for chunk_id, payload in (state.get("chunk_payload_map") or {}).items()
        if isinstance(payload, dict)
    }
    normalized_relation_metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, value in (state.get("relation_metadata") or {}).items():
        normalized = dict(value or {})
        normalized["sources"] = set(normalized.get("sources") or [])
        normalized_relation_metadata[key] = normalized
    state["relation_metadata"] = normalized_relation_metadata
    return state, meta


def extract_graph_from_nodes(
    enriched_nodes: list[Any],
    llm_client: Any,
    *,
    checkpoint_dir: str | Path = "./checkpoints/extraction_v4",
    force_rebuild: bool = False,
    resume: bool = True,
    save_every: int = 25,
    progress_every: int = 25,
    max_nodes: int | None = None,
    prompt_version: str = GRAPH_PROMPT_VERSION,
) -> tuple[GraphArtifacts, dict[str, Any]]:
    root = _ensure_dir(checkpoint_dir)
    if graph_checkpoint_exists(root) and not force_rebuild:
        return load_graph_checkpoint(root)

    nodes_hash = _build_nodes_hash(enriched_nodes)
    model_name = _get_model_name(llm_client)
    chunk_citation_map = build_chunk_citation_map(enriched_nodes)
    chunk_payload_map = build_chunk_payload_map(enriched_nodes)

    if force_rebuild:
        for filename in (PARTIAL_STATE_FILE, PARTIAL_META_FILE, GRAPH_PROGRESS_FILE, GRAPH_META_FILE):
            path = root / filename
            if path.exists():
                path.unlink()

    state = _empty_graph_state(chunk_citation_map, chunk_payload_map)
    progress_path = root / GRAPH_PROGRESS_FILE
    start_idx = 0

    if resume and not force_rebuild:
        partial = _load_partial_graph_state(
            root,
            nodes_hash=nodes_hash,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        if partial is not None:
            state, _ = partial
            start_idx = int(state.get("next_index") or 0)
            print(
                f"Resuming graph extraction from chunk {start_idx + 1}/{len(enriched_nodes)} "
                f"using partial checkpoint in {root}."
            )

    stats = state["stats"]
    if not isinstance(stats, GraphExtractionStats):
        stats = GraphExtractionStats(**dict(stats or {}))
        state["stats"] = stats

    effective_total = min(len(enriched_nodes), max_nodes) if max_nodes else len(enriched_nodes)
    progress_every = max(1, int(progress_every))
    if progress_path.exists() and start_idx == 0 and force_rebuild:
        progress_path.unlink()

    progress = ProgressPrinter(
        label="GRAPH",
        total=effective_total,
        every=progress_every,
        start_count=start_idx,
    )
    if start_idx > 0:
        progress.update(start_idx, extra="resume point", force=True)

    for idx in range(start_idx, effective_total):
        node = enriched_nodes[idx]
        chunk_id = getattr(node, "node_id")
        node_triplets: list[dict[str, Any]] = []

        try:
            system_msg = ChatMessage(role="system", content=EXTRACTION_PROMPT)
            user_msg = ChatMessage(role="user", content=f"Extract from this text:\n\n{node.text}")
            response = llm_client.chat([system_msg, user_msg])
            response_text = _extract_response_text(response).strip()

            if not response_text:
                stats.skipped_chunks += 1
                extracted = {"entities": [], "relations": []}
            else:
                try:
                    extracted = _load_json_payload(response_text)
                except json.JSONDecodeError:
                    stats.json_parse_errors += 1
                    stats.skipped_chunks += 1
                    extracted = {"entities": [], "relations": []}

            raw_entities = extracted.get("entities", [])
            raw_relations = extracted.get("relations", [])
            local_id_map: dict[str, str] = {}

            for entity_data in raw_entities:
                raw_name = str(entity_data.get("name", "")).strip()
                raw_type = str(entity_data.get("type", "")).strip().upper()
                local_id = str(entity_data.get("id", "")).strip()
                description = str(entity_data.get("description", "")).strip()

                if not raw_name or not raw_type or not local_id:
                    continue

                canon = derive_entity_name(raw_name, raw_type, description)
                local_id_map[local_id] = canon

                if raw_type not in ALLOWED_TYPES:
                    stats.skipped_triplets += 1
                    continue
                if is_garbage_entity(canon):
                    stats.garbage_rejected += 1
                    continue

                forbidden_names = ALLOWED_TYPES | ALLOWED_RELATIONS | {"SUBJECT", "OBJECT", "ENTITY"}
                if canon in forbidden_names:
                    stats.skipped_triplets += 1
                    continue

                custom_entities = state["custom_entities"]
                entity_id_to_node = state["entity_id_to_node"]
                entity_descriptions = state["entity_descriptions"]
                entity_sources = state["entity_sources"]
                relation_index = state["relation_index"]

                if canon not in custom_entities:
                    new_node = EntityNode(
                        name=canon,
                        label=raw_type,
                        properties={"description": description, "sources": [chunk_id]},
                    )
                    custom_entities[canon] = new_node
                    entity_id_to_node[new_node.id] = new_node
                    entity_descriptions[canon] = description
                    entity_sources[canon] = {chunk_id}
                    relation_index[canon] = {"out": [], "in": []}
                else:
                    existing = custom_entities[canon]
                    resolved_type = resolve_type(existing.label, raw_type)
                    if resolved_type != existing.label:
                        stats.type_conflicts += 1
                        existing.label = resolved_type
                    if len(description) > len(entity_descriptions.get(canon, "")):
                        entity_descriptions[canon] = description
                        existing.properties["description"] = description
                    entity_sources.setdefault(canon, set()).add(chunk_id)
                    existing.properties["sources"] = sorted(entity_sources[canon])

            for relation_data in raw_relations:
                source_local = str(relation_data.get("source", "")).strip()
                target_local = str(relation_data.get("target", "")).strip()
                relation_label = str(relation_data.get("relation", "")).strip().upper()
                relation_desc = str(relation_data.get("description", "")).strip()
                relation_strength = _parse_strength(relation_data.get("strength", 5))

                subject = local_id_map.get(source_local)
                target = local_id_map.get(target_local)
                if not subject or not target:
                    stats.skipped_triplets += 1
                    continue
                if subject not in state["custom_entities"] or target not in state["custom_entities"]:
                    stats.skipped_triplets += 1
                    continue
                if relation_label not in ALLOWED_RELATIONS:
                    stats.skipped_triplets += 1
                    continue

                subject_type = state["custom_entities"][subject].label
                target_type = state["custom_entities"][target].label
                if not is_valid_triplet(subject_type, relation_label, target_type):
                    stats.semantic_rejected += 1
                    continue

                relation_key = (subject, relation_label, target)
                if relation_key in state["seen_relations"]:
                    existing_meta = state["relation_metadata"].get(relation_key)
                    if existing_meta is not None:
                        existing_meta["strength"] = max(existing_meta.get("strength", 5), relation_strength)
                        existing_meta["sources"].add(chunk_id)
                    continue

                state["seen_relations"].add(relation_key)
                relation = Relation(
                    source_id=state["custom_entities"][subject].id,
                    target_id=state["custom_entities"][target].id,
                    label=relation_label,
                )
                state["custom_relations"].append(relation)
                state["relation_metadata"][relation_key] = {
                    "description": relation_desc,
                    "strength": relation_strength,
                    "sources": {chunk_id},
                }
                state["relation_index"].setdefault(subject, {"out": [], "in": []})
                state["relation_index"].setdefault(target, {"out": [], "in": []})
                state["relation_index"][subject]["out"].append((relation_label, target))
                state["relation_index"][target]["in"].append((relation_label, subject))
                node_triplets.append(
                    {
                        "subj": subject,
                        "subj_type": subject_type,
                        "rel": relation_label,
                        "obj": target,
                        "obj_type": target_type,
                    }
                )
        except Exception as exc:
            stats.skipped_chunks += 1
            print(f"  [graph-extract] chunk {idx + 1} failed: {exc}")

        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"node_idx": idx, "triplets": node_triplets}) + "\n")

        state["next_index"] = idx + 1
        progress.update(
            idx + 1,
            extra=(
                f"chunk {str(chunk_id)[:8]} | "
                f"{len(state['custom_entities'])} entities | {len(state['custom_relations'])} relations"
            ),
        )
        if (idx + 1) % save_every == 0 or idx + 1 == effective_total:
            partial_meta = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "next_index": state["next_index"],
                "nodes_hash": nodes_hash,
                "node_count": effective_total,
                "model_name": model_name,
                "prompt_version": prompt_version,
            }
            _snapshot_graph_state(root, state, partial_meta)
            print(
                f"[GRAPH] checkpoint saved at {idx + 1}/{effective_total} "
                f"-> {len(state['custom_entities'])} entities, {len(state['custom_relations'])} relations",
                flush=True,
            )

    artifacts = GraphArtifacts(
        custom_entities=state["custom_entities"],
        entity_id_to_node=state["entity_id_to_node"],
        custom_relations=state["custom_relations"],
        relation_metadata=state["relation_metadata"],
        entity_descriptions=state["entity_descriptions"],
        entity_sources=state["entity_sources"],
        relation_index=state["relation_index"],
        chunk_citation_map=state["chunk_citation_map"],
        chunk_payload_map=state.get("chunk_payload_map", {}),
    )
    meta = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes_hash": nodes_hash,
        "node_count": effective_total,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "stats": stats.__dict__,
    }
    save_graph_checkpoint(root, artifacts, meta=meta)
    return artifacts, meta


def detect_communities(artifacts: GraphArtifacts) -> dict[int, list[str]]:
    if not artifacts.custom_entities:
        return {}

    try:
        sys.modules.setdefault("numpy.char", importlib.import_module("numpy.core.defchararray"))
    except Exception:
        sys.modules.setdefault("numpy.char", np.char)

    graph = nx.Graph()
    for entity in artifacts.custom_entities.values():
        graph.add_node(entity.id, name=entity.name, label=entity.label)
    for relation in artifacts.custom_relations:
        if relation.source_id in graph and relation.target_id in graph:
            graph.add_edge(relation.source_id, relation.target_id, relation=relation.label)

    if graph.number_of_edges() == 0:
        return {}

    partition = leiden(graph, random_seed=42)
    communities: dict[int, list[str]] = defaultdict(list)
    for node_id, community_id in partition.items():
        communities[int(community_id)].append(node_id)
    return dict(communities)


def save_communities(communities: dict[int, list[str]], path: str | Path) -> None:
    payload = {str(key): value for key, value in communities.items()}
    _json_dump(Path(path), payload)


def load_communities(path: str | Path) -> dict[int, list[str]]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    raw = _json_load(file_path)
    loaded: dict[int, list[str]] = {}
    for key, value in raw.items():
        try:
            cast_key = int(key)
        except (TypeError, ValueError):
            continue
        loaded[cast_key] = list(value or [])
    return loaded


def _community_members(
    member_ids: list[str],
    entity_id_to_node: dict[str, EntityNode],
) -> list[str]:
    members: list[str] = []
    for member_id in member_ids:
        entity = entity_id_to_node.get(member_id)
        if entity is None:
            continue
        if entity.name in FORBIDDEN_COMMUNITY_NAMES:
            continue
        if len(entity.name) <= 3:
            continue
        members.append(f"{entity.name} [{entity.label}]")
    return members


def _community_relations(
    member_ids: list[str],
    custom_relations: list[Relation],
    entity_id_to_node: dict[str, EntityNode],
) -> list[str]:
    community_set = set(member_ids)
    valid_ids = {
        member_id
        for member_id in member_ids
        if member_id in entity_id_to_node
        and entity_id_to_node[member_id].name not in FORBIDDEN_COMMUNITY_NAMES
        and len(entity_id_to_node[member_id].name) > 3
    }
    relations: list[str] = []
    for relation in custom_relations:
        if relation.source_id not in community_set or relation.target_id not in community_set:
            continue
        if relation.source_id not in valid_ids or relation.target_id not in valid_ids:
            continue
        source_name = entity_id_to_node[relation.source_id].name
        target_name = entity_id_to_node[relation.target_id].name
        relations.append(f"({source_name}) -[{relation.label}]-> ({target_name})")
    return relations


def _heuristic_community_summary(
    members: list[str],
    relations: list[str],
) -> str:
    labels = Counter()
    names: list[str] = []
    for member in members:
        if "[" in member and member.endswith("]"):
            name, label = member.rsplit("[", 1)
            names.append(name.strip())
            labels[label[:-1].strip()] += 1
        else:
            names.append(member)
    top_names = ", ".join(names[:5])
    top_labels = ", ".join(f"{label.lower()} x{count}" for label, count in labels.most_common(3))
    if relations:
        relation_counter = Counter()
        for relation in relations:
            match = re.search(r"-\[(.+?)\]->", relation)
            if match:
                relation_counter[match.group(1)] += 1
        top_relations = ", ".join(
            f"{label.lower()} x{count}" for label, count in relation_counter.most_common(3)
        )
    else:
        top_relations = "no internal relations"
    return (
        f"This community centers on {top_names}. "
        f"It is composed mainly of {top_labels or 'mixed entity types'}. "
        f"Key internal patterns include {top_relations}."
    )


def load_or_build_communities(
    artifacts: GraphArtifacts,
    *,
    checkpoint_dir: str | Path,
    force_rebuild: bool = False,
) -> dict[int, list[str]]:
    root = _ensure_dir(checkpoint_dir)
    path = root / COMMUNITIES_FILE
    if path.exists() and not force_rebuild:
        communities = load_communities(path)
        if communities:
            return communities

    communities = detect_communities(artifacts)
    save_communities(communities, path)
    return communities


def load_or_build_community_summaries(
    communities: dict[int, list[str]],
    artifacts: GraphArtifacts,
    *,
    checkpoint_dir: str | Path,
    llm_client: Any = None,
    force_rebuild: bool = False,
    progress_every: int = 1,
) -> dict[int, dict[str, Any]]:
    root = _ensure_dir(checkpoint_dir)
    path = root / COMMUNITY_SUMMARY_FILE
    if path.exists() and not force_rebuild:
        raw = _json_load(path)
        loaded: dict[int, dict[str, Any]] = {}
        for key, value in raw.items():
            try:
                cast_key = int(key)
            except (TypeError, ValueError):
                continue
            loaded[cast_key] = value
        if loaded:
            return loaded

    summaries: dict[int, dict[str, Any]] = {}
    ordered_communities = sorted(communities.items(), key=lambda item: -len(item[1]))
    progress = ProgressPrinter(
        label="SUMMARY",
        total=len(ordered_communities),
        every=max(1, int(progress_every)),
        start_count=0,
    )
    for index, (community_id, member_ids) in enumerate(ordered_communities, start=1):
        members = _community_members(member_ids, artifacts.entity_id_to_node)
        if len(members) < 2:
            progress.update(index, extra=f"community {community_id} skipped", force=False)
            continue
        relations = _community_relations(
            member_ids,
            artifacts.custom_relations,
            artifacts.entity_id_to_node,
        )

        if llm_client is not None:
            system_msg = ChatMessage(role="system", content=COMMUNITY_SUMMARY_PROMPT)
            user_msg = ChatMessage(
                role="user",
                content=(
                    "Summarize this mental health graph community.\n\n"
                    f"ENTITIES:\n" + "\n".join(f"- {member}" for member in members) + "\n\n"
                    f"RELATIONSHIPS:\n" + ("\n".join(f"- {relation}" for relation in relations) or "- none")
                ),
            )
            try:
                response = llm_client.chat([system_msg, user_msg])
                summary = _extract_response_text(response).strip()
            except Exception:
                summary = _heuristic_community_summary(members, relations)
        else:
            summary = _heuristic_community_summary(members, relations)

        summaries[community_id] = {
            "summary": summary,
            "entities": members,
            "relations": relations,
            "size": len(members),
            "prompt_version": COMMUNITY_SUMMARY_PROMPT_VERSION,
        }
        progress.update(index, extra=f"community {community_id} | {len(members)} entities")

    payload = {str(key): value for key, value in summaries.items()}
    _json_dump(path, payload)
    return summaries


def _tokenize_question(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _question_ngrams(question: str, *, max_words: int = 5) -> set[str]:
    tokens = normalize_lookup_text(question).split()
    grams: set[str] = set()
    for start in range(len(tokens)):
        for length in range(1, min(max_words, len(tokens) - start) + 1):
            grams.add(" ".join(tokens[start : start + length]))
    return grams


def _resolve_preferred_entity_name(
    preferred_names: list[str],
    custom_entities: dict[str, EntityNode],
    *,
    label: str | None = None,
) -> str | None:
    for preferred_name in preferred_names:
        resolved = resolve_entity_name(preferred_name, custom_entities, label=label)
        if resolved is not None:
            return resolved
    return None


def _match_symptom_aliases(
    question: str,
    custom_entities: dict[str, EntityNode],
) -> list[str]:
    normalized_question = f" {normalize_lookup_text(question)} "
    matched: list[str] = []
    for phrase, preferred_names in SYMPTOM_PHRASE_ALIASES.items():
        if f" {phrase} " not in normalized_question:
            continue
        resolved = _resolve_preferred_entity_name(preferred_names, custom_entities, label="SYMPTOM")
        if resolved and resolved not in matched:
            matched.append(resolved)
    return matched


def _fuzzy_match_entities(
    question: str,
    custom_entities: dict[str, EntityNode],
    *,
    label: str | None = None,
    max_results: int = 12,
) -> list[str]:
    ngrams = _question_ngrams(question)
    if not ngrams:
        return []

    scored: list[tuple[float, str]] = []
    ordered_entities = sorted(custom_entities.values(), key=lambda entity: len(entity.name), reverse=True)
    for entity in ordered_entities:
        if label and entity.label != label:
            continue

        normalized_name = normalize_lookup_text(entity.name)
        if len(normalized_name) < 5:
            continue
        target_words = len(normalized_name.split())
        threshold = 0.91 if target_words > 1 else 0.88
        best_score = 0.0
        for gram in ngrams:
            gram_words = len(gram.split())
            if abs(gram_words - target_words) > 1:
                continue
            if abs(len(gram) - len(normalized_name)) > max(4, len(normalized_name) // 3):
                continue
            score = SequenceMatcher(None, gram, normalized_name).ratio()
            if score > best_score:
                best_score = score
        if best_score >= threshold:
            scored.append((best_score, entity.name))

    scored.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    matched: list[str] = []
    for _, entity_name in scored:
        if entity_name not in matched:
            matched.append(entity_name)
        if len(matched) >= max_results:
            break
    return matched


def _looks_like_natural_symptom_question(
    question: str,
    custom_entities: dict[str, EntityNode] | None = None,
) -> bool:
    question_lower = question.lower()
    token_set = _tokenize_question(question_lower)
    has_pattern = any(pattern in question_lower for pattern in HUMAN_SYMPTOM_PATTERNS)
    has_follow_up = any(pattern in question_lower for pattern in FOLLOW_UP_PATTERNS)
    has_cues = bool(token_set & HUMAN_SYMPTOM_CUE_WORDS)
    if not ((has_pattern or has_follow_up) and has_cues):
        return False
    if custom_entities is None:
        return True
    if match_entities_in_text(question, custom_entities, label="SYMPTOM"):
        return True
    return True


def classify_query(
    query_text: str,
    custom_entities: dict[str, EntityNode] | None = None,
) -> str:
    safety = assess_safety_risk(query_text)
    if safety["is_crisis"]:
        return "CRISIS"
    query_lower = query_text.lower()
    if _tokenize_question(query_lower) & MENTAL_HEALTH_KEYWORDS:
        return "IN_SCOPE"
    if _looks_like_natural_symptom_question(query_text, custom_entities):
        return "IN_SCOPE"
    if detect_relation_intent(query_text)[0] is not None:
        return "IN_SCOPE"
    if custom_entities:
        if match_entities_in_text(query_text, custom_entities):
            return "IN_SCOPE"
    return "OUT_OF_SCOPE"


def match_entities_in_text(
    question: str,
    custom_entities: dict[str, EntityNode],
    *,
    label: str | None = None,
    allow_fuzzy: bool = True,
) -> list[str]:
    normalized_question = f" {normalize_lookup_text(question)} "
    matched: list[str] = []
    ordered_entities = sorted(custom_entities.values(), key=lambda entity: len(entity.name), reverse=True)

    for entity in ordered_entities:
        if label and entity.label != label:
            continue

        normalized_name = normalize_lookup_text(entity.name)
        if normalized_name and f" {normalized_name} " in normalized_question:
            matched.append(entity.name)
            continue

        for abbreviation, expanded in ABBREVIATION_MAP.items():
            if expanded == normalized_name and f" {abbreviation} " in normalized_question:
                matched.append(entity.name)
                break

    if label in {None, "SYMPTOM"}:
        for entity_name in _match_symptom_aliases(question, custom_entities):
            if entity_name not in matched:
                matched.append(entity_name)

    if allow_fuzzy:
        for entity_name in _fuzzy_match_entities(question, custom_entities, label=label):
            if entity_name not in matched:
                matched.append(entity_name)

    deduped: list[str] = []
    for item in matched:
        if item not in deduped:
            deduped.append(item)
    return deduped


def resolve_entity_name(
    name: str,
    custom_entities: dict[str, EntityNode],
    *,
    label: str | None = None,
) -> str | None:
    if name in custom_entities:
        if label is None or custom_entities[name].label == label:
            return name

    target_canonical = canonicalize(name)
    target_normalized = normalize_lookup_text(name)
    for candidate in custom_entities:
        if label is not None and custom_entities[candidate].label != label:
            continue
        if canonicalize(candidate) == target_canonical:
            return candidate
        if normalize_lookup_text(candidate) == target_normalized:
            return candidate
    fuzzy_matches = _fuzzy_match_entities(name, custom_entities, label=label, max_results=1)
    if fuzzy_matches:
        return fuzzy_matches[0]
    return None


RELATION_QUERY_HINTS = [
    ("diagnostic_criteria", ["diagnostic criteria", "criteria", "criterion"], ["HAS_DIAGNOSTIC_CRITERION"]),
    ("specifiers", ["specifier", "specifiers", "subtype", "subtypes"], ["HAS_SPECIFIER"]),
    ("risk_factors", ["risk factor", "risk factors", "prognostic factor", "prognostic factors"], ["HAS_RISK_FACTOR"]),
    ("differential_diagnosis", ["differential diagnosis", "differentiate", "distinguish"], ["DIFFERENTIAL_DIAGNOSIS"]),
    ("comorbidity", ["comorbidity", "comorbid", "co-occur", "co occur", "cooccur"], ["COMORBID_WITH"]),
    ("course", ["development and course", "course", "onset", "progression"], ["HAS_COURSE"]),
    ("prevalence", ["prevalence", "incidence", "how common", "epidemiology", "sex ratio"], ["HAS_PREVALENCE"]),
    (
        "treatment",
        ["treated", "treatment", "treatments", "medication", "medications", "used for", "help with", "helps with"],
        ["TREATED_BY", "PRESCRIBES"],
    ),
    ("symptoms", ["symptom", "symptoms", "signs"], ["HAS_SYMPTOM"]),
]

LOW_SIGNAL_SYMPTOM_NAMES = {
    "DEATH FROM SUICIDE",
    "FUNCTIONAL IMPAIRMENT",
    "MOOD STATES",
    "PHYSICAL HEALTH PROBLEM",
    "QUALITY OF LIFE",
    "SUICIDAL THOUGHT",
    "SUICIDE ATTEMPTS",
}


def detect_relation_intent(question: str) -> tuple[str | None, list[str] | None]:
    question_lower = question.lower()
    for intent_name, patterns, relation_types in RELATION_QUERY_HINTS:
        if any(pattern in question_lower for pattern in patterns):
            return intent_name, relation_types
    return None, None


def _format_relation_target(
    relation_type: str,
    target: dict[str, Any],
) -> str:
    target_name = target["name"]
    description = (target.get("description") or "").strip()
    strength_text = f" (strength {target['strength']})" if target.get("strength") else ""

    if relation_type in {"HAS_PREVALENCE", "HAS_COURSE"} and description:
        return f"- {description}{strength_text}"
    if relation_type == "HAS_DIAGNOSTIC_CRITERION" and description:
        return f"- {target_name}: {description}{strength_text}"
    if relation_type in {"COMORBID_WITH", "DIFFERENTIAL_DIAGNOSIS", "HAS_SPECIFIER"} and description:
        return f"- {target_name}: {description}{strength_text}"
    if relation_type == "HAS_RISK_FACTOR" and description:
        return f"- {target_name}: {description}{strength_text}"
    return f"- {target_name}{strength_text}"


def _filter_targets_for_display(
    query_type: str,
    relation_type: str,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filtered_targets = [
        target
        for target in targets
        if not is_low_signal_entity_name(
            target["name"],
            str(target.get("type", "") or ""),
            str(target.get("description", "") or ""),
        )
    ]
    if query_type == "symptoms" and relation_type == "HAS_SYMPTOM":
        filtered = [
            target
            for target in filtered_targets
            if normalize_lookup_text(target["name"]) not in LOW_SIGNAL_SYMPTOM_NAMES
        ]
        return filtered
    if query_type == "prevalence" and relation_type == "HAS_PREVALENCE":
        prevalence_only = [
            target
            for target in filtered_targets
            if str(target.get("type", "")).upper() == "PREVALENCE_STATEMENT"
        ]
        return prevalence_only or filtered_targets
    if query_type == "differential_diagnosis" and relation_type == "DIFFERENTIAL_DIAGNOSIS":
        explicit_differentials = [
            target
            for target in filtered_targets
            if "co-occur" not in str(target.get("description", "")).lower()
            and "comorbid" not in str(target.get("description", "")).lower()
        ]
        return explicit_differentials or filtered_targets
    return filtered_targets


def text_search_entities(
    question: str,
    artifacts: GraphArtifacts,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if len(term) > 2 and term not in QUERY_STOPWORDS
    }
    if not query_terms:
        return []

    scored: list[dict[str, Any]] = []
    for name, entity in artifacts.custom_entities.items():
        description = artifacts.entity_descriptions.get(name, "") or entity.properties.get("description", "")
        if is_low_signal_entity_name(name, entity.label, description):
            continue
        corpus = f"{name} {description}".lower()
        score = sum(1 for term in query_terms if term in corpus)
        if score <= 0:
            continue
        scored.append(
            {
                "name": name,
                "type": entity.label,
                "description": description,
                "score": score,
                "sources": sorted(artifacts.entity_sources.get(name, set())),
            }
        )

    scored.sort(key=lambda item: (-item["score"], item["name"]))
    return scored[:limit]


def entity_relation_lookup(
    artifacts: GraphArtifacts,
    entity_name: str,
    *,
    relation_filter: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], set[str]]:
    resolved_name = resolve_entity_name(entity_name, artifacts.custom_entities)
    canonical_name = resolved_name or canonicalize(entity_name)
    outgoing: dict[str, list[dict[str, Any]]] = {}
    incoming: dict[str, list[dict[str, Any]]] = {}
    citations: set[str] = set()

    for relation_label, target_name in artifacts.relation_index.get(canonical_name, {}).get("out", []):
        if relation_filter is not None and relation_label not in relation_filter:
            continue
        target_entity = artifacts.custom_entities.get(target_name)
        if target_entity is None:
            continue
        description = artifacts.entity_descriptions.get(target_name, "") or target_entity.properties.get("description", "")
        if is_low_signal_entity_name(target_name, target_entity.label, description):
            continue
        metadata = artifacts.relation_metadata.get((canonical_name, relation_label, target_name), {})
        citations.update(metadata.get("sources", set()))
        outgoing.setdefault(relation_label, []).append(
            {
                "name": target_name,
                "type": target_entity.label,
                "strength": metadata.get("strength", 5),
                "description": metadata.get("description", ""),
            }
        )

    for relation_label, source_name in artifacts.relation_index.get(canonical_name, {}).get("in", []):
        if relation_filter is not None and relation_label not in relation_filter:
            continue
        source_entity = artifacts.custom_entities.get(source_name)
        if source_entity is None:
            continue
        description = artifacts.entity_descriptions.get(source_name, "") or source_entity.properties.get("description", "")
        if is_low_signal_entity_name(source_name, source_entity.label, description):
            continue
        metadata = artifacts.relation_metadata.get((source_name, relation_label, canonical_name), {})
        citations.update(metadata.get("sources", set()))
        incoming.setdefault(relation_label, []).append(
            {
                "name": source_name,
                "type": source_entity.label,
                "strength": metadata.get("strength", 5),
                "description": metadata.get("description", ""),
            }
        )

    for values in outgoing.values():
        values.sort(key=lambda item: (-int(item.get("strength", 0) or 0), item["name"]))
    for values in incoming.values():
        values.sort(key=lambda item: (-int(item.get("strength", 0) or 0), item["name"]))
    return outgoing, incoming, citations


def forward_lookup(
    artifacts: GraphArtifacts,
    condition_name: str,
    *,
    relation_filter: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    resolved_name = resolve_entity_name(condition_name, artifacts.custom_entities)
    canonical_name = resolved_name or canonicalize(condition_name)
    grouped: dict[str, list[dict[str, Any]]] = {}
    citations: set[str] = set()
    for relation_label, target_name in artifacts.relation_index.get(canonical_name, {}).get("out", []):
        if relation_filter is not None and relation_label not in relation_filter:
            continue
        target_entity = artifacts.custom_entities.get(target_name)
        if target_entity is None:
            continue
        description = artifacts.entity_descriptions.get(target_name, "") or target_entity.properties.get("description", "")
        if is_low_signal_entity_name(target_name, target_entity.label, description):
            continue
        metadata = artifacts.relation_metadata.get((canonical_name, relation_label, target_name), {})
        citations.update(metadata.get("sources", set()))
        grouped.setdefault(relation_label, []).append(
            {
                "name": target_name,
                "type": target_entity.label,
                "strength": metadata.get("strength", 5),
                "description": metadata.get("description", ""),
            }
        )

    for values in grouped.values():
        values.sort(key=lambda item: (-int(item.get("strength", 0) or 0), item["name"]))
    return grouped, citations


def reverse_symptom_lookup(
    symptom_names: list[str],
    artifacts: GraphArtifacts,
) -> tuple[list[tuple[str, list[str]]], set[str]]:
    symptom_set = {
        resolve_entity_name(name, artifacts.custom_entities) or canonicalize(name)
        for name in symptom_names
    }
    condition_matches: dict[str, list[str]] = {}
    citations: set[str] = set()

    for symptom_name in symptom_set:
        if symptom_name not in artifacts.relation_index:
            continue
        for relation_label, source_name in artifacts.relation_index[symptom_name]["in"]:
            if relation_label != "HAS_SYMPTOM":
                continue
            source_entity = artifacts.custom_entities.get(source_name)
            if source_entity is None or source_entity.label != "CONDITION":
                continue
            symptom_entity = artifacts.custom_entities.get(symptom_name)
            symptom_description = artifacts.entity_descriptions.get(symptom_name, "") if symptom_entity else ""
            if symptom_entity and is_low_signal_entity_name(symptom_name, symptom_entity.label, symptom_description):
                continue
            condition_matches.setdefault(source_name, []).append(symptom_name)
            relation_key = (source_name, relation_label, symptom_name)
            metadata = artifacts.relation_metadata.get(relation_key, {})
            citations.update(metadata.get("sources", set()))

    ranked = sorted(
        ((condition, sorted(set(matches))) for condition, matches in condition_matches.items()),
        key=lambda item: (-len(item[1]), item[0]),
    )
    return ranked, citations


def find_shared_and_diverging(
    entity_names: list[str],
    artifacts: GraphArtifacts,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, set[str]], dict[str, dict[str, set[str]]], set[str]]:
    entity_neighbors: dict[str, dict[str, set[str]]] = {}
    citations: set[str] = set()

    for raw_name in entity_names:
        name = resolve_entity_name(raw_name, artifacts.custom_entities) or canonicalize(raw_name)
        if name not in artifacts.custom_entities:
            continue
        if name not in artifacts.relation_index:
            entity_neighbors[name] = {"out": set(), "in": set()}
            continue

        merged_neighbors: dict[str, set[str]] = {}
        for relation_label, neighbor_name in artifacts.relation_index[name]["out"]:
            neighbor_entity = artifacts.custom_entities.get(neighbor_name)
            if neighbor_entity is not None and is_low_signal_entity_name(
                neighbor_name,
                neighbor_entity.label,
                artifacts.entity_descriptions.get(neighbor_name, ""),
            ):
                continue
            merged_neighbors.setdefault(relation_label, set()).add(neighbor_name)
            metadata = artifacts.relation_metadata.get((name, relation_label, neighbor_name), {})
            citations.update(metadata.get("sources", set()))
        for relation_label, neighbor_name in artifacts.relation_index[name]["in"]:
            neighbor_entity = artifacts.custom_entities.get(neighbor_name)
            if neighbor_entity is not None and is_low_signal_entity_name(
                neighbor_name,
                neighbor_entity.label,
                artifacts.entity_descriptions.get(neighbor_name, ""),
            ):
                continue
            merged_neighbors.setdefault(f"<-{relation_label}", set()).add(neighbor_name)
            metadata = artifacts.relation_metadata.get((neighbor_name, relation_label, name), {})
            citations.update(metadata.get("sources", set()))
        entity_neighbors[name] = merged_neighbors

    if len(entity_neighbors) < 2:
        return entity_neighbors, {}, {}, citations

    all_relation_types = set()
    for neighbors in entity_neighbors.values():
        all_relation_types.update(neighbors.keys())

    shared: dict[str, set[str]] = {}
    unique: dict[str, dict[str, set[str]]] = {}
    valid_names = list(entity_neighbors.keys())

    for relation_type in all_relation_types:
        neighbor_sets = [entity_neighbors[name].get(relation_type, set()) for name in valid_names]
        if not neighbor_sets:
            continue
        intersection = neighbor_sets[0].copy()
        for neighbor_set in neighbor_sets[1:]:
            intersection &= neighbor_set
        if intersection:
            shared[relation_type] = intersection

        for idx, name in enumerate(valid_names):
            others_union: set[str] = set()
            for other_idx, neighbor_set in enumerate(neighbor_sets):
                if other_idx != idx:
                    others_union |= neighbor_set
            only_mine = neighbor_sets[idx] - others_union
            if only_mine:
                unique.setdefault(name, {})[relation_type] = only_mine

    return entity_neighbors, shared, unique, citations


def citation_strings(chunk_ids: set[str], chunk_citation_map: dict[str, str], limit: int = 12) -> list[str]:
    resolved = sorted(chunk_citation_map.get(chunk_id, chunk_id) for chunk_id in chunk_ids)
    deduped: list[str] = []
    for citation in resolved:
        if citation not in deduped:
            deduped.append(citation)
        if len(deduped) >= limit:
            break
    return deduped


def source_chunk_payloads(
    chunk_ids: set[str],
    artifacts: GraphArtifacts,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen_payloads: set[tuple[str, str]] = set()

    for chunk_id in sorted(chunk_ids, key=lambda item: (artifacts.chunk_citation_map.get(item, item), item)):
        payload = dict(artifacts.chunk_payload_map.get(chunk_id, {}))
        citation = str(payload.get("citation") or artifacts.chunk_citation_map.get(chunk_id, chunk_id))
        text = str(payload.get("text") or "")
        dedupe_key = (citation, text)
        if dedupe_key in seen_payloads:
            continue
        seen_payloads.add(dedupe_key)
        payloads.append(
            {
                "chunk_id": str(chunk_id),
                "citation": citation,
                "text": text,
                "section_title": str(payload.get("section_title", "") or ""),
                "context_tag": str(payload.get("context_tag", "") or ""),
                "source_label": str(payload.get("source_label", "") or ""),
                "file_path": str(payload.get("file_path", "") or ""),
                "header_path": str(payload.get("header_path", "") or ""),
            }
        )
        if len(payloads) >= limit:
            break
    return payloads


def global_query(question_text: str, community_summaries: dict[int, dict[str, Any]], top_k: int = 3) -> str:
    if not community_summaries:
        return ""

    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", question_text.lower())
        if term not in QUERY_STOPWORDS
    }
    if not query_terms:
        return ""

    scored: list[tuple[int, int, str]] = []
    for community_id, payload in community_summaries.items():
        if isinstance(payload, dict):
            summary = str(payload.get("summary", ""))
        else:
            summary = str(payload)
        summary_lower = summary.lower()
        score = sum(1 for term in query_terms if term in summary_lower)
        if score > 0:
            scored.append((score, int(community_id), summary))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return ""

    sections = [
        f"Community {community_id} (relevance {score}): {summary}"
        for score, community_id, summary in scored[:top_k]
    ]
    return "\n\n".join(sections)


def _fallback_answer(
    question: str,
    query_type: str,
    multipath_result: str,
    local_rows: list[dict[str, Any]],
    global_text: str,
) -> str:
    sections = [f"Question: {question}", f"Query type: {query_type}"]
    if multipath_result:
        sections.append(multipath_result)
    if local_rows:
        local_lines = [
            f"- {row['name']} [{row['type']}]: {row['description'][:180] if row.get('description') else 'No description'}"
            for row in local_rows[:5]
        ]
        sections.append("Relevant entities:\n" + "\n".join(local_lines))
    if global_text:
        sections.append("Community context:\n" + global_text)
    return "\n\n".join(sections)


def hybrid_query(
    question: str,
    artifacts: GraphArtifacts,
    *,
    community_summaries: dict[int, dict[str, Any]] | None = None,
    llm_client: Any = None,
    answer_with_llm: bool = False,
    conversation_context: list[str] | None = None,
) -> dict[str, Any]:
    community_summaries = community_summaries or {}
    conversation_context = [item.strip() for item in (conversation_context or []) if str(item or "").strip()]
    analysis_text = " ".join([*conversation_context, question]) if conversation_context else question
    query_class = classify_query(analysis_text, artifacts.custom_entities)
    safety = assess_safety_risk(question)

    if query_class == "CRISIS":
        return {
            "answer": safety["response"],
            "local_result": "",
            "local_was_useful": False,
            "global_result": "",
            "multipath_result": "",
            "communities_used": [],
            "citations": [],
            "source_chunks": [],
            "safety_resources": safety["resources"],
            "matched_safety_indicators": safety["matched_indicators"],
            "out_of_scope": False,
            "is_crisis": True,
            "query_type": "crisis",
        }

    if query_class == "OUT_OF_SCOPE":
        return {
            "answer": OUT_OF_SCOPE_RESPONSE,
            "local_result": "",
            "local_was_useful": False,
            "global_result": "",
            "multipath_result": "",
            "communities_used": [],
            "citations": [],
            "source_chunks": [],
            "safety_resources": [],
            "matched_safety_indicators": [],
            "out_of_scope": True,
            "is_crisis": False,
            "query_type": "out_of_scope",
        }

    mentioned_entities = match_entities_in_text(analysis_text, artifacts.custom_entities)
    mentioned_conditions = [
        name
        for name in mentioned_entities
        if artifacts.custom_entities.get(name) is not None
        and artifacts.custom_entities[name].label == "CONDITION"
    ]
    mentioned_symptoms = [
        name
        for name in mentioned_entities
        if artifacts.custom_entities.get(name) is not None
        and artifacts.custom_entities[name].label == "SYMPTOM"
    ]
    mentioned_other_entities = [
        name
        for name in mentioned_entities
        if artifacts.custom_entities.get(name) is not None
        and artifacts.custom_entities[name].label not in {"CONDITION", "SYMPTOM"}
    ]
    question_lower = question.lower()
    relation_intent, relation_filter = detect_relation_intent(question)
    comparison_patterns = ["difference between", "compare", "vs", "versus", "distinguish"]
    symptom_patterns = [
        "i have",
        "i feel",
        "i am feeling",
        "i'm feeling",
        "experiencing",
        "suffering from",
        "what could",
        "what might",
        "lost my appetite",
        "can't sleep",
        "cant sleep",
        "trouble sleeping",
    ]
    is_comparison = any(pattern in question_lower for pattern in comparison_patterns)
    is_symptom_query = (
        bool(mentioned_symptoms)
        and (
            any(pattern in question_lower for pattern in symptom_patterns)
            or _looks_like_natural_symptom_question(analysis_text, artifacts.custom_entities)
        )
    )

    multipath_result = ""
    query_type = "general"
    citation_ids: set[str] = set()

    if is_comparison and len(mentioned_conditions) >= 2:
        query_type = "comparison"
        first, second = mentioned_conditions[:2]
        _, shared, unique, relation_citations = find_shared_and_diverging([first, second], artifacts)
        citation_ids.update(relation_citations)
        lines = [f"Comparison: {first} vs {second}"]
        if shared:
            lines.append("Shared features:")
            for relation_type, values in sorted(shared.items()):
                lines.append(f"- {relation_type}: {', '.join(sorted(values))}")
        if unique.get(first):
            lines.append(f"Unique to {first}:")
            for relation_type, values in sorted(unique[first].items()):
                lines.append(f"- {relation_type}: {', '.join(sorted(values))}")
        if unique.get(second):
            lines.append(f"Unique to {second}:")
            for relation_type, values in sorted(unique[second].items()):
                lines.append(f"- {relation_type}: {', '.join(sorted(values))}")
        multipath_result = "\n".join(lines)

    elif is_symptom_query and mentioned_symptoms:
        query_type = "reverse_symptom"
        ranked, relation_citations = reverse_symptom_lookup(mentioned_symptoms, artifacts)
        citation_ids.update(relation_citations)
        if ranked:
            lines = [
                (
                    "Possible matching conditions based on the current symptom set "
                    f"({', '.join(mentioned_symptoms)}). This is not a final diagnosis, "
                    "and additional symptoms can change the ranking:"
                )
            ]
            for index, (condition_name, matches) in enumerate(ranked[:5], start=1):
                lines.append(f"{index}. {condition_name} (matches {len(matches)}: {', '.join(matches)})")
            multipath_result = "\n".join(lines)
        else:
            multipath_result = "No matching conditions were found for those symptoms."

    elif mentioned_conditions:
        query_type = relation_intent or "forward_lookup"
        condition_name = mentioned_conditions[0]
        grouped, relation_citations = forward_lookup(
            artifacts,
            condition_name,
            relation_filter=set(relation_filter) if relation_filter else None,
        )
        citation_ids.update(relation_citations)
        supplemental_criteria: list[dict[str, Any]] = []
        if query_type == "symptoms":
            criteria_grouped, criteria_citations = forward_lookup(
                artifacts,
                condition_name,
                relation_filter={"HAS_DIAGNOSTIC_CRITERION"},
            )
            supplemental_criteria = criteria_grouped.get("HAS_DIAGNOSTIC_CRITERION", [])[:4]
            citation_ids.update(criteria_citations)
        if grouped or supplemental_criteria:
            header_map = {
                "diagnostic_criteria": f"Diagnostic criteria for {condition_name}:",
                "specifiers": f"Specifiers for {condition_name}:",
                "risk_factors": f"Risk factors for {condition_name}:",
                "differential_diagnosis": f"Differential diagnosis for {condition_name}:",
                "comorbidity": f"Common comorbidities of {condition_name}:",
                "course": f"Course information for {condition_name}:",
                "prevalence": f"Prevalence information for {condition_name}:",
                "treatment": f"Treatment information for {condition_name}:",
                "symptoms": f"Symptoms of {condition_name}:",
            }
            lines = [header_map.get(query_type, f"Information about {condition_name}:")]
            for relation_type, targets in sorted(grouped.items()):
                display_targets = _filter_targets_for_display(query_type, relation_type, targets)
                if not display_targets:
                    continue
                if len(grouped) > 1:
                    lines.append(f"{relation_type}:")
                for target in display_targets[:10]:
                    lines.append(_format_relation_target(relation_type, target))
            if supplemental_criteria:
                lines.append("Related diagnostic criteria:")
                for target in supplemental_criteria:
                    lines.append(_format_relation_target("HAS_DIAGNOSTIC_CRITERION", target))
            multipath_result = "\n".join(lines)
        else:
            if relation_intent:
                multipath_result = f"No {relation_intent.replace('_', ' ')} facts were found for {condition_name}."
            else:
                multipath_result = f"No direct graph relationships were found for {condition_name}."

    elif mentioned_other_entities:
        entity_name = mentioned_other_entities[0]
        entity = artifacts.custom_entities.get(entity_name)
        query_type = "entity_lookup"
        outgoing, incoming, relation_citations = entity_relation_lookup(
            artifacts,
            entity_name,
            relation_filter=set(relation_filter) if relation_filter else None,
        )
        citation_ids.update(relation_citations)
        lines: list[str] = []
        if entity is not None and entity.label in {"MEDICATION", "TREATMENT"}:
            query_type = relation_intent or "entity_lookup"
            lines.append(f"How {entity_name} is used:")
            if incoming:
                for relation_type, sources in sorted(incoming.items()):
                    if relation_type in {"TREATED_BY", "PRESCRIBES", "SUITABLE_FOR"}:
                        lines.append(f"{relation_type}:")
                        for source in sources[:10]:
                            lines.append(_format_relation_target(relation_type, source))
            if outgoing:
                for relation_type, targets in sorted(outgoing.items()):
                    lines.append(f"{relation_type}:")
                    for target in targets[:10]:
                        lines.append(_format_relation_target(relation_type, target))
        else:
            lines.append(f"Information about {entity_name}:")
            if incoming:
                lines.append("Incoming relationships:")
                for relation_type, sources in sorted(incoming.items()):
                    lines.append(f"{relation_type}:")
                    for source in sources[:10]:
                        lines.append(_format_relation_target(relation_type, source))
            if outgoing:
                lines.append("Outgoing relationships:")
                for relation_type, targets in sorted(outgoing.items()):
                    lines.append(f"{relation_type}:")
                    for target in targets[:10]:
                        lines.append(_format_relation_target(relation_type, target))
        if len(lines) > 1:
            multipath_result = "\n".join(lines)
        else:
            multipath_result = f"No direct graph relationships were found for {entity_name}."

    include_local_rows = not ((mentioned_conditions or mentioned_other_entities) and multipath_result)
    local_rows = text_search_entities(question, artifacts) if include_local_rows else []
    for row in local_rows:
        citation_ids.update(row.get("sources", []))

    include_global = query_type in {"general", "comparison", "reverse_symptom"}
    global_text = global_query(question, community_summaries, top_k=3) if include_global else ""
    communities_used = re.findall(r"Community (\d+)", global_text) if global_text else []

    citations = citation_strings(citation_ids, artifacts.chunk_citation_map)
    source_chunks = source_chunk_payloads(citation_ids, artifacts)

    if answer_with_llm and llm_client is not None:
        context_parts = []
        if multipath_result:
            context_parts.append(f"GRAPH KNOWLEDGE:\n{multipath_result}")
        if local_rows:
            local_text = "\n".join(
                f"- {row['name']} [{row['type']}]: {row['description'][:220] if row.get('description') else 'No description'}"
                for row in local_rows[:8]
            )
            context_parts.append(f"ENTITY CONTEXT:\n{local_text}")
        if global_text:
            context_parts.append(f"COMMUNITY CONTEXT:\n{global_text}")
        if source_chunks:
            chunk_text = "\n\n".join(
                f"[{chunk['citation']}]\n{chunk['text']}"
                for chunk in source_chunks
                if chunk.get("text")
            )
            if chunk_text:
                context_parts.append(f"SOURCE CHUNKS:\n{chunk_text}")
        if citations:
            context_parts.append("CITATIONS:\n" + "\n".join(f"- {citation}" for citation in citations))

        answer_instructions = [
            "Answer the user's mental health information question using the provided context.",
            "Be factual, concise, and compassionate.",
            "If the information is incomplete, say so clearly.",
        ]
        if query_type == "reverse_symptom":
            answer_instructions.extend(
                [
                    "Treat symptom-based questions as a differential-style response, not a final diagnosis.",
                    "Lead with the possible matching conditions supported by the graph context.",
                    "State clearly that additional symptoms or history can change the ranking.",
                ]
            )
        if conversation_context:
            answer_instructions.append(
                "Use both the current question and the prior session context when interpreting the symptom set."
            )

        synthesis_prompt = (
            " ".join(answer_instructions)
            + "\n\n"
            + f"QUESTION:\n{question}\n\n"
            + (
                f"PRIOR SESSION CONTEXT:\n" + "\n".join(f"- {item}" for item in conversation_context) + "\n\n"
                if conversation_context
                else ""
            )
            + f"CONTEXT:\n{os.linesep.join(context_parts)}\n\nANSWER:"
        )
        try:
            answer = _extract_response_text(llm_client.complete(synthesis_prompt)).strip()
        except Exception as exc:
            answer = _fallback_answer(question, query_type, multipath_result, local_rows, global_text)
            answer += f"\n\nLLM synthesis failed: {exc}"
    else:
        answer = _fallback_answer(question, query_type, multipath_result, local_rows, global_text)

    return {
        "answer": answer,
        "local_result": "\n".join(
            f"{row['name']} [{row['type']}]: {row['description']}" for row in local_rows
        ),
        "local_was_useful": bool(local_rows),
        "global_result": global_text,
        "multipath_result": multipath_result,
        "communities_used": communities_used,
        "citations": citations,
        "source_chunks": source_chunks,
        "safety_resources": [],
        "matched_safety_indicators": [],
        "out_of_scope": False,
        "is_crisis": False,
        "query_type": query_type,
    }


def structural_checks(
    artifacts: GraphArtifacts,
    community_summaries: dict[int, dict[str, Any]] | None = None,
) -> list[tuple[str, bool, str]]:
    community_summaries = community_summaries or {}
    results: list[tuple[str, bool, str]] = [
        ("custom_entities populated", artifacts.entity_count > 0, str(artifacts.entity_count)),
        ("relation_index populated", len(artifacts.relation_index) > 0, str(len(artifacts.relation_index))),
        ("entity_sources populated", len(artifacts.entity_sources) > 0, str(len(artifacts.entity_sources))),
        ("chunk_citation_map populated", len(artifacts.chunk_citation_map) > 0, str(len(artifacts.chunk_citation_map))),
        ("relation_metadata populated", len(artifacts.relation_metadata) > 0, str(len(artifacts.relation_metadata))),
        ("community_summaries loaded", len(community_summaries) > 0, str(len(community_summaries))),
        ("classify_query: crisis", classify_query("I want to kill myself") == "CRISIS", classify_query("I want to kill myself")),
        ("classify_query: informational suicide query", classify_query("What is suicidal behavior disorder?") == "IN_SCOPE", classify_query("What is suicidal behavior disorder?")),
        ("classify_query: out of scope", classify_query("best pizza recipe") == "OUT_OF_SCOPE", classify_query("best pizza recipe")),
        ("classify_query: in scope", classify_query("What is depression?") == "IN_SCOPE", classify_query("What is depression?")),
        (
            "classify_query: medication entity in scope",
            classify_query("What is lithium used for?", artifacts.custom_entities) == "IN_SCOPE",
            classify_query("What is lithium used for?", artifacts.custom_entities),
        ),
    ]

    if "DEPRESSION" in artifacts.relation_index:
        results.append(
            (
                "relation_index has forward edges for DEPRESSION",
                len(artifacts.relation_index["DEPRESSION"]["out"]) > 0,
                str(len(artifacts.relation_index["DEPRESSION"]["out"])),
            )
        )
    else:
        results.append(("DEPRESSION in relation_index", False, "missing"))

    ranked, _ = reverse_symptom_lookup(["FATIGUE"], artifacts)
    results.append(("reverse_symptom_lookup returns results", len(ranked) > 0, str(ranked[:3])))

    if "DEPRESSION" in artifacts.custom_entities and "ANXIETY" in artifacts.custom_entities:
        _, shared, unique, _ = find_shared_and_diverging(["DEPRESSION", "ANXIETY"], artifacts)
        results.append(
            (
                "find_shared_and_diverging returns shared or unique",
                bool(shared) or bool(unique),
                f"shared={len(shared)}, unique={len(unique)}",
            )
        )
    return results


def summarize_check_results(results: list[tuple[str, bool, str]]) -> tuple[int, int]:
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    return passed, failed
