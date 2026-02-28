"""
AgentVault — Confidence Scorer
Heuristic scoring of LLM output reliability.
Detects hedging language, self-contradiction, and vague responses.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .models import ConfidenceFactor, ConfidenceScore

logger = logging.getLogger("agentvault.confidence")


# ---------------------------------------------------------------------------
# Hedging / uncertainty patterns
# ---------------------------------------------------------------------------

HEDGING_PATTERNS = [
    r"\bi think\b",
    r"\bprobably\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bmight\b",
    r"\bcould be\b",
    r"\bnot sure\b",
    r"\bnot certain\b",
    r"\bpossibly\b",
    r"\blikely\b",
    r"\bunlikely\b",
    r"\bseems like\b",
    r"\bappears to\b",
    r"\bI'm not (?:entirely |completely )?(?:sure|certain)\b",
    r"\bit's (?:hard|difficult) to (?:say|tell|know)\b",
    r"\bdon'?t quote me\b",
    r"\btake this with a grain of salt\b",
    r"\bapproximately\b",
    r"\broughly\b",
    r"\bI believe\b",
    r"\bif I (?:recall|remember) correctly\b",
    r"\bas far as I know\b",
    r"\bto (?:the best of )?my knowledge\b",
]

CONTRADICTION_MARKERS = [
    (r"\bis\b", r"\bis not\b"),
    (r"\byes\b", r"\bno\b"),
    (r"\balways\b", r"\bnever\b"),
    (r"\ball\b", r"\bnone\b"),
    (r"\bshould\b", r"\bshould not\b"),
    (r"\bcan\b", r"\bcannot\b"),
    (r"\bwill\b", r"\bwill not\b"),
    (r"\btrue\b", r"\bfalse\b"),
    (r"\bcorrect\b", r"\bincorrect\b"),
    (r"\baccurate\b", r"\binaccurate\b"),
]

VAGUE_INDICATORS = [
    r"\bsome\b",
    r"\bvarious\b",
    r"\bmany\b",
    r"\bseveral\b",
    r"\ba number of\b",
    r"\betc\.?\b",
    r"\band so on\b",
    r"\bthing(?:s)?\b",
    r"\bstuff\b",
    r"\bit depends\b",
    r"\bin general\b",
    r"\btypically\b",
    r"\busually\b",
]

SPECIFIC_INDICATORS = [
    r"\b\d{4}\b",  # years
    r"\b\d+(?:\.\d+)?%\b",  # percentages
    r"\b\d+(?:\.\d+)?\s*(?:MB|GB|KB|ms|s|kg|km|m)\b",  # measurements
    r"\bhttps?://\S+\b",  # URLs
    r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",  # proper nouns
    r"\"\w+\"",  # quoted terms
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",  # formatted numbers
]


class ConfidenceScorer:
    """
    Scores the reliability of LLM outputs using heuristic analysis.
    
    Factors:
    - Hedging language detection (uncertainty markers)
    - Self-contradiction scanning (conflicting statements)
    - Specificity scoring (concrete facts vs vague hand-waving)
    
    Returns 0.0-1.0 confidence score with explanation.
    """

    def __init__(
        self,
        hedging_weight: float = 0.35,
        contradiction_weight: float = 0.35,
        specificity_weight: float = 0.30,
    ) -> None:
        self._hedging_weight = hedging_weight
        self._contradiction_weight = contradiction_weight
        self._specificity_weight = specificity_weight

    def score(self, text: str) -> ConfidenceScore:
        """Score the confidence of a text output."""
        if not text or not text.strip():
            return ConfidenceScore(
                value=0.0,
                factors=[],
                reasoning="Empty response — zero confidence",
            )

        factors = []

        # 1. Hedging detection
        hedging_score = self._score_hedging(text)
        factors.append(hedging_score)

        # 2. Contradiction detection
        contradiction_score = self._score_contradictions(text)
        factors.append(contradiction_score)

        # 3. Specificity scoring
        specificity_score = self._score_specificity(text)
        factors.append(specificity_score)

        # Weighted average
        total = (
            hedging_score.score * self._hedging_weight
            + contradiction_score.score * self._contradiction_weight
            + specificity_score.score * self._specificity_weight
        )

        # Clamp to [0, 1]
        total = max(0.0, min(1.0, total))

        # Build reasoning
        low_factors = [f for f in factors if f.score < 0.5]
        if low_factors:
            reasoning = "Concerns: " + "; ".join(f.detail for f in low_factors)
        else:
            reasoning = "Output appears reliable and specific"

        return ConfidenceScore(
            value=round(total, 3),
            factors=factors,
            reasoning=reasoning,
        )

    def _score_hedging(self, text: str) -> ConfidenceFactor:
        """Detect hedging/uncertainty language. Lower hedging = higher score."""
        text_lower = text.lower()
        matches = []

        for pattern in HEDGING_PATTERNS:
            found = re.findall(pattern, text_lower, re.IGNORECASE)
            matches.extend(found)

        # Normalize by text length (per 100 words)
        word_count = max(len(text.split()), 1)
        hedging_density = len(matches) / (word_count / 100)

        # Score: 0 hedges = 1.0, 5+ hedges per 100 words = 0.0
        score = max(0.0, 1.0 - (hedging_density / 5.0))

        detail = ""
        if matches:
            unique = list(set(matches))[:5]
            detail = f"Found {len(matches)} hedging phrases: {', '.join(unique)}"
        else:
            detail = "No hedging language detected"

        return ConfidenceFactor(name="hedging", score=round(score, 3), detail=detail)

    def _score_contradictions(self, text: str) -> ConfidenceFactor:
        """Detect self-contradictions. No contradictions = higher score."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]

        contradictions_found = 0

        for i, sent_a in enumerate(sentences):
            for sent_b in sentences[i + 1:]:
                for pos_pattern, neg_pattern in CONTRADICTION_MARKERS:
                    pos_in_a = bool(re.search(pos_pattern, sent_a))
                    neg_in_b = bool(re.search(neg_pattern, sent_b))
                    neg_in_a = bool(re.search(neg_pattern, sent_a))
                    pos_in_b = bool(re.search(pos_pattern, sent_b))

                    if (pos_in_a and neg_in_b) or (neg_in_a and pos_in_b):
                        contradictions_found += 1

        # Score: 0 contradictions = 1.0, 3+ = 0.0
        score = max(0.0, 1.0 - (contradictions_found / 3.0))

        if contradictions_found:
            detail = f"Found {contradictions_found} potential self-contradictions"
        else:
            detail = "No self-contradictions detected"

        return ConfidenceFactor(
            name="contradiction", score=round(score, 3), detail=detail
        )

    def _score_specificity(self, text: str) -> ConfidenceFactor:
        """Score how specific/concrete the text is. More specifics = higher score."""
        specific_matches = 0
        vague_matches = 0

        for pattern in SPECIFIC_INDICATORS:
            specific_matches += len(re.findall(pattern, text))

        text_lower = text.lower()
        for pattern in VAGUE_INDICATORS:
            vague_matches += len(re.findall(pattern, text_lower))

        total = specific_matches + vague_matches
        if total == 0:
            score = 0.5  # neutral
            detail = "No strong specificity or vagueness indicators"
        else:
            score = specific_matches / total
            detail = f"{specific_matches} specific and {vague_matches} vague indicators"

        return ConfidenceFactor(
            name="specificity", score=round(score, 3), detail=detail
        )

    @staticmethod
    def is_reliable(score: ConfidenceScore, threshold: float = 0.6) -> bool:
        """Check if a confidence score meets the reliability threshold."""
        return score.value >= threshold
