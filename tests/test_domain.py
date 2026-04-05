"""
Unit tests for the current domain-layer APIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.error_detection import ErrorType, detect_errors
from domain.learning_logic import compute_accuracy, should_explain
from domain.phoneme_alignment import build_word_alignment
from domain.severity_scoring import Severity, annotate_errors, score_severity


class TestPhonemeAlignment:
    def test_build_word_alignment_exposes_sequence(self):
        alignment = build_word_alignment(
            word="think",
            start=0.1,
            end=0.8,
            phoneme_data=[
                {"phoneme": "TH", "start": 0.1, "end": 0.2, "confidence": 0.9},
                {"phoneme": "IH", "start": 0.2, "end": 0.4, "confidence": 0.8},
            ],
        )

        assert alignment.word == "think"
        assert alignment.phoneme_sequence == ["TH", "IH"]
        assert alignment.average_confidence == pytest.approx(0.85)

    def test_average_confidence_empty_alignment(self):
        alignment = build_word_alignment("test", 0.0, 0.0, [])
        assert alignment.phoneme_sequence == []
        assert alignment.average_confidence == 0.0


class TestErrorDetection:
    def test_perfect_match_has_no_errors(self):
        errors = detect_errors("think", ["TH", "IH", "NG", "K"], ["TH", "IH", "NG", "K"])
        assert errors == []

    def test_substitution_detected(self):
        errors = detect_errors("think", ["TH", "IH", "NG", "K"], ["T", "IH", "NG", "K"])
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.SUBSTITUTION
        assert errors[0].expected_phoneme == "TH"
        assert errors[0].detected_phoneme == "T"

    def test_deletion_detected(self):
        errors = detect_errors("think", ["TH", "IH", "NG", "K"], ["IH", "NG", "K"])
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.DELETION
        assert errors[0].expected_phoneme == "TH"
        assert errors[0].detected_phoneme is None

    def test_insertion_detected(self):
        errors = detect_errors("bed", ["B", "EH", "D"], ["B", "EH", "D", "Z"])
        assert len(errors) == 1
        assert errors[0].error_type == ErrorType.INSERTION
        assert errors[0].expected_phoneme is None
        assert errors[0].detected_phoneme == "Z"


class TestLearningLogic:
    def test_compute_accuracy_perfect(self):
        assert compute_accuracy(["TH", "IH", "NG", "K"], ["TH", "IH", "NG", "K"]) == 1.0

    def test_compute_accuracy_partial_credit(self):
        score = compute_accuracy(["DH", "IY", "Z"], ["D", "IH", "Z"])
        assert 0.0 < score < 1.0

    def test_should_explain_uses_threshold(self):
        errors = [
            {"severity": "minor"},
            {"severity": "moderate"},
        ]
        assert should_explain(errors, severity_threshold="moderate") is True
        assert should_explain(errors, severity_threshold="severe") is False


class TestSeverityScoring:
    def test_similar_high_confidence_substitution_is_minor(self):
        error = detect_errors("think", ["TH"], ["DH"])[0]
        assert score_severity(error, confidence=0.9) == Severity.MINOR

    def test_low_confidence_substitution_is_severe(self):
        error = detect_errors("these", ["DH"], ["B"])[0]
        assert score_severity(error, confidence=0.1) == Severity.SEVERE

    def test_deletion_is_at_least_moderate(self):
        error = detect_errors("think", ["TH"], [])[0]
        assert score_severity(error, confidence=0.8) in (Severity.MODERATE, Severity.SEVERE)

    def test_annotate_errors_adds_confidence_and_severity(self):
        errors = detect_errors("think", ["TH"], ["T"])
        annotated = annotate_errors(errors, [0.25])

        assert len(annotated) == 1
        assert annotated[0]["confidence"] == 0.25
        assert annotated[0]["severity"] in {"minor", "moderate", "severe"}
