"""PII scrubbing for the dataset, through Presidio.

Two passes around aliasing (see chat.clean):
    contact   links, emails, phone numbers, addresses, towns and passwords, before
              aliasing, which would otherwise rewrite names inside them
    names     people the aliases missed, after aliasing, with the pseudonyms allowed

Organisations are counted, not replaced: most are shops, games and brands people
talk about. The ones that identify someone, like an employer, belong in [aliases].

Models download on first use: sv_core_news_sm (spaCy) and gliner_multi_pii-v1.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    GLiNERRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
    UrlRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from understudy.people import fold

LANGUAGE = "sv"
SPACY_MODEL = "sv_core_news_sm"
GLINER_MODEL = "urchade/gliner_multi_pii-v1"
PHONE_REGIONS = ("SE", "US", "GB", "DE")

CONTACT = ["URL", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS", "LOCATION"]
PLACEHOLDERS = {
    "EMAIL_ADDRESS": "[email]",
    "PHONE_NUMBER": "[number]",
    "ADDRESS": "[address]",
    "LOCATION": "[place]",
    "PERSON": "[name]",
}

RE_LINK = re.compile(
    r"^([(\['\"<]*)(?:https?://)?(?:www\.)?([^\s/?#<>'\"]+).*?([.,!?)\]'\">]*)$", re.IGNORECASE | re.DOTALL
)
# key=value forms the model misses; the key stays, so the sentence still reads
RE_PASSWORD = re.compile(r"\b(password|passcode|lösenord|pwd)(\s*[:=]\s*)\S+", re.IGNORECASE)
RE_DIGIT = re.compile(r"\d")
RE_POSTCODE = re.compile(r"[\s,]*\d{3}\s?\d{2}(?!\d)")
RE_JOIN = re.compile(r"[\s,]*(?:\d{3}\s?\d{2}[\s,]*)?")


class FixedLabelGLiNER(GLiNERRecognizer):
    # the stock analyze() adds every requested Presidio entity to the model's labels, for good
    def analyze(self, text, entities, nlp_artifacts=None):
        return [r for r in super().analyze(text, [], nlp_artifacts) if r.entity_type in entities]


def link(text: str) -> str:
    """A link down to its domain; quotes and punctuation the detector swallowed go back."""
    m = RE_LINK.match(text)
    return f"{m.group(1)}[link {m.group(2).lower()}]{m.group(3)}" if m else "[link]"


def merge_places(text: str, results: list) -> list:
    """Overlapping places become one; a street with the postcode and town after it
    ("Storgatan 1, 211 22 Malmö") becomes one address."""
    places = sorted((r for r in results if r.entity_type in ("ADDRESS", "LOCATION")), key=lambda r: (r.start, -r.end))
    merged = []
    for r in places:
        last = merged[-1] if merged else None
        if last and (
            r.start < last.end or (last.entity_type == "ADDRESS" and RE_JOIN.fullmatch(text[last.end : r.start]))
        ):
            last.end = max(last.end, r.end)
            if r.entity_type == "ADDRESS":
                last.entity_type = "ADDRESS"
        else:
            merged.append(r)
    for r in merged:
        # the model finds postcodes unreliably; one right after a street is part of it
        m = RE_POSTCODE.match(text, r.end) if r.entity_type == "ADDRESS" else None
        if m:
            r.end = m.end()
    return [r for r in results if r.entity_type not in ("ADDRESS", "LOCATION")] + merged


def spacy_model() -> None:
    import spacy.util
    from spacy.cli.download import download

    if not spacy.util.is_package(SPACY_MODEL):
        download(SPACY_MODEL)


class Scrubber:
    def __init__(self, allow: set[str]) -> None:
        self.allow = allow  # pseudonyms and personas, folded
        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        spacy_model()
        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": LANGUAGE, "model_name": SPACY_MODEL}],
                "ner_model_configuration": {"model_to_presidio_entity_mapping": {"PRS": "PERSON", "LOC": "LOCATION"}},
            }
        ).create_engine()
        registry = RecognizerRegistry(supported_languages=[LANGUAGE])
        for recognizer in (
            UrlRecognizer(supported_language=LANGUAGE),
            EmailRecognizer(supported_language=LANGUAGE),
            PhoneRecognizer(supported_language=LANGUAGE, supported_regions=PHONE_REGIONS),
            # spaCy also finds lowercase towns GLiNER misses, and countries with them
            SpacyRecognizer(supported_language=LANGUAGE, supported_entities=["PERSON", "LOCATION"]),
            FixedLabelGLiNER(
                supported_language=LANGUAGE,
                model_name=GLINER_MODEL,
                entity_mapping={
                    "street address": "ADDRESS",
                    "city": "LOCATION",
                    "town": "LOCATION",
                    "email": "EMAIL_ADDRESS",
                    "organization": "ORGANIZATION",
                },
            ),
        ):
            registry.add_recognizer(recognizer)
        self.analyzer = AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=[LANGUAGE])
        self.anonymizer = AnonymizerEngine()
        self.operators = {
            entity: OperatorConfig("replace", {"new_value": value}) for entity, value in PLACEHOLDERS.items()
        }
        self.operators["URL"] = OperatorConfig("custom", {"lambda": link})
        self.organizations: Counter[str] = Counter()

    def _replace(self, text: str, results) -> str:
        if not results:
            return text
        return self.anonymizer.anonymize(text, results, self.operators).text

    def contact(self, text: str) -> str:
        """Links, emails, phone numbers, addresses, towns and passwords; organisations counted."""
        if not text:
            return text
        text = RE_PASSWORD.sub(r"\1\2[password]", text)
        results = self.analyzer.analyze(text, LANGUAGE, entities=[*CONTACT, "ORGANIZATION"])
        # a street without a number ("på gatan") identifies no one
        contact = [
            r
            for r in results
            if r.entity_type in CONTACT and (r.entity_type != "ADDRESS" or RE_DIGIT.search(text[r.start : r.end]))
        ]
        contact = merge_places(text, contact)
        for r in results:
            if r.entity_type == "ORGANIZATION" and not any(c.start < r.end and r.start < c.end for c in contact):
                self.organizations[text[r.start : r.end].lower()] += 1
        return self._replace(text, contact)

    def names(self, text: str) -> str:
        """People not covered by the aliases."""
        if not text:
            return text
        # not Presidio's allow_list: its regex mode matches substrings, its exact mode is case-sensitive
        results = [
            r
            for r in self.analyzer.analyze(text, LANGUAGE, entities=["PERSON"])
            if fold(text[r.start : r.end]) not in self.allow
        ]
        return self._replace(text, results)
