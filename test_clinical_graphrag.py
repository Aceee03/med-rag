from __future__ import annotations

import os
import unittest

from graphrag_pipeline import classify_query, hybrid_query, load_graph_checkpoint
from retrieval_stack import _prepare_sentence_transformers_environment


class ClinicalGraphRAGRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts, cls.meta = load_graph_checkpoint("./checkpoints/clinical_graph_clean")

    def query(self, question: str) -> dict[str, object]:
        return hybrid_query(question, self.artifacts, community_summaries={})

    def test_sentence_transformers_environment_is_configured(self) -> None:
        _prepare_sentence_transformers_environment()
        self.assertEqual(os.environ.get("USE_TF"), "0")
        self.assertEqual(os.environ.get("TRANSFORMERS_NO_TF"), "1")

    def test_medication_query_is_in_scope_and_grounded(self) -> None:
        self.assertEqual(
            classify_query("What is LITHIUM used for?", self.artifacts.custom_entities),
            "IN_SCOPE",
        )
        result = self.query("What is LITHIUM used for?")
        self.assertFalse(result["out_of_scope"])
        self.assertEqual(result["query_type"], "treatment")
        self.assertIn("BIPOLAR DISORDER", result["answer"])

    def test_citations_are_normalized(self) -> None:
        result = self.query("What are the symptoms of PTSD?")
        citations = result["citations"]
        self.assertTrue(citations)
        for citation in citations:
            self.assertNotIn("\r", citation)
            self.assertNotIn("__", citation)
            self.assertNotIn("intentionally omitted", citation.lower())
        self.assertTrue(any(citation.startswith("NIMH: PTSD") for citation in citations))

    def test_prevalence_queries_return_curated_facts(self) -> None:
        depression = self.query("What is the prevalence of depression?")
        gad = self.query("What is the prevalence of generalized anxiety disorder?")
        self.assertIn("4% of the global population", depression["answer"])
        self.assertIn("2.9% among adults", gad["answer"])

    def test_differential_queries_return_explicit_diagnoses(self) -> None:
        ptsd = self.query("What is the differential diagnosis of PTSD?")
        gad = self.query("What is the differential diagnosis of generalized anxiety disorder?")
        self.assertIn("ACUTE STRESS DISORDER", ptsd["answer"])
        self.assertIn("PANIC DISORDER", gad["answer"])
        self.assertNotIn("co-occur", ptsd["answer"].lower())
        self.assertNotIn("co-occur", gad["answer"].lower())

    def test_autism_treatment_query_excludes_screening_and_evaluation(self) -> None:
        result = self.query("What treatments are available for autism spectrum disorder?")
        answer = result["answer"]
        self.assertIn("BEHAVIORAL THERAPY", answer)
        self.assertIn("PSYCHOSOCIAL INTERVENTIONS", answer)
        self.assertNotIn("SCREENING FOR AUTISM", answer)
        self.assertNotIn("DIAGNOSTIC EVALUATION", answer)

    def test_comparison_query_omits_known_noise_entities(self) -> None:
        result = self.query("What is the difference between depression and bipolar disorder?")
        answer = result["answer"]
        for noisy_term in [
            "ABILITY TO FUNCTION",
            "ADVERSE CIRCUMSTANCES",
            "BRAIN STRUCTURE AND FUNCTION",
            "DISABILITY",
            "STIGMA",
            "VIRUSES",
        ]:
            self.assertNotIn(noisy_term, answer)


if __name__ == "__main__":
    unittest.main()
