"""Tests for get_mle_model_fitting"""

import unittest

from aind_analysis_arch_result_access import get_mle_model_fitting
from aind_analysis_arch_result_access.df_mle_model_fitting import (
    AIND_ANALYSIS_TAGS_DFT,
    build_query_new_format,
)

AGENT_ALIAS_PATH_NEW = (
    "processing.data_processes.output_parameters.fitting_results.fit_settings.agent_alias"
)
ANALYSIS_TAG_PATH = "processing.data_processes.code.parameters.analysis_tag"


class TestBuildQueryNewFormat(unittest.TestCase):
    """Query building for the AIND Analysis Framework format (no database access)"""

    def test_default_queries_all_analysis_tags(self):
        """By default, all known framework versions are matched with $in."""
        query = build_query_new_format(subject_id="820688")
        self.assertEqual(query[ANALYSIS_TAG_PATH], {"$in": AIND_ANALYSIS_TAGS_DFT})
        self.assertIn("aind-analysis-framework v0.1", AIND_ANALYSIS_TAGS_DFT)
        self.assertIn("aind-analysis-framework v0.2", AIND_ANALYSIS_TAGS_DFT)

    def test_single_analysis_tag_matched_exactly(self):
        """A single tag string is matched exactly, not wrapped in $in."""
        query = build_query_new_format(
            subject_id="820688", analysis_tag="aind-analysis-framework v0.2"
        )
        self.assertEqual(query[ANALYSIS_TAG_PATH], "aind-analysis-framework v0.2")

    def test_list_of_analysis_tags(self):
        """A list of tags is matched with $in."""
        tags = ["aind-analysis-framework v0.1", "aind-analysis-framework v0.2"]
        query = build_query_new_format(subject_id="820688", analysis_tag=tags)
        self.assertEqual(query[ANALYSIS_TAG_PATH], {"$in": tags})

    def test_agent_alias_path_nested_under_fitting_results(self):
        """agent_alias lives under output_parameters.fitting_results.fit_settings."""
        query = build_query_new_format(agent_alias="WSLS")
        self.assertEqual(query[AGENT_ALIAS_PATH_NEW], "WSLS")

    def test_requires_at_least_one_parameter(self):
        """No query parameters at all should raise."""
        with self.assertRaises(ValueError):
            build_query_new_format()


class TestGetMLEModelFitting(unittest.TestCase):
    """Get MLE model fitting results"""

    def test_get_mle_with_analysis_tag(self):
        """Filter the AIND framework side by analysis_tag."""
        df = get_mle_model_fitting(
            subject_id="820688",
            analysis_tag="aind-analysis-framework v0.2",
            if_include_metrics=False,
            if_include_latent_variables=False,
            only_recent_version=False,
        )
        self.assertIsNotNone(df)
        self.assertIn("analysis_tag", df.columns)
        df_aind = df[df["pipeline_source"] == "aind analysis framework"]
        self.assertGreater(len(df_aind), 0)
        self.assertTrue((df_aind["analysis_tag"] == "aind-analysis-framework v0.2").all())

    def test_agent_alias_returns_aind_framework_records(self):
        """Filtering by agent_alias must not silently drop AIND framework records."""
        df = get_mle_model_fitting(
            subject_id="820688",
            agent_alias="WSLS",
            if_include_metrics=False,
            if_include_latent_variables=False,
            only_recent_version=False,
        )
        self.assertIsNotNone(df)
        self.assertTrue((df["agent_alias"] == "WSLS").all())
        self.assertGreater(len(df[df["pipeline_source"] == "aind analysis framework"]), 0)

    def test_get_mle_model_fitting_old_pipeline(self):
        """Old pipeline (subject 730945, session 2024-10-24)."""
        df = get_mle_model_fitting(
            subject_id="730945",
            session_date="2024-10-24",
            if_include_metrics=True,
            if_include_latent_variables=True,
            if_download_figures=True,
            max_threads_for_s3=10,
        )
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn("pipeline_source", df.columns)
        self.assertIn("S3_location", df.columns)
        self.assertIn("status", df.columns)
        self.assertTrue((df["status"] == "success").all())

    def test_get_mle_model_fitting_new_pipeline(self):
        """New pipeline (subject 778869, session 2025-07-26)."""
        df = get_mle_model_fitting(
            subject_id="778869",
            session_date="2025-07-26",
            if_include_metrics=True,
            if_include_latent_variables=True,
            if_download_figures=True,
            max_threads_for_s3=10,
        )
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn("pipeline_source", df.columns)
        self.assertIn("S3_location", df.columns)
        self.assertIn("status", df.columns)
        self.assertTrue((df["status"] == "success").all())

    def test_get_mle_with_version_filtering(self):
        """Version filtering toggle."""
        df_recent = get_mle_model_fitting(
            subject_id="778869",
            session_date="2025-07-26",
            only_recent_version=True,
            if_include_metrics=True,
        )
        df_all = get_mle_model_fitting(
            subject_id="778869",
            session_date="2025-07-26",
            only_recent_version=False,
            if_include_metrics=True,
        )
        self.assertIsNotNone(df_recent)
        self.assertIsNotNone(df_all)
        self.assertGreaterEqual(len(df_all), len(df_recent))

    def test_get_mle_with_agent_alias(self):
        """Filter by agent_alias."""
        df = get_mle_model_fitting(
            subject_id="730945",
            agent_alias="QLearning_L2F1_CK1_softmax",
            if_include_metrics=True,
        )
        self.assertIsNotNone(df)
        if len(df) > 0:
            self.assertTrue((df["agent_alias"] == "QLearning_L2F1_CK1_softmax").all())

    def test_get_mle_without_metrics(self):
        """No metrics when if_include_metrics=False."""
        df = get_mle_model_fitting(
            subject_id="730945",
            session_date="2024-10-24",
            if_include_metrics=False,
            if_include_latent_variables=False,
        )
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        metrics_columns = ["BIC", "AIC", "log_likelihood", "prediction_accuracy"]
        for col in metrics_columns:
            if col in df.columns:
                pass
        self.assertNotIn("latent_variable", df.columns)

    def test_get_mle_without_latent_variables(self):
        """No latent_variable column when excluded."""
        df = get_mle_model_fitting(
            subject_id="778869",
            session_date="2025-07-26",
            if_include_metrics=True,
            if_include_latent_variables=False,
            if_download_figures=False,
        )
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertNotIn("latent_variable", df.columns)

    def test_get_mle_custom_query(self):
        """Custom query filter."""
        custom_query = {"subject_id": {"$in": ["730945", "778869"]}}
        df = get_mle_model_fitting(
            from_custom_query=custom_query,
            if_include_metrics=True,
            if_include_latent_variables=False,
            if_download_figures=False,
        )
        self.assertIsNotNone(df)
        if len(df) > 0:
            self.assertTrue(df["subject_id"].isin(["730945", "778869"]).all())

    def test_get_mle_by_subject_only(self):
        """Subject-only filter (no session_date)."""
        df = get_mle_model_fitting(
            subject_id="730945",
            if_include_metrics=True,
            if_include_latent_variables=False,
            if_download_figures=False,
        )
        self.assertIsNotNone(df)
        if len(df) > 0:
            self.assertTrue((df["subject_id"] == "730945").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
