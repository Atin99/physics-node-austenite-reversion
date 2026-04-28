"""Tests for real data integrity and provenance tracking."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from real_data import (
    load_all_experimental, load_by_composition, get_kinetic_curves,
    get_temperature_sweeps, get_study_summary, validate_data_integrity,
    EXPERIMENTAL_STUDIES
)


class TestRealDataIntegrity:
    """Verify all real data passes sanity checks."""

    def test_loads_without_error(self):
        df = load_all_experimental()
        assert len(df) > 0, "No experimental data loaded"

    def test_minimum_data_count(self):
        df = load_all_experimental()
        assert len(df) >= 100, f"Expected ≥100 data points, got {len(df)}"

    def test_minimum_study_count(self):
        assert len(EXPERIMENTAL_STUDIES) >= 15, f"Expected ≥15 studies, got {len(EXPERIMENTAL_STUDIES)}"

    def test_f_RA_in_valid_range(self):
        df = load_all_experimental()
        assert (df['f_RA'] >= 0).all(), "Found negative f_RA values"
        assert (df['f_RA'] <= 1).all(), "Found f_RA > 1"

    def test_temperature_in_valid_range(self):
        df = load_all_experimental()
        assert (df['T_celsius'] >= 0).all(), "T below 0°C"
        assert (df['T_celsius'] <= 1100).all(), "T above 1100°C"

    def test_time_non_negative(self):
        df = load_all_experimental()
        assert (df['t_seconds'] >= 0).all(), "Negative time found"

    def test_all_have_provenance(self):
        df = load_all_experimental()
        assert 'provenance' in df.columns
        assert (df['provenance'] == 'experimental').all()

    def test_all_have_doi(self):
        df = load_all_experimental()
        assert 'doi' in df.columns
        assert df['doi'].notna().all()

    def test_all_have_quality_flag(self):
        df = load_all_experimental()
        valid_flags = {'table', 'text_reported', 'digitized_figure', 'user_provided'}
        assert df['data_quality'].isin(valid_flags).all()

    def test_all_have_source_ref(self):
        df = load_all_experimental()
        assert 'source_ref' in df.columns
        assert df['source_ref'].notna().all()

    def test_validation_passes(self):
        result = validate_data_integrity()
        assert result['valid'], f"Validation failed: {result['issues']}"


class TestGibbs2011:
    """Verify the Gibbs 2011 data specifically — these are exact table values."""

    def test_exists(self):
        df = load_by_composition(Mn_range=(7.0, 7.2), C_range=(0.09, 0.11))
        assert len(df) >= 5

    def test_exact_values(self):
        """These are from Table III — they must be exact."""
        df = load_all_experimental()
        gibbs = df[df['study_id'] == 'gibbs_2011']
        assert len(gibbs) == 5

        expected = {575: 23.0, 600: 34.3, 625: 42.8, 650: 43.5, 675: 1.4}
        for _, row in gibbs.iterrows():
            T = row['T_celsius']
            assert T in expected, f"Unexpected temperature {T}"
            assert abs(row['f_RA_pct'] - expected[T]) < 0.01, \
                f"At {T}°C: expected {expected[T]}%, got {row['f_RA_pct']}%"

    def test_peak_at_650(self):
        """RA should peak at 650°C for Fe-7.1Mn-0.1C."""
        df = load_all_experimental()
        gibbs = df[df['study_id'] == 'gibbs_2011']
        peak_T = gibbs.loc[gibbs['f_RA'].idxmax(), 'T_celsius']
        assert peak_T == 650, f"Peak at {peak_T}°C, expected 650°C"

    def test_drop_at_675(self):
        """RA should drop sharply at 675°C (above optimal)."""
        df = load_all_experimental()
        gibbs = df[df['study_id'] == 'gibbs_2011']
        ra_650 = gibbs[gibbs['T_celsius'] == 650]['f_RA'].values[0]
        ra_675 = gibbs[gibbs['T_celsius'] == 675]['f_RA'].values[0]
        assert ra_675 < ra_650 * 0.1, "675°C RA should be <<< 650°C RA"


class TestProvenance:
    """Verify provenance tracking works correctly."""

    def test_all_experimental(self):
        df = load_all_experimental()
        assert (df['provenance'] == 'experimental').all()

    def test_filter_by_composition(self):
        df = load_by_composition(Mn_range=(4.0, 6.0))
        assert len(df) > 0
        assert (df['Mn'] >= 4.0).all()
        assert (df['Mn'] <= 6.0).all()

    def test_kinetic_curves_exist(self):
        curves = get_kinetic_curves()
        # Luo 2011 should have a kinetic curve at 650°C
        assert len(curves) > 0, "No kinetic curves found"

    def test_temperature_sweeps_exist(self):
        sweeps = get_temperature_sweeps()
        # Gibbs 2011 has a T-sweep at 604800s
        assert len(sweeps) > 0, "No temperature sweeps found"


class TestStudySummary:
    def test_prints_without_error(self):
        summary = get_study_summary()
        assert "EXPERIMENTAL DATA SUMMARY" in summary
        assert "TOTAL" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
