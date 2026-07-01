"""
tests/test_ml_pipeline.py — Regression test suite for the ML pipeline.

Run with:
    pytest tests/test_ml_pipeline.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pytest
import numpy as np
import pandas as pd
import joblib

DB_PATH    = "data/athlete.db"
MODEL_PATH = "ml/injury_model.pkl"


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def features_df():
    from ml.injury_model import build_features
    return build_features(DB_PATH)


@pytest.fixture(scope="module")
def advisor():
    from ml.performance_advisor import PerformanceAdvisor
    return PerformanceAdvisor(db_path=DB_PATH)


# ── Feature engineering tests ──────────────────────────────────────────────────
class TestFeatureEngineering:

    def test_engineered_columns_present(self, features_df):
        from ml.injury_model import ENGINEERED_FEATURES
        for col in ENGINEERED_FEATURES:
            assert col in features_df.columns, f"Colonne manquante : {col}"

    def test_no_infinite_values(self, features_df):
        from ml.injury_model import FEATURE_COLUMNS
        num = features_df[FEATURE_COLUMNS].select_dtypes(include=[np.number])
        assert not np.isinf(num.values).any(), "Valeurs infinies détectées"

    def test_acwr_squared_consistency(self, features_df):
        diff = (features_df["acwr_squared"] - features_df["acwr"]**2).abs()
        assert diff.max() < 1e-6

    def test_fatigue_trend_no_lookahead(self, features_df):
        """fatigue_trend_7d must use shift(1) — no current-day leakage."""
        sample = features_df[features_df["player_id"] == features_df["player_id"].iloc[0]]
        sample = sample.sort_values("date")
        if len(sample) > 1:
            # trend at row i should not equal raw fatigue at row i (generally)
            mismatches = (sample["fatigue_trend_7d"] != sample["fatigue"]).sum()
            assert mismatches > 0, "Le trend semble identique à fatigue brute (fuite possible)"


# ── Model tests ────────────────────────────────────────────────────────────────
class TestInjuryModel:

    def test_model_loads(self, model):
        assert model is not None
        assert hasattr(model, "threshold_")

    def test_threshold_in_valid_range(self, model):
        assert 0.0 < model.threshold_ < 1.0

    def test_predict_proba_shape(self, model, features_df):
        from ml.injury_model import FEATURE_COLUMNS
        X = features_df[FEATURE_COLUMNS].head(10).fillna(0)
        proba = model.predict_proba(X)
        assert proba.shape == (10, 2)

    def test_predict_proba_sums_to_one(self, model, features_df):
        from ml.injury_model import FEATURE_COLUMNS
        X = features_df[FEATURE_COLUMNS].head(20).fillna(0)
        proba = model.predict_proba(X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_predict_returns_binary(self, model, features_df):
        from ml.injury_model import FEATURE_COLUMNS
        X = features_df[FEATURE_COLUMNS].head(20).fillna(0)
        pred = model.predict(X)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_high_fatigue_increases_risk(self, model, features_df):
        """Sanity check: pushing fatigue/ACWR up should not decrease predicted risk on average."""
        from ml.injury_model import FEATURE_COLUMNS
        baseline = features_df[FEATURE_COLUMNS].median().to_frame().T
        stressed = baseline.copy()
        stressed["fatigue"] = 9.5
        stressed["acwr"] = 1.9
        stressed["sommeil_h"] = 4.0
        p_base = model.predict_proba(baseline)[0, 1]
        p_stress = model.predict_proba(stressed)[0, 1]
        assert p_stress >= p_base, (
            f"Risque sous stress ({p_stress:.3f}) < risque baseline ({p_base:.3f})")


# ── Sport model tests ──────────────────────────────────────────────────────────
class TestSportModels:

    @pytest.mark.parametrize("sport", [
        "Football", "Basketball", "Tennis", "Swimming",
        "Rugby", "Athletics", "MMA/Boxing", "Cycling",
    ])
    def test_sport_model_loads(self, sport):
        from ml.injury_model import load_sport_model
        m = load_sport_model(sport)
        assert m is not None, f"Modèle introuvable pour {sport}"
        assert hasattr(m, "threshold_")

    def test_unknown_sport_returns_none(self):
        from ml.injury_model import load_sport_model
        assert load_sport_model("Curling") is None


# ── Advisor tests ──────────────────────────────────────────────────────────────
class TestPerformanceAdvisor:

    def test_advise_from_values_returns_report(self, advisor):
        report = advisor.advise_from_values({"fatigue": 5.0, "acwr": 1.0})
        assert hasattr(report, "injury_prob")
        assert 0.0 <= report.injury_prob <= 1.0

    def test_advise_from_values_body_risk_keys(self, advisor):
        report = advisor.advise_from_values({"fatigue": 5.0})
        assert len(report.body_risk) > 0
        assert all(0.0 <= v <= 1.0 for v in report.body_risk.values())

    def test_advise_from_values_causes_sorted(self, advisor):
        report = advisor.advise_from_values({
            "fatigue": 9.0, "acwr": 1.9, "sommeil_h": 4.0,
        })
        contribs = [c.risk_contrib for c in report.causes]
        assert contribs == sorted(contribs, reverse=True)

    def test_advise_existing_player(self, advisor):
        pid = int(advisor.history["player_id"].iloc[0])
        report = advisor.advise(pid)
        assert report.player_id == pid
        assert 0.0 <= report.injury_prob <= 1.0


# ── Trend utils tests ────────────────────────────────────────────────────────
class TestTrendUtils:

    def test_compute_trend_increasing(self):
        from ml.trend_utils import compute_trend
        s = pd.Series([1]*7 + [5]*7)
        delta, arrow = compute_trend(s, window=7)
        assert arrow == "↑"
        assert delta > 0

    def test_compute_trend_stable(self):
        from ml.trend_utils import compute_trend
        s = pd.Series([3.0]*14)
        delta, arrow = compute_trend(s, window=7)
        assert arrow == "→"

    def test_adaptive_threshold_blends(self):
        from ml.trend_utils import adaptive_threshold
        high_player = np.random.uniform(0.5, 0.7, 30)
        thr = adaptive_threshold(high_player, global_threshold=0.25)
        assert thr > 0.25  # naturally high-risk player should get a higher bar

    def test_adaptive_threshold_falls_back_with_few_samples(self):
        from ml.trend_utils import adaptive_threshold
        thr = adaptive_threshold(np.array([0.9, 0.9]), global_threshold=0.25)
        assert thr == 0.25


# ── RTP protocol tests ──────────────────────────────────────────────────────
class TestRtpProtocol:

    def test_non_injured_player_no_plan(self, features_df):
        from ml.rtp_protocol import build_rtp_plan
        # find a player whose last session is NOT an injury
        for pid in features_df["player_id"].unique():
            ph = features_df[features_df["player_id"] == pid].sort_values("date")
            if ph.iloc[-1]["injury_label"] == 0:
                plan = build_rtp_plan(features_df, int(pid))
                assert plan.is_injured is False
                return
        pytest.skip("Aucun joueur non blessé trouvé dans l'échantillon")

    def test_rtp_weeks_structure(self, features_df):
        from ml.rtp_protocol import build_rtp_plan, RTP_WEEKS
        for pid in features_df["player_id"].unique():
            ph = features_df[features_df["player_id"] == pid].sort_values("date")
            if ph.iloc[-1]["injury_label"] == 1:
                plan = build_rtp_plan(features_df, int(pid))
                assert plan.is_injured is True
                assert len(plan.weeks) == len(RTP_WEEKS)
                return
        pytest.skip("Aucun joueur blessé trouvé dans l'échantillon")


# ── Auth tests ───────────────────────────────────────────────────────────────
class TestAuth:

    def test_default_accounts_exist(self):
        from database.auth import authenticate
        assert authenticate("coach", "coach123") is not None
        assert authenticate("medical", "medical123") is not None

    def test_wrong_password_rejected(self):
        from database.auth import authenticate
        assert authenticate("coach", "wrongpassword") is None

    def test_unknown_user_rejected(self):
        from database.auth import authenticate
        assert authenticate("nonexistent_user", "whatever") is None

    def test_role_permissions(self):
        from database.auth import (
            authenticate, can_see_raw_features, can_manage_users,
        )
        coach = authenticate("coach", "coach123")
        medical = authenticate("medical", "medical123")
        assert can_see_raw_features(coach) is False
        assert can_see_raw_features(medical) is True
        assert can_manage_users(coach) is False


# ── CSV import tests ─────────────────────────────────────────────────────────
class TestCsvImport:

    def test_alias_normalisation(self, tmp_path):
        from ml.csv_import import CsvImporter
        csv_content = "Total Distance,Sprint Count,Sleep Hours\n10.5,12,8.0\n"
        f = tmp_path / "test.csv"
        f.write_text(csv_content)
        df, warns = CsvImporter().load(str(f))
        assert "distance_km" in df.columns
        assert df["distance_km"].iloc[0] == 10.5
        assert df["sprints_count"].iloc[0] == 12

    def test_missing_columns_flagged(self, tmp_path):
        from ml.csv_import import CsvImporter
        f = tmp_path / "sparse.csv"
        f.write_text("distance_km\n5.0\n")
        df, warns = CsvImporter().load(str(f))
        assert len(warns) > 0
        assert any("colonne" in w.lower() for w in warns)


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
