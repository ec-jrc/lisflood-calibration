"""
Tests for the Objective class hierarchy: factory dispatch, filter_objectives_for_deap,
ObjectiveMulti.weights computation.
"""

import pytest
import pandas as pd

from liscal.objective import (
    ObjectiveKGE,
    ObjectiveKGEJSD,
    ObjectiveMulti,
    create_objective,
)


# ---------------------------------------------------------------------------
# Mock setup
# ---------------------------------------------------------------------------

class MockDEAPParam:
    def __init__(self, objectives_list):
        self.objectives_list = objectives_list


class MockCfg:
    def __init__(self, objectives_list):
        self.deap_param = MockDEAPParam(objectives_list)
        self.param_ranges = pd.DataFrame()


class MockSubcatch:
    def __init__(self):
        self.path_station = '/tmp/nonexistent'


# ---------------------------------------------------------------------------
# 6.1 Factory function dispatch tests
# ---------------------------------------------------------------------------

class TestFactoryDispatch:

    def test_kge_returns_objective_kge(self):
        cfg = MockCfg(['KGE'])
        subcatch = MockSubcatch()
        obj = create_objective(cfg, subcatch, read_observations=False)
        assert isinstance(obj, ObjectiveKGE)

    def test_kge_jsd_returns_objective_kge_jsd(self):
        cfg = MockCfg(['KGE_JSD'])
        subcatch = MockSubcatch()
        obj = create_objective(cfg, subcatch, read_observations=False)
        assert isinstance(obj, ObjectiveKGEJSD)

    def test_multi_returns_objective_multi(self):
        cfg = MockCfg(['CORR', 'BIAS', 'Y'])
        subcatch = MockSubcatch()
        obj = create_objective(cfg, subcatch, read_observations=False)
        assert isinstance(obj, ObjectiveMulti)

    def test_empty_objectives_raises_value_error(self):
        cfg = MockCfg([])
        subcatch = MockSubcatch()
        with pytest.raises(ValueError):
            create_objective(cfg, subcatch, read_observations=False)

    def test_single_non_standard_returns_objective_multi(self):
        cfg = MockCfg(['CORR'])
        subcatch = MockSubcatch()
        obj = create_objective(cfg, subcatch, read_observations=False)
        assert isinstance(obj, ObjectiveMulti)


# ---------------------------------------------------------------------------
# 6.2 filter_objectives_for_deap correctness tests
# ---------------------------------------------------------------------------

class TestFilterObjectivesForDeap:

    def test_objective_kge_returns_kge_value(self):
        cfg = MockCfg(['KGE'])
        subcatch = MockSubcatch()
        obj = ObjectiveKGE(cfg, subcatch, read_observations=False)

        objectives = (0.8, 0.9, 1.1, 0.95, 0.05)
        result = obj.filter_objectives_for_deap(objectives, {})
        assert result == [0.8]

    def test_objective_kge_jsd_returns_kge_jsd_value(self):
        cfg = MockCfg(['KGE_JSD'])
        subcatch = MockSubcatch()
        obj = ObjectiveKGEJSD(cfg, subcatch, read_observations=False)

        objectives = (0.8, 0.9, 1.1, 0.95, 0.05)
        additional_metrics = {"KGE_JSD": 0.75}
        result = obj.filter_objectives_for_deap(objectives, additional_metrics)
        assert result == [0.75]

    def test_objective_multi_corr_bias_y(self):
        """ObjectiveMulti with ['CORR','BIAS','Y'] applies (x-1)^2 transform."""
        cfg = MockCfg(['CORR', 'BIAS', 'Y'])
        subcatch = MockSubcatch()
        obj = ObjectiveMulti(cfg, subcatch, read_observations=False)

        objectives = (0.8, 0.9, 1.1, 0.95, 0.05)
        result = obj.filter_objectives_for_deap(objectives, {})
        # CORR: (0.9-1)^2 = 0.01, BIAS: (1.1-1)^2 = 0.01, Y: (0.95-1)^2 = 0.0025
        expected = [0.01, 0.01, 0.0025]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10

    def test_objective_multi_kge_jsd(self):
        """ObjectiveMulti with ['KGE','JSD'] returns [KGE, JSD]."""
        cfg = MockCfg(['KGE', 'JSD'])
        subcatch = MockSubcatch()
        obj = ObjectiveMulti(cfg, subcatch, read_observations=False)

        objectives = (0.8, 0.9, 1.1, 0.95, 0.05)
        additional_metrics = {"JSD": 0.3}
        result = obj.filter_objectives_for_deap(objectives, additional_metrics)
        assert result == [0.8, 0.3]

    def test_objective_multi_kge_kge_jsd(self):
        """ObjectiveMulti with ['KGE','KGE_JSD'] returns [KGE, KGE_JSD]."""
        cfg = MockCfg(['KGE', 'KGE_JSD'])
        subcatch = MockSubcatch()
        obj = ObjectiveMulti(cfg, subcatch, read_observations=False)

        objectives = (0.8, 0.9, 1.1, 0.95, 0.05)
        additional_metrics = {"KGE_JSD": 0.75}
        result = obj.filter_objectives_for_deap(objectives, additional_metrics)
        assert result == [0.8, 0.75]


# ---------------------------------------------------------------------------
# 6.3 ObjectiveMulti.weights tests
# ---------------------------------------------------------------------------

class TestObjectiveMultiWeights:

    def test_weights_corr_bias_y(self):
        cfg = MockCfg(['CORR', 'BIAS', 'Y'])
        subcatch = MockSubcatch()
        obj = ObjectiveMulti(cfg, subcatch, read_observations=False)
        expected = {"KGE": 0, "CORR": -1, "BIAS": -1, "Y": -1, "SAE": 0, "JSD": 0, "KGE_JSD": 0}
        assert obj.weights == expected

    def test_weights_kge_jsd(self):
        cfg = MockCfg(['KGE', 'JSD'])
        subcatch = MockSubcatch()
        obj = ObjectiveMulti(cfg, subcatch, read_observations=False)
        expected = {"KGE": 1, "CORR": 0, "BIAS": 0, "Y": 0, "SAE": 0, "JSD": -1, "KGE_JSD": 0}
        assert obj.weights == expected

    def test_weights_kge_jsd_sae(self):
        cfg = MockCfg(['KGE_JSD', 'SAE'])
        subcatch = MockSubcatch()
        obj = ObjectiveMulti(cfg, subcatch, read_observations=False)
        expected = {"KGE": 0, "CORR": 0, "BIAS": 0, "Y": 0, "SAE": -1, "JSD": 0, "KGE_JSD": 1}
        assert obj.weights == expected

    def test_weights_all_objectives(self):
        cfg = MockCfg(['KGE', 'CORR', 'BIAS', 'Y', 'SAE', 'JSD', 'KGE_JSD'])
        subcatch = MockSubcatch()
        obj = ObjectiveMulti(cfg, subcatch, read_observations=False)
        expected = {"KGE": 1, "CORR": -1, "BIAS": -1, "Y": -1, "SAE": -1, "JSD": -1, "KGE_JSD": 1}
        assert obj.weights == expected
        # All values must be non-zero
        assert all(v != 0 for v in obj.weights.values())



class TestCreateObjectiveValueError:
    """Req 5.3: create_objective raises ValueError on empty objectives_list."""

    def test_empty_list_raises_value_error(self):
        cfg = MockCfg([])
        subcatch = MockSubcatch()
        with pytest.raises(ValueError, match="objectives_list is empty"):
            create_objective(cfg, subcatch, read_observations=False)

    def test_none_objectives_list_raises(self):
        cfg = MockCfg(None)
        subcatch = MockSubcatch()
        # None is falsy, should also trigger ValueError
        with pytest.raises((ValueError, TypeError)):
            create_objective(cfg, subcatch, read_observations=False)
