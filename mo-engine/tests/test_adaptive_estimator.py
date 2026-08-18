"""Unit tests for the replaceable objective estimator (weighted k-NN)."""
import numpy as np
import pytest

from lib.adaptive import ObjectiveEstimator, ObjectivePrediction, WeightedKNNEstimator


def _fit(k=3, epsilon=1e-9, X=None, Y=None) -> WeightedKNNEstimator:
    estimator = WeightedKNNEstimator(k=k, epsilon=epsilon)
    if X is not None:
        estimator.fit(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))
    return estimator


class TestContract:
    def test_satisfies_the_protocol(self):
        assert isinstance(WeightedKNNEstimator(), ObjectiveEstimator)

    def test_rejects_invalid_hyperparameters(self):
        with pytest.raises(ValueError):
            WeightedKNNEstimator(k=0)
        with pytest.raises(ValueError):
            WeightedKNNEstimator(epsilon=0.0)

    def test_untrained_estimator_predicts_nothing(self):
        estimator = WeightedKNNEstimator()
        assert estimator.n_samples == 0
        assert estimator.predict([1.0, 2.0]) is None

    def test_fitting_empty_arrays_keeps_it_untrained(self):
        estimator = _fit(X=np.zeros((0, 3)), Y=np.zeros((0, 2)))
        assert estimator.n_samples == 0
        assert estimator.predict([0.0, 0.0, 0.0]) is None

    def test_sample_count_mismatch_is_rejected(self):
        estimator = WeightedKNNEstimator()
        with pytest.raises(ValueError):
            estimator.fit(np.zeros((3, 2)), np.zeros((2, 2)))

    def test_descriptor_length_mismatch_is_rejected(self):
        estimator = _fit(X=[[0.0, 0.0], [1.0, 1.0]], Y=[[1.0], [2.0]])
        with pytest.raises(ValueError):
            estimator.predict([0.0, 0.0, 0.0])


class TestPrediction:
    def test_exact_neighbour_is_returned_verbatim(self):
        estimator = _fit(
            k=3,
            X=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            Y=[[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]],
        )
        prediction = estimator.predict([1.0, 0.0])

        assert prediction is not None
        assert prediction.exact is True
        assert prediction.descriptor_distance == 0.0
        assert prediction.confidence == 1.0
        assert list(prediction.mean) == [20.0, 2.0]
        assert list(prediction.uncertainty) == [0.0, 0.0]

    def test_weighted_mean_matches_the_closed_form(self):
        # One feature so the normalisation is a plain [0, 1] rescale.
        X = [[0.0], [1.0], [2.0]]
        Y = [[0.0], [10.0], [20.0]]
        estimator = _fit(k=3, epsilon=1e-9, X=X, Y=Y)
        prediction = estimator.predict([0.5])

        # Normalised descriptors: 0, .5, 1 -> distances .25, 0.0? no: query .25
        normalized_x = np.asarray([0.0, 0.5, 1.0])
        query = 0.25
        distances = np.abs(normalized_x - query)
        weights = 1.0 / (distances + 1e-9)
        expected_mean = float((weights * np.asarray([0.0, 10.0, 20.0])).sum() / weights.sum())
        expected_sigma = float(
            np.sqrt((weights * (np.asarray([0.0, 10.0, 20.0]) - expected_mean) ** 2).sum() / weights.sum())
        )

        assert prediction.mean[0] == pytest.approx(expected_mean)
        assert prediction.uncertainty[0] == pytest.approx(expected_sigma)
        assert prediction.descriptor_distance == pytest.approx(float(distances.min()))

    def test_uncertainty_is_zero_when_neighbours_agree(self):
        estimator = _fit(k=3, X=[[0.0], [1.0], [2.0]], Y=[[5.0], [5.0], [5.0]])
        prediction = estimator.predict([0.4])

        assert prediction.mean[0] == pytest.approx(5.0)
        assert prediction.uncertainty[0] == pytest.approx(0.0)

    def test_uncertainty_grows_when_neighbours_disagree(self):
        agree = _fit(k=2, X=[[0.0], [1.0]], Y=[[5.0], [5.2]]).predict([0.5])
        disagree = _fit(k=2, X=[[0.0], [1.0]], Y=[[5.0], [95.0]]).predict([0.5])

        assert disagree.uncertainty[0] > agree.uncertainty[0]

    def test_normalisation_equalises_feature_scales(self):
        # Feature 1 spans [0, 1000], feature 2 spans [0, 1]: without
        # normalisation the first would dominate the distance entirely.
        X = [[0.0, 0.0], [1000.0, 0.0], [0.0, 1.0], [1000.0, 1.0]]
        Y = [[0.0], [1.0], [2.0], [3.0]]
        estimator = _fit(k=1, X=X, Y=Y)

        assert estimator.predict([10.0, 0.9]).mean[0] == pytest.approx(2.0)
        assert estimator.predict([990.0, 0.1]).mean[0] == pytest.approx(1.0)

    def test_constant_feature_does_not_break_normalisation(self):
        estimator = _fit(k=2, X=[[1.0, 0.0], [1.0, 1.0]], Y=[[0.0], [10.0]])
        prediction = estimator.predict([1.0, 0.2])
        assert prediction is not None
        assert np.isfinite(prediction.mean).all()

    def test_fewer_samples_than_k_uses_what_exists(self):
        estimator = _fit(k=7, X=[[0.0], [1.0]], Y=[[1.0], [3.0]])
        prediction = estimator.predict([0.5])

        assert len(prediction.neighbors) == 2
        assert 1.0 <= prediction.mean[0] <= 3.0

    def test_single_sample_returns_it(self):
        estimator = _fit(k=5, X=[[0.3, 0.7]], Y=[[42.0, -1.0]])
        prediction = estimator.predict([0.9, 0.1])

        assert list(prediction.mean) == [42.0, -1.0]
        assert list(prediction.uncertainty) == [0.0, 0.0]

    @pytest.mark.parametrize("n_obj", [1, 2, 3, 5])
    def test_arbitrary_objective_counts(self, n_obj):
        rng = np.random.default_rng(3)
        X = rng.random((20, 4))
        Y = rng.random((20, n_obj))
        prediction = _fit(k=4, X=X, Y=Y).predict(rng.random(4))

        assert prediction.mean.shape == (n_obj,)
        assert prediction.uncertainty.shape == (n_obj,)

    def test_confidence_decreases_with_distance(self):
        estimator = _fit(k=2, X=[[0.0], [1.0]], Y=[[0.0], [1.0]])
        near = estimator.predict([0.05])
        far = estimator.predict([0.5])

        assert 0.0 < far.confidence <= near.confidence <= 1.0

    def test_prediction_is_serialisable(self):
        prediction = _fit(k=2, X=[[0.0], [1.0]], Y=[[0.0], [1.0]]).predict([0.4])
        payload = prediction.as_dict()

        assert set(payload) >= {"mean", "uncertainty", "descriptor_distance", "confidence", "exact"}
        assert isinstance(payload["mean"], list)
        assert isinstance(ObjectivePrediction(**{
            "mean": np.asarray([0.0]),
            "uncertainty": np.asarray([0.0]),
            "neighbors": (),
            "neighbor_distances": (),
            "descriptor_distance": 0.0,
            "confidence": 1.0,
        }), ObjectivePrediction)
