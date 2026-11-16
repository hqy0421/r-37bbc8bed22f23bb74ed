import pickle
import unittest
from unittest import mock

import numpy as np
import pytest

from datasketch import HNSW
from datasketch.hnsw_classifier import ClassificationResult, HNSWNearestNeighborClassifier


def l2_distance(x, y):
    return np.linalg.norm(x - y)


class TestHNSWNearestNeighborClassifier(unittest.TestCase):
    def _create_classifier(self, labels=("a", "b", "c"), **kwargs):
        return HNSWNearestNeighborClassifier(labels=list(labels), distance_func=l2_distance, **kwargs)

    # ------------------------------------------------------------------
    # Construction and validation
    # ------------------------------------------------------------------
    def test_invalid_labels(self):
        self.assertRaises(ValueError, HNSWNearestNeighborClassifier, labels=[], distance_func=l2_distance)
        self.assertRaises(ValueError, HNSWNearestNeighborClassifier, labels=["a", "a"], distance_func=l2_distance)
        self.assertRaises(TypeError, HNSWNearestNeighborClassifier, labels=["a", ["b"]], distance_func=l2_distance)

    def test_labels_property_is_fixed_ordered_copy(self):
        clf = self._create_classifier(labels=("c", "a", "b"))
        self.assertEqual(clf.labels, ["c", "a", "b"])
        returned = clf.labels
        returned.append("d")
        self.assertEqual(clf.labels, ["c", "a", "b"])

    def test_add_validation_errors_change_nothing(self):
        clf = self._create_classifier()
        clf.add(ids=[0], X=[np.array([1.0])], y=["a"])

        # Duplicate ID.
        self.assertRaises(ValueError, clf.add, ids=[0], X=[np.array([2.0])], y=["b"])
        # Unknown label.
        self.assertRaises(ValueError, clf.add, ids=[1], X=[np.array([2.0])], y=["z"])
        # Mismatched lengths.
        self.assertRaises(ValueError, clf.add, ids=[1, 2], X=[np.array([2.0])], y=["b", "a"])
        # Bad weights.
        for bad_weight in (0.0, -1.0, float("nan"), float("inf"), "heavy", True):
            self.assertRaises(
                (ValueError, TypeError),
                clf.add,
                ids=[1],
                X=[np.array([2.0])],
                y=["b"],
                weights=[bad_weight],
            )
        # Nothing was added beyond the original sample.
        self.assertEqual(len(clf), 1)
        self.assertEqual(set(clf._sample_labels.keys()), {0})

    def test_empty_batch_rejected(self):
        clf = self._create_classifier()
        self.assertRaises(ValueError, clf.add, [], [], [])
        self.assertRaises(ValueError, clf.remove, [])

    # ------------------------------------------------------------------
    # Prediction and weighted inverse-distance voting
    # ------------------------------------------------------------------
    def test_weighted_inverse_distance_voting(self):
        clf = self._create_classifier()
        # Points on a line at distances 1, 2, 3 from the query at the origin.
        clf.add(
            ids=[0, 1, 2],
            X=[np.array([1.0]), np.array([2.0]), np.array([3.0])],
            y=["a", "b", "a"],
            weights=[1.0, 4.0, 1.0],
        )
        results = clf.predict([np.array([0.0])], k=3)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, ClassificationResult)
        # Votes: a = 1/1 + 1/3, b = 4/2 = 2.0 -> b wins.
        self.assertEqual(result.label, "b")
        self.assertEqual(len(result.probabilities), 3)
        assert result.probabilities == pytest.approx([0.4, 0.6, 0.0])
        assert sum(result.probabilities) == pytest.approx(1.0)
        self.assertEqual(set(result.neighbor_ids), {0, 1, 2})
        self.assertEqual(len(result.neighbor_distances), 3)

    def test_default_weight_is_one(self):
        clf = self._create_classifier(labels=("a", "b"))
        clf.add(ids=[0, 1], X=[np.array([1.0]), np.array([-1.0])], y=["a", "b"])
        _, _, weight = clf.get(0)
        self.assertEqual(weight, 1.0)

    def test_tie_broken_by_fixed_label_order(self):
        clf = self._create_classifier(labels=("a", "b"))
        clf.add(ids=[0, 1], X=[np.array([1.0]), np.array([-1.0])], y=["a", "b"])
        result = clf.predict([np.array([0.0])], k=2)[0]
        self.assertEqual(result.label, "a")
        assert result.probabilities == pytest.approx([0.5, 0.5])

        # Reversing the label order reverses the tie-break and the columns.
        clf_rev = HNSWNearestNeighborClassifier(labels=["b", "a"], distance_func=l2_distance)
        clf_rev.add(ids=[0, 1], X=[np.array([1.0]), np.array([-1.0])], y=["a", "b"])
        result_rev = clf_rev.predict([np.array([0.0])], k=2)[0]
        self.assertEqual(result_rev.label, "b")
        assert result_rev.probabilities == pytest.approx([0.5, 0.5])

    def test_zero_distance_only_zero_distance_neighbors_vote(self):
        clf = self._create_classifier()
        clf.add(
            ids=[0, 1, 2, 3],
            X=[
                np.array([1.0]),
                np.array([2.0]),
                np.array([3.0]),
                np.array([0.0]),
            ],
            y=["a", "b", "a", "c"],
            weights=[100.0, 100.0, 100.0, 1.0],
        )
        # Querying an exact stored point: only the zero-distance neighbor may
        # vote, regardless of the other neighbors' large weights.
        result = clf.predict([np.array([0.0])], k=4)[0]
        self.assertEqual(result.neighbor_ids, [3])
        self.assertEqual(result.neighbor_distances, [0.0])
        self.assertEqual(result.label, "c")
        assert result.probabilities == pytest.approx([0.0, 0.0, 1.0])

    def test_multiple_zero_distance_neighbors_vote_by_raw_weight(self):
        clf = self._create_classifier(labels=("a", "b"))
        # Three distinct IDs share one point; every one of them is at zero
        # distance from a query at that point.
        clf.add(
            ids=[0, 1, 2],
            X=[np.array([5.0]), np.array([5.0]), np.array([5.0])],
            y=["a", "b", "b"],
            weights=[1.0, 1.0, 2.0],
        )
        clf.add(ids=[3], X=[np.array([0.0])], y=["a"], weights=[1000.0])
        result = clf.predict([np.array([5.0])], k=4)[0]
        self.assertEqual(set(result.neighbor_ids), {0, 1, 2})
        self.assertTrue(all(distance == 0.0 for distance in result.neighbor_distances))
        self.assertEqual(result.label, "b")
        # a: 1.0, b: 3.0 out of a total weight of 4.0.
        assert result.probabilities == pytest.approx([0.25, 0.75])

    def test_predict_on_empty_classifier_raises(self):
        clf = self._create_classifier()
        self.assertRaises(ValueError, clf.predict, [np.array([0.0])])
        # Still raises after every sample has been removed.
        clf.add(ids=[0], X=[np.array([1.0])], y=["a"])
        clf.remove([0])
        self.assertEqual(len(clf), 0)
        self.assertRaises(ValueError, clf.predict, [np.array([0.0])])

    def test_probability_columns_remain_fixed_after_label_emptied(self):
        clf = self._create_classifier(labels=("a", "b", "c"))
        clf.add(
            ids=[0, 1, 2, 3],
            X=[
                np.array([0.0]),
                np.array([10.0]),
                np.array([11.0]),
                np.array([12.0]),
            ],
            y=["a", "b", "b", "c"],
        )
        # Delete every "c" sample; its column must remain in position 2.
        clf.remove([3])
        result = clf.predict([np.array([0.0])], k=3)[0]
        self.assertEqual(len(result.probabilities), 3)
        self.assertEqual(result.probabilities[2], 0.0)
        self.assertEqual(clf.labels, ["a", "b", "c"])
        # Re-adding under a previously emptied label fills the same column.
        # Query a point that is not itself stored (no zero-distance shortcut)
        # but much closer to the newly added "c" sample.
        clf.add(ids=[4], X=[np.array([0.1])], y=["c"], weights=[1000.0])
        result = clf.predict([np.array([0.05])], k=1)[0]
        self.assertEqual(result.label, "c")
        self.assertEqual(len(result.probabilities), 3)
        self.assertGreater(result.probabilities[2], 0.0)

    # ------------------------------------------------------------------
    # Replace
    # ------------------------------------------------------------------
    def test_replace_label_feature_and_weight(self):
        clf = self._create_classifier()
        clf.add(
            ids=[0, 1],
            X=[np.array([0.0]), np.array([10.0])],
            y=["a", "b"],
            weights=[1.0, 1.0],
        )
        # Replace only the label and weight; feature stays.
        clf.replace(ids=[0], y=["c"], weights=[5.0])
        point, label, weight = clf.get(0)
        self.assertEqual(label, "c")
        self.assertEqual(weight, 5.0)
        self.assertTrue(np.array_equal(point, np.array([0.0])))

        # Replace the feature too: move sample 0 far away, next to sample 1.
        clf.replace(ids=[0], y=["c"], X=[np.array([100.0])])
        point, label, weight = clf.get(0)
        self.assertTrue(np.array_equal(point, np.array([100.0])))
        self.assertEqual(label, "c")
        self.assertEqual(weight, 5.0)

        # The moved point is now the nearest stored point to sample 1.
        result = clf.predict([np.array([100.0])], k=1)[0]
        self.assertEqual(result.neighbor_ids, [0])
        self.assertEqual(result.label, "c")

    def test_replace_missing_id_raises_keyerror(self):
        clf = self._create_classifier()
        clf.add(ids=[0], X=[np.array([0.0])], y=["a"])
        self.assertRaises(KeyError, clf.replace, ids=[1], y=["b"])
        # State untouched.
        self.assertEqual(len(clf), 1)
        self.assertEqual(clf.get(0)[1], "a")

    def test_replace_unknown_or_invalid_label_and_duplicate_ids_rejected(self):
        clf = self._create_classifier()
        clf.add(ids=[0, 1], X=[np.array([0.0]), np.array([1.0])], y=["a", "b"])
        # Unknown labels are rejected.
        self.assertRaises(ValueError, clf.replace, ids=[0, 1], y=["z", "a"])
        # Duplicate IDs within the batch are rejected.
        self.assertRaises(ValueError, clf.replace, ids=[0, 0], y=["a", "b"])

    # ------------------------------------------------------------------
    # Removal: both soft and hard variants
    # ------------------------------------------------------------------
    def test_remove_soft_and_hard(self):
        for hard in (False, True):
            clf = self._create_classifier(reversed_edges=hard)
            clf.add(
                ids=[0, 1, 2],
                X=[np.array([0.0]), np.array([1.0]), np.array([2.0])],
                y=["a", "b", "c"],
            )
            clf.remove([1], hard=hard)
            self.assertNotIn(1, clf)
            self.assertRaises(KeyError, clf.get, 1)
            self.assertEqual(len(clf), 2)
            result = clf.predict([np.array([1.0])], k=2)[0]
            self.assertEqual(set(result.neighbor_ids), {0, 2})
            # Removing an unknown ID fails the whole batch.
            self.assertRaises(KeyError, clf.remove, [0, 9])
            self.assertIn(0, clf)
            self.assertEqual(len(clf), 2)

    def test_remove_duplicate_ids_rejected(self):
        clf = self._create_classifier()
        clf.add(ids=[0, 1], X=[np.array([0.0]), np.array([1.0])], y=["a", "b"])
        self.assertRaises(ValueError, clf.remove, [0, 0])
        self.assertEqual(len(clf), 2)

    # ------------------------------------------------------------------
    # Atomicity: injected mid-batch failures leave no partial change
    # ------------------------------------------------------------------
    def test_add_failure_rolls_back(self):
        clf = self._create_classifier()
        clf.add(ids=[9], X=[np.array([9.0])], y=["a"])
        with mock.patch.object(clf.index, "insert", side_effect=[None, RuntimeError("boom")]):
            self.assertRaises(
                RuntimeError,
                clf.add,
                ids=[0, 1],
                X=[np.array([0.0]), np.array([1.0])],
                y=["a", "b"],
            )
        # Neither the failed sample nor its rolled-back predecessor remains.
        self.assertEqual(len(clf), 1)
        self.assertEqual(set(clf._sample_labels.keys()), {9})
        self.assertNotIn(0, clf.index)
        self.assertNotIn(1, clf.index)

    def test_replace_failure_rolls_back(self):
        clf = self._create_classifier()
        clf.add(
            ids=[0, 1],
            X=[np.array([0.0]), np.array([10.0])],
            y=["a", "b"],
            weights=[1.0, 2.0],
        )
        # Fail on the second forward insert; the two rollback inserts of the
        # old points must still succeed.
        original_insert = clf.index.insert

        def flaky_insert(*args, **kwargs):
            flaky_insert.calls += 1
            if flaky_insert.calls == 2:
                raise RuntimeError("boom")
            return original_insert(*args, **kwargs)

        flaky_insert.calls = 0
        with mock.patch.object(clf.index, "insert", side_effect=flaky_insert):
            self.assertRaises(
                RuntimeError,
                clf.replace,
                ids=[0, 1],
                y=["c", "c"],
                X=[np.array([5.0]), np.array([6.0])],
                weights=[3.0, 4.0],
            )
        # Both samples restored to their previous point, label, and weight.
        point0, label0, weight0 = clf.get(0)
        point1, label1, weight1 = clf.get(1)
        self.assertTrue(np.array_equal(point0, np.array([0.0])))
        self.assertTrue(np.array_equal(point1, np.array([10.0])))
        self.assertEqual((label0, label1), ("a", "b"))
        self.assertEqual((weight0, weight1), (1.0, 2.0))

    def test_remove_failure_rolls_back(self):
        clf = self._create_classifier()
        clf.add(
            ids=[0, 1, 2],
            X=[np.array([0.0]), np.array([1.0]), np.array([2.0])],
            y=["a", "b", "c"],
        )
        with mock.patch.object(clf.index, "remove", side_effect=[None, RuntimeError("boom")]):
            self.assertRaises(RuntimeError, clf.remove, [0, 1, 2])
        self.assertEqual(len(clf), 3)
        for sample_id in (0, 1, 2):
            self.assertIn(sample_id, clf)
            self.assertIn(sample_id, clf.index)

    # ------------------------------------------------------------------
    # Copy and pickle produce independently updatable classifiers
    # ------------------------------------------------------------------
    def test_copy_is_independent(self):
        clf = self._create_classifier()
        clf.add(ids=[0, 1], X=[np.array([0.0]), np.array([1.0])], y=["a", "b"])
        clone = clf.copy()
        self.assertIsNot(clone, clf)
        self.assertIsNot(clone.index, clf.index)

        clf.add(ids=[2], X=[np.array([2.0])], y=["c"])
        clone.remove([0])
        self.assertIn(2, clf)
        self.assertNotIn(2, clone)
        self.assertIn(0, clf)
        self.assertNotIn(0, clone)
        # The clone still predicts on its own index.
        self.assertEqual(clone.predict([np.array([0.0])], k=1)[0].label, "b")

    def test_pickle_roundtrip_is_independent(self):
        clf = self._create_classifier()
        clf.add(
            ids=[0, 1, 2],
            X=[np.array([0.0]), np.array([1.0]), np.array([2.0])],
            y=["a", "b", "c"],
            weights=[1.0, 2.0, 3.0],
        )
        loaded = pickle.loads(pickle.dumps(clf))
        self.assertEqual(loaded.labels, clf.labels)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded.get(2)[1:], ("c", 3.0))
        self.assertTrue(np.array_equal(loaded.get(2)[0], np.array([2.0])))

        # Updating the loaded classifier must not touch the original.
        loaded.replace(ids=[2], y=["a"])
        loaded.add(ids=[3], X=[np.array([3.0])], y=["b"])
        self.assertEqual(clf.get(2)[1], "c")
        self.assertNotIn(3, clf)
        self.assertEqual(len(loaded), 4)
        # And predictions still work after unpickling.
        result = loaded.predict([np.array([0.0])], k=2)[0]
        self.assertIn(result.label, ("a", "b"))

    # ------------------------------------------------------------------
    # The original HNSW interface and approximate semantics are retained
    # ------------------------------------------------------------------
    def test_underlying_hnsw_interface_retained(self):
        clf = self._create_classifier()
        self.assertIsInstance(clf.index, HNSW)
        clf.add(
            ids=[0, 1, 2],
            X=[np.array([0.0]), np.array([1.0]), np.array([2.0])],
            y=["a", "b", "c"],
        )
        # The HNSW query interface is unchanged: (id, distance) pairs.
        neighbors = clf.index.query(np.array([0.0]), k=2)
        self.assertEqual([sample_id for sample_id, _ in neighbors], [0, 1])
        self.assertEqual(neighbors[0][1], 0.0)
        # Mapping-style HNSW access still works.
        self.assertTrue(np.array_equal(clf.index[2], np.array([2.0])))
        self.assertEqual(len(clf.index), 3)

    def test_contains_and_get(self):
        clf = self._create_classifier()
        clf.add(ids=["x"], X=[np.array([7.0])], y=["b"], weights=[2.5])
        self.assertIn("x", clf)
        self.assertNotIn("y", clf)
        point, label, weight = clf.get("x")
        self.assertTrue(np.array_equal(point, np.array([7.0])))
        self.assertEqual(label, "b")
        self.assertEqual(weight, 2.5)
        self.assertRaises(KeyError, clf.get, "y")


if __name__ == "__main__":
    unittest.main()
