from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from datasketch.hnsw import HNSW


@dataclass
class ClassificationResult:
    """Result of a single nearest-neighbor classification prediction.

    Attributes:
        label: The predicted label. Defensively set to the first label in the
            classifier's label order if no positive vote was cast, which
            cannot occur through normal use because sample weights are
            validated to be positive.
        probabilities: A list of vote shares in the fixed label order given
            when the classifier was initialized. The entries sum to 1.0.
            Labels whose samples were all deleted keep their original column
            and simply receive a zero share.
        neighbor_ids: The IDs of the samples that actually cast a vote, in
            the order returned by the underlying :class:`~datasketch.hnsw.HNSW`
            query.
        neighbor_distances: The distances of the voting samples, aligned with
            ``neighbor_ids``.

    """

    label: Hashable
    probabilities: list[float]
    neighbor_ids: list[Hashable]
    neighbor_distances: list[float]


class HNSWNearestNeighborClassifier:
    """Nearest-neighbor classifier that maintains per-sample features, labels,
    and positive weights on top of an :class:`~datasketch.hnsw.HNSW` index.

    This classifier is designed for data that is continuously re-labeled:
    samples are keyed by ID and can be added, replaced, and deleted in
    batches. Predictions reuse the approximate nearest neighbors returned by
    the underlying HNSW index, so the approximate retrieval semantics of
    HNSW are preserved.

    Each neighbor casts a vote for its label equal to its weight divided by
    its distance to the query point (``weight / distance``). If any neighbor
    has a zero distance, only the zero-distance neighbors participate and
    their raw weights are used as votes. The predicted label is the label
    with the largest accumulated vote. Ties are broken by the fixed label
    order given at initialization. Probabilities are the vote shares in that
    same fixed order, so a label never changes column even after all of its
    samples have been deleted.

    Args:
        labels (Sequence[Hashable]): The fixed, ordered set of labels. The
            order defines both the probability columns and the tie-breaking
            order. Labels must be hashable and unique, and the sequence must
            not be empty.
        distance_func: A function that takes two vectors and returns a float
            representing the distance between them. Forwarded to the
            underlying HNSW index.
        m (int): The number of neighbors to keep for each node in the HNSW
            index.
        ef_construction (int): The number of neighbors to consider during
            HNSW construction.
        m0 (Optional[int]): The number of neighbors to keep for each node at
            the 0th HNSW level. If None, defaults to 2 * m.
        seed (Optional[int]): The random seed for the HNSW index's random
            number generator.
        reversed_edges (bool): Whether to maintain reverse edges in the HNSW
            graph. Speeds up hard removal at the cost of memory and slower
            insertion.

    Examples:
        .. code-block:: python

            import numpy as np
            from datasketch.hnsw_classifier import HNSWNearestNeighborClassifier

            data = np.random.random_sample((100, 10))
            labels = np.random.randint(0, 3, size=100)
            clf = HNSWNearestNeighborClassifier(
                labels=["a", "b", "c"],
                distance_func=lambda x, y: np.linalg.norm(x - y),
            )
            clf.add(ids=range(100), X=data, y=labels)

            # Predict the label of a new point using its 5 approximate
            # nearest neighbors.
            result = clf.predict([data[0]], k=5)[0]
            print(result.label, result.probabilities)

    """

    def __init__(
        self,
        labels: Sequence[Hashable],
        distance_func: Callable[[np.ndarray, np.ndarray], float],
        m: int = 16,
        ef_construction: int = 200,
        m0: Optional[int] = None,
        seed: Optional[int] = None,
        reversed_edges: bool = False,
    ) -> None:
        labels = list(labels)
        if len(labels) == 0:
            raise ValueError("labels must not be empty.")
        if any(not isinstance(label, Hashable) for label in labels):
            raise TypeError("All labels must be hashable.")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must not contain duplicates.")
        self._labels: list[Hashable] = labels
        self._label_to_index: dict[Hashable, int] = {label: i for i, label in enumerate(labels)}
        self._index = HNSW(
            distance_func,
            m=m,
            ef_construction=ef_construction,
            m0=m0,
            seed=seed,
            reversed_edges=reversed_edges,
        )
        # Per-sample metadata keyed by sample ID. Every key here corresponds
        # to a non-soft-removed point in the HNSW index and vice versa.
        self._sample_labels: dict[Hashable, Hashable] = {}
        self._sample_weights: dict[Hashable, float] = {}

    @property
    def labels(self) -> list[Hashable]:
        """Return a copy of the fixed, ordered labels of the classifier."""
        return list(self._labels)

    @property
    def index(self) -> HNSW:
        """Return the underlying HNSW index.

        The index is exposed so that the full HNSW interface (e.g.
        :meth:`~datasketch.hnsw.HNSW.query`,
        :meth:`~datasketch.hnsw.HNSW.clean`,
        :meth:`~datasketch.hnsw.HNSW.copy`, pickling) remains available.
        Mutating the index directly (inserting or removing keys) desynchronizes
        it from the classifier's per-sample metadata; use
        :meth:`add`, :meth:`replace`, and :meth:`remove` instead for updates.
        """
        return self._index

    def __len__(self) -> int:
        """Return the number of samples currently in the classifier."""
        return len(self._sample_labels)

    def __contains__(self, sample_id: Hashable) -> bool:
        """Return ``True`` if the sample ID is currently in the classifier."""
        return sample_id in self._sample_labels

    def get(self, sample_id: Hashable) -> tuple[np.ndarray, Hashable, float]:
        """Return the ``(point, label, weight)`` of the sample with the given ID.

        Raises:
            KeyError: If the sample ID is not in the classifier.

        """
        if sample_id not in self._sample_labels:
            raise KeyError(sample_id)
        return (
            self._index[sample_id],
            self._sample_labels[sample_id],
            self._sample_weights[sample_id],
        )

    def add(
        self,
        ids: Sequence[Hashable],
        X: Sequence[np.ndarray],
        y: Sequence[Hashable],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """Add a batch of new samples.

        The batch is applied atomically: all inputs are validated before any
        change is made, and if an insertion fails, every insertion made by
        this call is rolled back so the classifier is left in its previous
        state.

        Args:
            ids (Sequence[Hashable]): The sample IDs. Must be unique within
                the batch and must not already exist in the classifier.
            X (Sequence[np.ndarray]): The feature vectors, aligned with
                ``ids``.
            y (Sequence[Hashable]): The labels, aligned with ``ids``. Each
                label must be one of the classifier's labels.
            weights (Optional[Sequence[float]]): Positive per-sample weights,
                aligned with ``ids``. Defaults to weight 1.0 for every sample.

        Raises:
            ValueError: If the batch is empty, the sequences have mismatched
                lengths, a label is unknown, a weight is not positive, or an
                ID already exists in the classifier.
            TypeError: If an ID or label is not hashable, or a weight is not
                numeric.

        """
        records = self._validate_batch(ids, X, y, weights, expect_existing=False)
        if any(point is None for _, point, _, _ in records):
            raise ValueError("X is required when adding samples and must be aligned with ids.")
        inserted: list[Hashable] = []
        try:
            for sample_id, point, label, weight in records:
                self._index.insert(sample_id, point)
                inserted.append(sample_id)
                self._sample_labels[sample_id] = label
                # A None weight means the caller did not provide one.
                self._sample_weights[sample_id] = 1.0 if weight is None else weight
        except BaseException:
            # Roll back the partial batch so no half-applied change remains.
            for sample_id in reversed(inserted):
                self._rollback_insert(sample_id)
            raise

    def replace(
        self,
        ids: Sequence[Hashable],
        y: Sequence[Hashable],
        X: Optional[Sequence[np.ndarray]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """Replace the labels (and optionally features and weights) of a batch
        of existing samples.

        The batch is applied atomically: all inputs are validated before any
        change is made, and if a replacement fails, every change made by this
        call is rolled back so the classifier is left in its previous state.

        Args:
            ids (Sequence[Hashable]): The sample IDs. Must be unique within
                the batch and must already exist in the classifier.
            y (Sequence[Hashable]): The new labels, aligned with ``ids``. Each
                label must be one of the classifier's labels.
            X (Optional[Sequence[np.ndarray]): The new feature vectors,
                aligned with ``ids``. If None, the existing features are kept.
            weights (Optional[Sequence[float]]): The new positive per-sample
                weights, aligned with ``ids``. If None, the existing weights
                are kept.

        Raises:
            ValueError: If the batch is empty, the sequences have mismatched
                lengths, a label is unknown, or a weight is not positive.
            KeyError: If an ID does not exist in the classifier.
            TypeError: If an ID or label is not hashable, or a weight is not
                numeric.

        """
        if X is None:
            points: Optional[list[np.ndarray]] = None
        else:
            points = list(X)
        records = self._validate_batch(ids, points, y, weights, expect_existing=True)
        # Save the previous metadata so the batch can be rolled back.
        old_points: dict[Hashable, np.ndarray] = {}
        old_labels: dict[Hashable, Hashable] = {}
        old_weights: dict[Hashable, float] = {}
        updated: list[Hashable] = []
        try:
            for sample_id, point, label, weight in records:
                old_points[sample_id] = self._index[sample_id]
                old_labels[sample_id] = self._sample_labels[sample_id]
                old_weights[sample_id] = self._sample_weights[sample_id]
                if point is not None:
                    self._index.insert(sample_id, point)
                self._sample_labels[sample_id] = label
                if weight is not None:
                    self._sample_weights[sample_id] = weight
                updated.append(sample_id)
        except BaseException:
            # Roll back in reverse order.
            for sample_id in reversed(updated):
                if sample_id in old_points:
                    # Re-inserting the old point repairs the HNSW graph.
                    self._index.insert(sample_id, old_points[sample_id])
                self._sample_labels[sample_id] = old_labels[sample_id]
                self._sample_weights[sample_id] = old_weights[sample_id]
            raise

    def remove(self, ids: Sequence[Hashable], hard: bool = False) -> None:
        """Remove a batch of samples.

        The batch is applied atomically: all inputs are validated before any
        change is made, and if a removal fails, every removal made by this
        call is rolled back so the classifier is left in its previous state.

        Args:
            ids (Sequence[Hashable]): The sample IDs to remove. Must be unique
                within the batch and must currently exist in the classifier.
            hard (bool): If True, hard-remove the points from the underlying
                HNSW index. Otherwise, soft-remove them (the HNSW default).

        Raises:
            KeyError: If an ID does not exist in the classifier.
            TypeError: If an ID is not hashable.
            ValueError: If the batch is empty or contains duplicate IDs.

        """
        sample_ids = self._validate_ids(ids)
        # Snapshot metadata before any mutation so a failed batch can be
        # fully restored.
        removed: list[tuple[Hashable, np.ndarray, Hashable, float]] = []
        for sample_id in sample_ids:
            if sample_id not in self._sample_labels:
                raise KeyError(sample_id)
            removed.append(
                (
                    sample_id,
                    self._index[sample_id],
                    self._sample_labels[sample_id],
                    self._sample_weights[sample_id],
                )
            )
        # Drop the metadata together before touching the index: on failure
        # the whole snapshot is re-inserted below.
        for sample_id, _, _, _ in removed:
            del self._sample_labels[sample_id]
            del self._sample_weights[sample_id]
        try:
            for sample_id, _, _, _ in removed:
                self._index.remove(sample_id, hard=hard)
        except BaseException:
            # Restore the removed points first (re-insertion repairs the
            # graph after hard removal), then their metadata.
            for sample_id, point, label, weight in reversed(removed):
                self._index.insert(sample_id, point)
                self._sample_labels[sample_id] = label
                self._sample_weights[sample_id] = weight
            raise

    def predict(
        self,
        X: Sequence[np.ndarray],
        k: Optional[int] = None,
        ef: Optional[int] = None,
    ) -> list[ClassificationResult]:
        """Predict the labels of a batch of query points.

        Each prediction reuses the nearest neighbors returned by the
        underlying HNSW index's :meth:`~datasketch.hnsw.HNSW.query` method:
        neighbors vote with weight divided by distance, except that when one
        or more neighbors have zero distance, only those zero-distance
        neighbors vote, using their raw weights.

        Args:
            X (Sequence[np.ndarray]): The query points.
            k (Optional[int]): The number of nearest neighbors to retrieve.
                If None, all neighbors found by the HNSW ef-search are used.
            ef (Optional[int]): The search ef forwarded to the HNSW query.
                If None, the index's construction ef is used.

        Returns:
            list[ClassificationResult]: One classification result per query
            point, in input order. The ``probabilities`` field of every result
            follows the classifier's fixed label order.

        Raises:
            ValueError: If the classifier contains no samples (the underlying
                index has no entry point).

        """
        if len(self) == 0:
            raise ValueError("Cannot predict with an empty classifier: no samples have been added.")
        results = []
        for query_point in X:
            neighbors = self._index.query(query_point, k=k, ef=ef)
            votes = [0.0 for _ in self._labels]
            zero_distance_ids = {sample_id for sample_id, distance in neighbors if distance == 0.0}
            if zero_distance_ids:
                # Only zero-distance neighbors vote, with their raw weights.
                voting_neighbors = [
                    (sample_id, distance) for sample_id, distance in neighbors if sample_id in zero_distance_ids
                ]
                for sample_id, _ in voting_neighbors:
                    label_index = self._label_to_index[self._sample_labels[sample_id]]
                    votes[label_index] += self._sample_weights[sample_id]
            else:
                # Every neighbor votes with weight / distance.
                voting_neighbors = neighbors
                for sample_id, distance in voting_neighbors:
                    label_index = self._label_to_index[self._sample_labels[sample_id]]
                    votes[label_index] += self._sample_weights[sample_id] / distance
            neighbor_ids = [sample_id for sample_id, _ in voting_neighbors]
            neighbor_distances = [distance for _, distance in voting_neighbors]
            total_vote = math.fsum(votes)
            if total_vote > 0.0:
                probabilities = [vote / total_vote for vote in votes]
                # argmax returns the first maximum, matching label order, so
                # ties are broken deterministically by the fixed label order.
                label = self._labels[int(np.argmax(np.asarray(votes)))]
            else:
                probabilities = [0.0 for _ in self._labels]
                label = self._labels[0]
            results.append(
                ClassificationResult(
                    label=label,
                    probabilities=probabilities,
                    neighbor_ids=neighbor_ids,
                    neighbor_distances=neighbor_distances,
                )
            )
        return results

    def copy(self) -> HNSWNearestNeighborClassifier:
        """Create a deep copy of the classifier.

        The copy has its own independent HNSW index and per-sample metadata,
        so subsequent updates to either classifier do not affect the other.
        """
        new_clf = HNSWNearestNeighborClassifier.__new__(HNSWNearestNeighborClassifier)
        new_clf._labels = list(self._labels)
        new_clf._label_to_index = dict(self._label_to_index)
        new_clf._index = self._index.copy()
        new_clf._sample_labels = dict(self._sample_labels)
        new_clf._sample_weights = dict(self._sample_weights)
        return new_clf

    def __getstate__(self) -> dict:
        return {
            "labels": self._labels,
            "index": self._index,
            "sample_labels": self._sample_labels,
            "sample_weights": self._sample_weights,
        }

    def __setstate__(self, state: dict) -> None:
        self._labels = list(state["labels"])
        self._label_to_index = {label: i for i, label in enumerate(self._labels)}
        self._index = state["index"]
        self._sample_labels = dict(state["sample_labels"])
        self._sample_weights = dict(state["sample_weights"])

    def _validate_ids(self, ids: Sequence[Hashable]) -> list[Hashable]:
        sample_ids = list(ids)
        if len(sample_ids) == 0:
            raise ValueError("ids must not be empty.")
        for sample_id in sample_ids:
            if not isinstance(sample_id, Hashable):
                raise TypeError(f"Sample IDs must be hashable, got {sample_id!r}.")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("ids must not contain duplicates.")
        return sample_ids

    def _validate_batch(
        self,
        ids: Sequence[Hashable],
        X: Union[Sequence[np.ndarray], list[np.ndarray], None],
        y: Sequence[Hashable],
        weights: Optional[Sequence[float]],
        expect_existing: bool,
    ) -> list[tuple[Hashable, Optional[np.ndarray], Hashable, Optional[float]]]:
        sample_ids = self._validate_ids(ids)
        labels = list(y)
        if len(labels) != len(sample_ids):
            raise ValueError(f"y has length {len(labels)} but ids has length {len(sample_ids)}; they must be aligned.")
        if X is not None and len(X) != len(sample_ids):
            raise ValueError(f"X has length {len(X)} but ids has length {len(sample_ids)}; they must be aligned.")
        if weights is not None and len(weights) != len(sample_ids):
            raise ValueError(
                f"weights has length {len(weights)} but ids has length {len(sample_ids)}; they must be aligned."
            )
        normalized_weights: list[Optional[float]]
        if weights is None:
            normalized_weights = [None for _ in sample_ids]
        else:
            normalized_weights = []
            for weight in weights:
                if isinstance(weight, bool) or not isinstance(weight, (int, float, np.floating, np.integer)):
                    raise TypeError(f"weights must be numeric, got {weight!r}.")
                weight = float(weight)
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(f"weights must be positive and finite, got {weight}.")
                normalized_weights.append(weight)
        records = []
        for i, sample_id in enumerate(sample_ids):
            label = labels[i]
            if not isinstance(label, Hashable):
                raise TypeError(f"Labels must be hashable, got {label!r}.")
            if label not in self._label_to_index:
                raise ValueError(f"Unknown label {label!r}; expected one of {self._labels}.")
            if expect_existing:
                if sample_id not in self._sample_labels:
                    raise KeyError(sample_id)
            elif sample_id in self._sample_labels:
                raise ValueError(f"Sample ID {sample_id!r} already exists; use replace to update it.")
            point = None if X is None else X[i]
            records.append((sample_id, point, label, normalized_weights[i]))
        return records

    def _rollback_insert(self, sample_id: Hashable) -> None:
        # Metadata is removed first; a hard remove keeps the HNSW index in
        # sync with the metadata dictionary.
        self._sample_labels.pop(sample_id, None)
        self._sample_weights.pop(sample_id, None)
        if sample_id in self._index:
            self._index.remove(sample_id, hard=True)
