from __future__ import annotations

from collections.abc import Iterable, Mapping

PUBLIC_SPLITS = frozenset({"development", "diagnostic"})


def validate_structural_splits(
    records: Iterable[Mapping[str, object]],
    *,
    descriptor_fields: tuple[str, ...],
    parent_field: str,
) -> None:
    """Reject parent leakage and cross-split structural near-duplicates."""
    parents: dict[object, str] = {}
    descriptors: list[tuple[str, tuple[object, ...]]] = []
    for record in records:
        split = record.get("split")
        if split not in PUBLIC_SPLITS:
            raise ValueError("records must use development or diagnostic split")
        parent = record.get(parent_field)
        previous_split = parents.setdefault(parent, str(split))
        if previous_split != split:
            raise ValueError(f"{parent_field} spans structural splits")
        try:
            descriptor = tuple(record[field] for field in descriptor_fields)
        except KeyError as error:
            raise ValueError(
                f"structural descriptor is missing {error.args[0]}"
            ) from error
        for other_split, other in descriptors:
            if other_split == split:
                continue
            differences = sum(left != right for left, right in zip(descriptor, other))
            if differences <= 1:
                raise ValueError("cross-split structural near-duplicate")
        descriptors.append((str(split), descriptor))
