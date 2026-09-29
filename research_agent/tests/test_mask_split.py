import numpy as np

from research_agent.core.masks import split_observed_mask


def test_observed_split_is_disjoint_complete_and_reproducible():
    observed = np.ones((10, 12), dtype=np.bool_)
    observed[2:5, 4:8] = False

    first_train, first_validation = split_observed_mask(observed, 0.2, seed=13)
    second_train, second_validation = split_observed_mask(observed, 0.2, seed=13)

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_validation, second_validation)
    assert not np.logical_and(first_train, first_validation).any()
    assert np.array_equal(np.logical_or(first_train, first_validation), observed)
    assert not first_train[~observed].any()
    assert not first_validation[~observed].any()
    assert first_train.any()
    assert first_validation.any()


def test_mask_matched_split_forms_a_compact_pseudo_hole_for_block_masks():
    observed = np.ones((40, 50), dtype=np.bool_)
    observed[12:28, 17:33] = False

    _, validation = split_observed_mask(
        observed,
        0.1,
        seed=21,
        strategy="mask_matched",
    )

    coordinates = np.argwhere(validation)
    box_area = (np.ptp(coordinates[:, 0]) + 1) * (
        np.ptp(coordinates[:, 1]) + 1
    )
    assert box_area <= 2 * int(validation.sum())
    assert not validation[~observed].any()
