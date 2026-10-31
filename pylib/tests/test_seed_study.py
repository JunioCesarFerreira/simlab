"""The seed study: its seed lists, and the statistics behind "how many seeds"."""
import pytest

from pylib.seed_study import (
    DEFAULT_GUARD,
    LOW16,
    MAX_SEEDS,
    SEED_LIMIT,
    diagnose_seeds,
    generate_seeds,
    study_rng,
    validate_seeds,
)


class TestGenerateSeeds:
    def test_the_same_master_seed_gives_the_same_list(self):
        assert generate_seeds(30, 7) == generate_seeds(30, 7)

    def test_a_shorter_list_is_a_prefix_of_a_longer_one(self):
        """What lets a study be extended, and "the first m seeds" be used."""
        assert generate_seeds(40, 7)[:25] == generate_seeds(25, 7)

    def test_a_different_master_seed_gives_a_different_list(self):
        assert generate_seeds(10, 1) != generate_seeds(10, 2)

    def test_values_stay_in_the_range_cooja_can_add_a_mote_id_to(self):
        seeds = generate_seeds(MAX_SEEDS, 3)

        assert all(1 <= s < SEED_LIMIT for s in seeds)
        assert SEED_LIMIT + LOW16 == 2**31

    def test_the_largest_list_keeps_its_distance_in_the_low_16_bits(self):
        report = diagnose_seeds(generate_seeds(MAX_SEEDS, 3))

        assert report["low16_collisions"] == 0
        assert report["consecutive_pairs"] == 0

    def test_a_list_that_cannot_fit_says_so(self):
        with pytest.raises(ValueError, match="could not place"):
            generate_seeds(100, 1, guard=LOW16 // 8)

    def test_a_count_out_of_range_is_refused(self):
        with pytest.raises(ValueError, match="count"):
            generate_seeds(MAX_SEEDS + 1, 1)


class TestValidateSeeds:
    def test_an_imported_list_keeps_its_order(self):
        assert validate_seeds([9, 3.0, 7]) == [9, 3, 7]

    @pytest.mark.parametrize("values, mentions", [
        ([], "non-empty"),
        ("123", "non-empty"),
        ([1, 2, 2.5], "whole numbers"),
        ([1, 2, True], "whole numbers"),
        ([1, 2, 0], "must be in"),
        ([1, 2, SEED_LIMIT], "must be in"),
        ([1, 2], "between"),
        ([1, 2, 2], "duplicate"),
    ])
    def test_what_is_refused(self, values, mentions):
        with pytest.raises(ValueError, match=mentions):
            validate_seeds(values)


class TestDiagnoseSeeds:
    def test_seeds_with_the_same_low_bits_collide(self):
        report = diagnose_seeds([5, 5 + LOW16, 20000])

        assert report["low16_collisions"] == 1

    def test_consecutive_seeds_are_counted(self):
        """The wizard's 1, 2, 3 preset: three pairs inside the guard."""
        report = diagnose_seeds([1, 2, 3])

        assert report == {"low16_collisions": 0, "consecutive_pairs": 3,
                          "guard": DEFAULT_GUARD}

    def test_the_distance_wraps_around_the_low_16_bits(self):
        assert diagnose_seeds([1, LOW16 - 1, 30000])["consecutive_pairs"] == 1


class TestStudyRng:
    def test_it_is_reproducible(self):
        assert (study_rng([1, 2], "latency").random()
                == study_rng([1, 2], "latency").random())

    def test_labels_keep_streams_apart(self):
        assert (study_rng([1, 2], "latency").random()
                != study_rng([1, 2], "energy").random())
