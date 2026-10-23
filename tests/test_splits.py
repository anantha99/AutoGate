"""The OOD holdout in ``autogate_bench.splits`` is balanced by tag.

Run R2 showed what happens otherwise: half of ``cloud_preferred`` was held
out, so the capability head saw two intents of that class in training and
could not learn it. These checks keep every tag value learnable and make the
holdout test every value at least once.
"""

from collections import Counter

import pytest

from autogate_bench import ActuationClass, Capability, Consequence, DistractionLevel, IntentGroup
from autogate_bench.intents import INTENTS, INTENTS_BY_NAME
from autogate_bench.splits import OOD_INTENTS

TAGS = {
    "capability": Capability,
    "distraction": DistractionLevel,
    "actuation": ActuationClass,
    "consequence": Consequence,
}
MIN_TRAIN_INTENTS = 6
MIN_TRAIN_SHARE = 0.7

OOD = [INTENTS_BY_NAME[n] for n in OOD_INTENTS]
TRAIN = [i for i in INTENTS if i.name not in OOD_INTENTS]


def test_ood_intents_are_known_and_distinct():
    assert len(set(OOD_INTENTS)) == len(OOD_INTENTS) == 8
    assert set(OOD_INTENTS) <= set(INTENTS_BY_NAME)


@pytest.mark.parametrize("tag", TAGS)
def test_every_tag_value_keeps_enough_training_intents(tag):
    total = Counter(getattr(i, tag) for i in INTENTS)
    train = Counter(getattr(i, tag) for i in TRAIN)
    short = {
        str(v): (train[v], total[v])
        for v in TAGS[tag]
        if train[v] < MIN_TRAIN_INTENTS or train[v] < MIN_TRAIN_SHARE * total[v]
    }
    assert short == {}


@pytest.mark.parametrize("tag", TAGS)
def test_ood_covers_every_tag_value(tag):
    assert {getattr(i, tag) for i in OOD} == set(TAGS[tag])


def test_ood_spans_at_least_five_groups():
    assert len({i.group for i in OOD}) >= 5
    assert {i.group for i in OOD} <= set(IntentGroup)
