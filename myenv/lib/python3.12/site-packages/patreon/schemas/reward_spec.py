import pytest

from patreon.schemas import reward


@pytest.fixture
def attributes():
    return [
        'amount',
        'amount_cents',
        'user_limit',
        'remaining',
        'description',
        'requires_shipping',
        'created_at',
        'url',
        'patron_count',
    ]


@pytest.fixture
def relationships():
    return [
        'creator',
    ]


def test_schema_attributes_are_properly_formatted(attributes):
    for attribute_name in attributes:
        value = getattr(reward.Attributes, attribute_name, None)
        assert value is not None and value is attribute_name


def test_schema_relationships_are_properly_formatted(relationships):
    for relationship_name in relationships:
        value = getattr(reward.Relationships, relationship_name, None)
        assert value is not None and value is relationship_name


def test_schema_has_expected_default_attributes():
    assert reward.Attributes.amount in reward.default_attributes
    assert reward.Attributes.amount_cents in reward.default_attributes
    assert reward.Attributes.user_limit in reward.default_attributes
    assert reward.Attributes.remaining in reward.default_attributes
    assert reward.Attributes.description in reward.default_attributes
    assert reward.Attributes.requires_shipping in reward.default_attributes
    assert reward.Attributes.created_at in reward.default_attributes
    assert reward.Attributes.url in reward.default_attributes
    assert reward.Attributes.patron_count in reward.default_attributes


def test_schema_has_expected_default_relationships():
    assert reward.Relationships.creator in reward.default_relationships
