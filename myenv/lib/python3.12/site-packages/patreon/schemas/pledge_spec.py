import pytest

from patreon.schemas import pledge


@pytest.fixture
def attributes():
    return [
        'amount_cents',
        'total_historical_amount_cents',
        'declined_since',
        'created_at',
        'pledge_cap_cents',
        'patron_pays_fees',
        'unread_count',
    ]


@pytest.fixture
def relationships():
    return [
        'patron',
        'reward',
        'creator',
        'address',
        'card',
        'pledge_vat_location',
    ]


def test_schema_attributes_are_properly_formatted(attributes):
    for attribute_name in attributes:
        value = getattr(pledge.Attributes, attribute_name, None)
        assert value is not None and value is attribute_name


def test_schema_relationships_are_properly_formatted(relationships):
    for relationship_name in relationships:
        value = getattr(pledge.Relationships, relationship_name, None)
        assert value is not None and value is relationship_name


def test_schema_has_expected_default_attributes():
    assert pledge.Attributes.amount_cents in pledge.default_attributes
    assert pledge.Attributes.declined_since in pledge.default_attributes
    assert pledge.Attributes.created_at in pledge.default_attributes
    assert pledge.Attributes.pledge_cap_cents in pledge.default_attributes
    assert pledge.Attributes.patron_pays_fees in pledge.default_attributes


def test_schema_has_expected_default_relationships():
    assert pledge.Relationships.address in pledge.default_relationships
    assert pledge.Relationships.creator in pledge.default_relationships
    assert pledge.Relationships.patron in pledge.default_relationships
    assert pledge.Relationships.pledge_vat_location in pledge.default_relationships
    assert pledge.Relationships.reward in pledge.default_relationships
