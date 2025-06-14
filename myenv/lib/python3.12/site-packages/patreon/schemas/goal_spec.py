import pytest

from patreon.schemas import goal


@pytest.fixture
def attributes():
    return [
        'amount_cents',
        'title',
        'description',
        'created_at',
        'reached_at',
    ]


@pytest.fixture
def relationships():
    return []


def test_schema_attributes_are_properly_formatted(attributes):
    for attribute_name in attributes:
        value = getattr(goal.Attributes, attribute_name, None)
        assert value is not None and value is attribute_name


def test_schema_relationships_are_properly_formatted(relationships):
    assert len(relationships) is 0


def test_schema_has_expected_default_attributes():
    assert goal.Attributes.amount_cents == 'amount_cents'
    assert goal.Attributes.title == 'title'
    assert goal.Attributes.description == 'description'
    assert goal.Attributes.created_at == 'created_at'
    assert goal.Attributes.reached_at == 'reached_at'


def test_schema_has_expected_default_relationships():
    assert len(goal.default_relationships) is 0
