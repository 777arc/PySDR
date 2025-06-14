import pytest

from patreon.schemas import campaign


@pytest.fixture
def attributes():
    return [
        'summary',
        'creation_name',
        'pay_per_name',
        'one_liner',
        'main_video_embed',
        'main_video_url',
        'image_small_url',
        'image_url',
        'thanks_video_url',
        'thanks_embed',
        'thanks_msg',
        'is_monthly',
        'is_nsfw',
        'is_charged_immediately',
        'is_plural',
        'created_at',
        'published_at',
        'pledge_url',
        'pledge_sum',
        'patron_count',
        'creation_count',
        'outstanding_payment_amount_cents',
    ]


@pytest.fixture
def relationships():
    return [
        'rewards',
        'creator',
        'goals',
        'pledges',
        'current_user_pledge',
        'post_aggregation',
        'categories',
        'preview_token',
    ]


def test_schema_attributes_are_properly_formatted(attributes):
    for attribute_name in attributes:
        value = getattr(campaign.Attributes, attribute_name, None)
        assert value is not None and value is attribute_name


def test_schema_relationships_are_properly_formatted(relationships):
    for relationship_name in relationships:
        value = getattr(campaign.Relationships, relationship_name, None)
        assert value is not None and value is relationship_name


def test_schema_has_expected_default_attributes():
    assert campaign.Attributes.summary in campaign.default_attributes
    assert campaign.Attributes.creation_name in campaign.default_attributes
    assert campaign.Attributes.pay_per_name in campaign.default_attributes
    assert campaign.Attributes.one_liner in campaign.default_attributes
    assert campaign.Attributes.main_video_embed in campaign.default_attributes
    assert campaign.Attributes.main_video_url in campaign.default_attributes
    assert campaign.Attributes.image_small_url in campaign.default_attributes
    assert campaign.Attributes.image_url in campaign.default_attributes
    assert campaign.Attributes.thanks_video_url in campaign.default_attributes
    assert campaign.Attributes.thanks_embed in campaign.default_attributes
    assert campaign.Attributes.thanks_msg in campaign.default_attributes
    assert campaign.Attributes.is_monthly in campaign.default_attributes
    assert campaign.Attributes.is_nsfw in campaign.default_attributes
    assert campaign.Attributes.is_charged_immediately in campaign.default_attributes
    assert campaign.Attributes.is_plural in campaign.default_attributes
    assert campaign.Attributes.created_at in campaign.default_attributes
    assert campaign.Attributes.published_at in campaign.default_attributes
    assert campaign.Attributes.pledge_url in campaign.default_attributes
    assert campaign.Attributes.pledge_sum in campaign.default_attributes
    assert campaign.Attributes.patron_count in campaign.default_attributes
    assert campaign.Attributes.creation_count in campaign.default_attributes
    assert campaign.Attributes.outstanding_payment_amount_cents in campaign.default_attributes


def test_schema_has_expected_default_relationships():
    assert campaign.Relationships.rewards in campaign.default_relationships
    assert campaign.Relationships.creator in campaign.default_relationships
    assert campaign.Relationships.goals in campaign.default_relationships
