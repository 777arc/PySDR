import pytest

from patreon.schemas import user


@pytest.fixture
def attributes():
    return [
        'email',
        'first_name',
        'last_name',
        'full_name',
        'gender',
        'status',
        'vanity',
        'about',
        'facebook_id',
        'image_url',
        'thumb_url',
        'thumbnails',
        'youtube',
        'twitter',
        'facebook',
        'twitch',
        'is_suspended',
        'is_deleted',
        'is_nuked',
        'created',
        'url',
        'like_count',
        'comment_count',
        'is_creator',
        'hide_pledges',
        'two_factor_enabled',
    ]


@pytest.fixture
def relationships():
    return [
        'pledges',
        'cards',
        'follows',
        'campaign',
        'presence',
        'session',
        'locations',
        'current_user_follow',
        'pledge_to_current_user',
    ]


def test_schema_attributes_are_properly_formatted(attributes):
    for attribute_name in attributes:
        value = getattr(user.Attributes, attribute_name, None)
        assert value is not None and value is attribute_name


def test_schema_relationships_are_properly_formatted(relationships):
    for relationship_name in relationships:
        value = getattr(user.Relationships, relationship_name, None)
        assert value is not None and value is relationship_name


def test_schema_has_expected_default_attributes():
    assert user.Attributes.email in user.default_attributes
    assert user.Attributes.first_name in user.default_attributes
    assert user.Attributes.last_name in user.default_attributes
    assert user.Attributes.full_name in user.default_attributes
    assert user.Attributes.gender in user.default_attributes
    assert user.Attributes.status in user.default_attributes
    assert user.Attributes.vanity in user.default_attributes
    assert user.Attributes.about in user.default_attributes
    assert user.Attributes.facebook_id in user.default_attributes
    assert user.Attributes.image_url in user.default_attributes
    assert user.Attributes.thumb_url in user.default_attributes
    assert user.Attributes.thumbnails in user.default_attributes
    assert user.Attributes.youtube in user.default_attributes
    assert user.Attributes.twitter in user.default_attributes
    assert user.Attributes.facebook in user.default_attributes
    assert user.Attributes.twitch in user.default_attributes
    assert user.Attributes.is_suspended in user.default_attributes
    assert user.Attributes.is_deleted in user.default_attributes
    assert user.Attributes.is_nuked in user.default_attributes
    assert user.Attributes.created in user.default_attributes
    assert user.Attributes.url in user.default_attributes


def test_schema_has_expected_default_relationships():
    assert user.Relationships.campaign in user.default_relationships
    assert user.Relationships.pledges in user.default_relationships
