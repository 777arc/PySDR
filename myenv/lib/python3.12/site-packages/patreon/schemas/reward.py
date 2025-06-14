class Attributes(object):
    amount = 'amount'
    amount_cents = 'amount_cents'
    user_limit = 'user_limit'
    remaining = 'remaining'
    description = 'description'
    requires_shipping = 'requires_shipping'
    created_at = 'created_at'
    url = 'url'
    patron_count = 'patron_count'


class Relationships(object):
    creator = 'creator'


default_attributes = [
    Attributes.amount,
    Attributes.amount_cents,
    Attributes.user_limit,
    Attributes.remaining,
    Attributes.description,
    Attributes.requires_shipping,
    Attributes.created_at,
    Attributes.url,
    Attributes.patron_count,
]

default_relationships = [
    Relationships.creator
]
