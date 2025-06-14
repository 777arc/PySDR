class Attributes(object):
    amount_cents = 'amount_cents'
    title = 'title'
    description = 'description'
    created_at = 'created_at'
    reached_at = 'reached_at'


class Relationships(object):
    pass


default_attributes = [
    Attributes.amount_cents,
    Attributes.title,
    Attributes.description,
    Attributes.created_at,
    Attributes.reached_at,
]

default_relationships = [
]
