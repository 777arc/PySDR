class Attributes(object):
    amount_cents = 'amount_cents'
    total_historical_amount_cents = 'total_historical_amount_cents'
    declined_since = 'declined_since'
    created_at = 'created_at'
    pledge_cap_cents = 'pledge_cap_cents'
    patron_pays_fees = 'patron_pays_fees'
    unread_count = 'unread_count'


class Relationships(object):
    patron = 'patron'
    reward = 'reward'
    creator = 'creator'
    address = 'address'
    card = 'card'
    pledge_vat_location = 'pledge_vat_location'


default_attributes = [
    Attributes.amount_cents,
    Attributes.declined_since,
    Attributes.created_at,
    Attributes.pledge_cap_cents,
    Attributes.patron_pays_fees,
]

default_relationships = [
    Relationships.patron,
    Relationships.reward,
    Relationships.creator,
    Relationships.address,
    Relationships.pledge_vat_location,
]
