class Attributes(object):
    email = 'email'
    first_name = 'first_name'
    last_name = 'last_name'
    full_name = 'full_name'
    gender = 'gender'
    status = 'status'
    vanity = 'vanity'
    about = 'about'
    facebook_id = 'facebook_id'
    image_url = 'image_url'
    thumb_url = 'thumb_url'
    thumbnails = 'thumbnails'
    youtube = 'youtube'
    twitter = 'twitter'
    facebook = 'facebook'
    twitch = 'twitch'
    is_suspended = 'is_suspended'
    is_deleted = 'is_deleted'
    is_nuked = 'is_nuked'
    created = 'created'
    url = 'url'
    like_count = 'like_count'
    comment_count = 'comment_count'
    is_creator = 'is_creator'
    hide_pledges = 'hide_pledges'
    two_factor_enabled = 'two_factor_enabled'


class Relationships(object):
    pledges = 'pledges'
    cards = 'cards'
    follows = 'follows'
    campaign = 'campaign'
    presence = 'presence'
    session = 'session'
    locations = 'locations'
    current_user_follow = 'current_user_follow'
    pledge_to_current_user = 'pledge_to_current_user'


default_attributes = [
    Attributes.email,
    Attributes.first_name,
    Attributes.last_name,
    Attributes.full_name,
    Attributes.gender,
    Attributes.status,
    Attributes.vanity,
    Attributes.about,
    Attributes.facebook_id,
    Attributes.image_url,
    Attributes.thumb_url,
    Attributes.thumbnails,
    Attributes.youtube,
    Attributes.twitter,
    Attributes.facebook,
    Attributes.twitch,
    Attributes.is_suspended,
    Attributes.is_deleted,
    Attributes.is_nuked,
    Attributes.created,
    Attributes.url,
]

default_relationships = [
    Relationships.campaign,
    Relationships.pledges,
]
