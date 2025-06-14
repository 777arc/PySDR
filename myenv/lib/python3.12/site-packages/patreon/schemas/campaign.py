class Attributes(object):
    summary = 'summary'
    creation_name = 'creation_name'
    pay_per_name = 'pay_per_name'
    one_liner = 'one_liner'
    main_video_embed = 'main_video_embed'
    main_video_url = 'main_video_url'
    image_small_url = 'image_small_url'
    image_url = 'image_url'
    thanks_video_url = 'thanks_video_url'
    thanks_embed = 'thanks_embed'
    thanks_msg = 'thanks_msg'
    is_monthly = 'is_monthly'
    is_nsfw = 'is_nsfw'
    is_charged_immediately = 'is_charged_immediately'
    is_charge_upfront_eligible = 'is_charge_upfront_eligible'
    is_plural = 'is_plural'
    created_at = 'created_at'
    published_at = 'published_at'
    pledge_url = 'pledge_url'
    pledge_sum = 'pledge_sum'
    patron_count = 'patron_count'
    creation_count = 'creation_count'
    outstanding_payment_amount_cents = 'outstanding_payment_amount_cents'


class Relationships(object):
    rewards = 'rewards'
    creator = 'creator'
    goals = 'goals'
    pledges = 'pledges'
    current_user_pledge = 'current_user_pledge'
    post_aggregation = 'post_aggregation'
    categories = 'categories'
    preview_token = 'preview_token'


default_attributes = [
    Attributes.summary,
    Attributes.creation_name,
    Attributes.pay_per_name,
    Attributes.one_liner,
    Attributes.main_video_embed,
    Attributes.main_video_url,
    Attributes.image_small_url,
    Attributes.image_url,
    Attributes.thanks_video_url,
    Attributes.thanks_embed,
    Attributes.thanks_msg,
    Attributes.is_monthly,
    Attributes.is_nsfw,
    Attributes.is_charged_immediately,
    Attributes.is_plural,
    Attributes.created_at,
    Attributes.published_at,
    Attributes.pledge_url,
    Attributes.pledge_sum,
    Attributes.patron_count,
    Attributes.creation_count,
    Attributes.outstanding_payment_amount_cents,
]

default_relationships = [
    Relationships.rewards,
    Relationships.creator,
    Relationships.goals,
]
