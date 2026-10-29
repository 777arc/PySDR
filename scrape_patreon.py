import os
from datetime import datetime, timezone

import requests

# Patreon API v2, the v1 API (and the "patreon" PyPI package that wraps it) is being retired.
# The creator access token comes from https://www.patreon.com/portal/registration/register-clients
# and needs the "campaigns" and "campaigns.members" scopes.
API_ROOT = "https://www.patreon.com/api/oauth2/v2"

# Dan Boschen supports PySDR outside of Patreon, so he gets sorted into the list as if he
# had joined on this date, instead of always being stuck at the end
DAN_BOSCHEN = '<a href="https://dsp-coach.com" style="border-bottom: 0;" target="_blank">Dan Boschen</a>'
DAN_BOSCHEN_JOINED = datetime(2026, 4, 19, tzinfo=timezone.utc)

# needed by sphinx
def setup(app):
    return

def load_dotenv():
    """ For local builds, pull CREATOR_ID out of a .env file (gitignored) if it's not already
        in the environment, CI sets it as a real env var so this is a no-op there """
    try:
        with open(".env", encoding="utf-8") as env_file:
            lines = env_file.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def fetch_json(url, access_token, params=None):
    response = requests.get(
        url,
        params=params,
        headers={'Authorization': f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

def fetch_members(access_token):
    """ Yields the attributes dict of every member of the campaign, following the cursor
        pagination until Patreon stops handing back a next cursor """
    campaign_id = fetch_json(f"{API_ROOT}/campaigns", access_token)['data'][0]['id']
    params = {
        'fields[member]': 'full_name,patron_status,pledge_relationship_start,will_pay_amount_cents,is_free_trial,is_gifted',
        'page[count]': 100,
    }
    while True:
        payload = fetch_json(f"{API_ROOT}/campaigns/{campaign_id}/members", access_token, params)
        for member in payload.get('data', []):
            yield member.get('attributes', {})
        cursor = payload.get('meta', {}).get('pagination', {}).get('cursors', {}).get('next')
        if not cursor:
            break
        params['page[cursor]'] = cursor

def parse_joined(value):
    if not value:
        return None
    try:
        joined = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if joined.tzinfo is None:
        joined = joined.replace(tzinfo=timezone.utc)
    return joined

def display_name(full_name):
    """ Manual substitutions to make it look nicer, returns None for patrons to leave out """
    full_name = full_name.replace("Jon Kraft, Analog Devices", "Jon Kraft")
    full_name = full_name.replace("vince baker", "Vince Baker")
    full_name = full_name.replace("meh", "MH")
    if full_name == "Дмитрий Ступаков":
        return None
    if full_name == "Al Grant":
        return 'Al Grant <img width="15px" height="12px" src="https://pysdr.org/_static/kiwi-bird.svg">'
    if full_name == "Hash" or full_name == "RECESSIM":
        return f'<a href="https://www.youtube.com/@RECESSIM" style="border-bottom: 0;" target="_blank">{full_name} <img width="15px" height="12px" src="https://pysdr.org/_static/hash.svg"></a>'
    return full_name

def scrape_patreon():
    load_dotenv()
    creator_id = os.environ.get('CREATOR_ID')
    if creator_id:
        patrons = [] # list of (joined datetime, display name)
        for attributes in fetch_members(creator_id):
            # v2's members include followers and former patrons, v1's pledges didn't, so filter
            if attributes.get('patron_status') != 'active_patron':
                continue
            # active_patron still covers free tiers, free trials, paused pledges, and gifted
            # memberships that lapsed years ago but were never flipped out of active_patron,
            # so go by what they will actually be charged going forward
            if not attributes.get('will_pay_amount_cents'):
                continue
            if attributes.get('is_free_trial') or attributes.get('is_gifted'):
                continue
            name = display_name(attributes.get('full_name') or '')
            if not name:
                continue
            patrons.append((parse_joined(attributes.get('pledge_relationship_start')), name))
        patrons.append((DAN_BOSCHEN_JOINED, DAN_BOSCHEN))
        # Oldest patrons first, anyone missing a join date sorts to the bottom
        newest = datetime.max.replace(tzinfo=timezone.utc)
        patrons.sort(key=lambda patron: patron[0] or newest)
        names = [name for _, name in patrons]
        # Patreon Supporters
        html_string = ''
        html_string += '<div style="font-size: 120%; margin-top: 5px;">A big thanks to all PySDR<br><a href="https://www.patreon.com/PySDR" target="_blank">Patreon</a> supporters:</div>'
        html_string += '<div style="font-size: 120%; margin-bottom: 80px; margin-top: 0px;">'
        for name in names:
            html_string += '&#9900; ' + name + "<br />"
        # Organizations that are sponsoring (Manually added to get logo included)
        html_string += '<div style="margin-top: 5px;">and organization-level supporters:</div>'
        html_string += '<img width="12px" height="12px" src="https://pysdr.org/_static/adi.svg">' + ' <a style="border-bottom: 0;" target="_blank" href="https://www.analog.com/en/design-center/reference-designs/circuits-from-the-lab/cn0566.html">Analog Devices, Inc.</a>' + "<br />"
        html_string += "</div>"
        with open("_templates/patrons.html", "w", encoding="utf-8") as patron_file:
            patron_file.write(html_string)
    else:
        print("\n=====================================================")
        print("Warning- CREATOR_ID wasn't set, skipping patron list")
        print("=====================================================\n")
        with open("_templates/patrons.html", "w") as patron_file:
            patron_file.write('')
