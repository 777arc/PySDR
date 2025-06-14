import requests

from patreon.jsonapi.parser import JSONAPIParser
from patreon.jsonapi.url_util import build_url
from patreon.schemas import campaign
from patreon.utils import user_agent_string
from patreon.version_compatibility.utc_timezone import utc_timezone
from patreon.version_compatibility.urllib_parse import urlencode, urlparse, parse_qs


class API(object):
    def __init__(self, access_token):
        super(API, self).__init__()
        self.access_token = access_token

    def fetch_user(self, includes=None, fields=None):
        return self.__get_jsonapi_doc(
            build_url('current_user', includes=includes, fields=fields)
        )

    def fetch_campaign_and_patrons(self, includes=None, fields=None):
        if not includes:
            includes = campaign.default_relationships \
                + [campaign.Relationships.pledges]
        return self.fetch_campaign(includes=includes, fields=fields)

    def fetch_campaign(self, includes=None, fields=None):
        return self.__get_jsonapi_doc(
            build_url(
                'current_user/campaigns', includes=includes, fields=fields
            )
        )

    def fetch_page_of_pledges(
            self, campaign_id, page_size, cursor=None, includes=None,
            fields=None
    ):
        url = 'campaigns/{0}/pledges'.format(campaign_id)
        params = {'page[count]': page_size}
        if cursor:
            try:
                cursor = self.__as_utc(cursor).isoformat()
            except AttributeError:
                pass
            params.update({'page[cursor]': cursor})
        url += "?" + urlencode(params)
        return self.__get_jsonapi_doc(
            build_url(url, includes=includes, fields=fields)
        )

    @staticmethod
    def extract_cursor(jsonapi_document, cursor_path='links.next'):
        def head_and_tail(path):
            if path is None:
                return None, None
            head_tail = path.split('.', 1)
            return head_tail if len(head_tail) == 2 else (head_tail[0], None)

        if isinstance(jsonapi_document, JSONAPIParser):
            jsonapi_document = jsonapi_document.json_data

        head, tail = head_and_tail(cursor_path)
        current_dict = jsonapi_document
        while head and type(current_dict) == dict and head in current_dict:
            current_dict = current_dict[head]
            head, tail = head_and_tail(tail)

        # Path was valid until leaf, at which point nothing was found
        if current_dict is None or (head is not None and tail is None):
            return None
        # Path stopped before leaf was reached
        elif current_dict and type(current_dict) != str:
            raise Exception(
                'Provided cursor path did not result in a link', current_dict
            )

        link = current_dict
        query_string = urlparse(link).query
        parsed_query_string = parse_qs(query_string)
        if 'page[cursor]' in parsed_query_string:
            return parsed_query_string['page[cursor]'][0]
        else:
            return None

    # Internal methods
    def __get_jsonapi_doc(self, suffix):
        response_json = self.__get_json(suffix)
        if response_json.get('errors'):
            return response_json
        return JSONAPIParser(response_json)

    def __get_json(self, suffix):
        response = requests.get(
            "https://www.patreon.com/api/oauth2/api/{}".format(suffix),
            headers={
                'Authorization': "Bearer {}".format(self.access_token),
                'User-Agent': user_agent_string(),
            }
        )
        return response.json()

    @staticmethod
    def __as_utc(dt):
        if hasattr(dt, 'tzinfo'):
            if dt.tzinfo:
                return dt.astimezone(utc_timezone())
            else:
                return dt.replace(tzinfo=utc_timezone())
        return dt
