import datetime
import functools
import mock

from patreon import api
from patreon.jsonapi import url_util
from patreon.jsonapi.parser import JSONAPIParser
from patreon.utils import user_agent_string
from patreon.version_compatibility import urllib_parse
from patreon.version_compatibility.utc_timezone import utc_timezone

MOCK_CAMPAIGN_ID = 12
API_ROOT_ENDPOINT = 'https://www.patreon.com/api/oauth2/api/'
MOCK_ACCESS_TOKEN = 'mock token'
MOCK_CURSOR_VALUE = 'Mock Cursor Value'

DEFAULT_API_HEADERS = {
    'Authorization': 'Bearer ' + MOCK_ACCESS_TOKEN,
    'User-Agent': user_agent_string(),
}

client = api.API(access_token=MOCK_ACCESS_TOKEN)


def api_url(*segments, **query):
    path = '/'.join(map(str, segments))

    fields = query.get('fields', None)
    includes = query.get('includes', None)

    if fields:
        del query['fields']

    if includes:
        del query['includes']

    if query:
        path += '?' + urllib_parse.urlencode(query)

    return url_util.build_url(
        API_ROOT_ENDPOINT + path,
        fields=fields,
        includes=includes,
    )


def assert_valid_api_call(method, api_url, query=None, **kwargs):
    kwargs.setdefault('headers', DEFAULT_API_HEADERS)
    method.assert_called_once_with(api_url, **kwargs)


class MockResponse(object):
    def __init__(self, data=None, status_code=200):
        self.data = data or {}

        self.status_code = status_code

    def json(self):
        return self.data


def api_test(method='GET', **response_kwargs):
    """ Decorator to ensure API calls are made and return expected data. """

    method = method.lower()

    def api_test_factory(fn):
        @functools.wraps(fn)
        @mock.patch('requests.{}'.format(method))
        def execute_test(method_func, *args, **kwargs):
            method_func.return_value = MockResponse(**response_kwargs)

            expected_url, response = fn(*args, **kwargs)

            method_func.assert_called_once()
            assert_valid_api_call(method_func, expected_url)
            assert isinstance(response, JSONAPIParser)
            assert response.json_data is method_func.return_value.data

        return execute_test

    return api_test_factory


def test_extract_cursor_returns_cursor_when_provided():
    assert MOCK_CURSOR_VALUE == api.API.extract_cursor(
        {
            'links':
                {
                    'next':
                        'https://patreon.com/members?page[cursor]=' +
                        MOCK_CURSOR_VALUE,
                },
        }
    )


def test_extract_cursor_returns_None_when_no_cursor_provided():
    assert None is api.API.extract_cursor(
        {
            'links': {
                'next': 'https://patreon.com/members?page[offset]=25',
            },
        }
    )


def test_extract_cursor_returns_None_when_link_is_not_a_string():
    assert None is api.API.extract_cursor({
        'links': {
            'next': None,
        },
    })


def test_extract_cursor_returns_None_when_link_is_malformed():
    caught_exception = False

    try:
        api.API.extract_cursor({
            'links': {
                'next': 12,
            },
        })

    except Exception as e:
        caught_exception = True
        assert e.args[0] == 'Provided cursor path did not result in a link'

    assert caught_exception


@api_test()
def test_can_fetch_user():
    return api_url('current_user'), client.fetch_user()


@api_test()
def test_can_fetch_campaign():
    expected_url = api_url('current_user', 'campaigns')
    response = client.fetch_campaign()
    return expected_url, response


@api_test()
def test_can_fetch_api_and_patrons():
    response = client.fetch_campaign_and_patrons()

    expected_url = api_url(
        'current_user',
        'campaigns',
        includes=['rewards', 'creator', 'goals', 'pledges'],
    )

    return expected_url, response


@api_test()
def test_can_fetch_api_and_patrons_with_custom_includes():
    expected_url = api_url(
        'current_user',
        'campaigns',
        includes=['creator'],
    )

    response = client.fetch_campaign_and_patrons(
        includes=['creator'],
    )

    return expected_url, response


@api_test()
def test_can_fetch_page_of_pledges():
    PAGE_COUNT = 25

    response = client.fetch_page_of_pledges(MOCK_CAMPAIGN_ID, PAGE_COUNT)

    query_params = {'page[count]': PAGE_COUNT}

    expected_url = api_url(
        'campaigns', MOCK_CAMPAIGN_ID, 'pledges', **query_params
    )

    return expected_url, response


@api_test()
def test_can_fetch_page_of_pledges_with_arbitrary_cursor():
    PAGE_COUNT = 25
    MOCK_CURSOR = 'Mock Cursor'

    response = client.fetch_page_of_pledges(
        MOCK_CAMPAIGN_ID,
        PAGE_COUNT,
        cursor=MOCK_CURSOR,
    )

    query_params = {
        'page[count]': PAGE_COUNT,
        'page[cursor]': MOCK_CURSOR,
    }

    expected_url = api_url(
        'campaigns', MOCK_CAMPAIGN_ID, 'pledges', **query_params
    )

    return expected_url, response


@api_test()
def test_can_fetch_page_of_pledges_with_custom_options_without_tzinfo():
    PAGE_COUNT = 25
    MOCK_CURSOR = datetime.datetime.now()
    MOCK_FIELDS = {'field': ['value']}
    MOCK_INCLUDES = ['mock includes']

    EXPECTED_CURSOR = MOCK_CURSOR.replace(tzinfo=utc_timezone()).isoformat()

    response = client.fetch_page_of_pledges(
        MOCK_CAMPAIGN_ID,
        PAGE_COUNT,
        cursor=MOCK_CURSOR,
        includes=MOCK_INCLUDES,
        fields=MOCK_FIELDS,
    )

    query_params = {
        'page[count]': PAGE_COUNT,
        'page[cursor]': EXPECTED_CURSOR,
        'includes': MOCK_INCLUDES,
        'fields': MOCK_FIELDS,
    }

    expected_url = api_url(
        'campaigns', MOCK_CAMPAIGN_ID, 'pledges', **query_params
    )

    return expected_url, response


@api_test()
def test_can_fetch_page_of_pledges_with_custom_options_with_tzinfo():
    PAGE_COUNT = 25
    MOCK_CURSOR = datetime.datetime.now().replace(tzinfo=utc_timezone())
    MOCK_FIELDS = {'field': ['value']}
    MOCK_INCLUDES = ['mock includes']

    EXPECTED_CURSOR = MOCK_CURSOR.isoformat()

    response = client.fetch_page_of_pledges(
        MOCK_CAMPAIGN_ID,
        PAGE_COUNT,
        cursor=MOCK_CURSOR,
        includes=MOCK_INCLUDES,
        fields=MOCK_FIELDS,
    )

    query_params = {
        'page[count]': PAGE_COUNT,
        'page[cursor]': EXPECTED_CURSOR,
        'includes': MOCK_INCLUDES,
        'fields': MOCK_FIELDS,
    }

    expected_url = api_url(
        'campaigns', MOCK_CAMPAIGN_ID, 'pledges', **query_params
    )

    return expected_url, response
