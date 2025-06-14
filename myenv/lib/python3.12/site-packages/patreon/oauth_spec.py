import mock
import pytest

from patreon import oauth
from patreon.utils import user_agent_string

MOCK_CLIENT_ID = 'Mock Client ID'
MOCK_CLIENT_SECRET = 'Mock Client Secret'
MOCK_CODE = 'Mock Code'
MOCK_REDIRECT_URI = 'https://www.patreon.com/members'
MOCK_REFRESH_TOKEN = 'Mock Refresh Token'


@pytest.fixture
def client():
    return oauth.OAuth(MOCK_CLIENT_ID, MOCK_CLIENT_SECRET)


@mock.patch('requests.post')
def test_get_tokens_requests_tokens_as_expected(post, client):
    client.get_tokens(
        code=MOCK_CODE,
        redirect_uri=MOCK_REDIRECT_URI,
    )

    post.assert_called_once_with(
        'https://www.patreon.com/api/oauth2/token',
        params={
            'grant_type': 'authorization_code',
            'code': MOCK_CODE,
            'client_id': MOCK_CLIENT_ID,
            'client_secret': MOCK_CLIENT_SECRET,
            'redirect_uri': MOCK_REDIRECT_URI,
        },
        headers={
            'User-Agent': user_agent_string(),
        },
    )


@mock.patch('requests.post')
def test_refresh_token_gets_a_new_token_as_expected(post, client):
    client.refresh_token(
        redirect_uri=MOCK_REDIRECT_URI,
        refresh_token=MOCK_REFRESH_TOKEN,
    )

    post.assert_called_once_with(
        'https://www.patreon.com/api/oauth2/token',
        params={
            'grant_type': 'refresh_token',
            'refresh_token': MOCK_REFRESH_TOKEN,
            'client_id': MOCK_CLIENT_ID,
            'client_secret': MOCK_CLIENT_SECRET,
        },
        headers={
            'User-Agent': user_agent_string(),
        },
    )
