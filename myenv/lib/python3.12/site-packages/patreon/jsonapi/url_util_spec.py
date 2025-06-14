from patreon.jsonapi import url_util

MOCK_URL_ENDPOINT = 'https://patreon.com'


def test_build_url_generates_url_without_params():
    assert url_util.build_url(MOCK_URL_ENDPOINT) == MOCK_URL_ENDPOINT


def test_build_url_generates_url_with_fields():
    expectation = MOCK_URL_ENDPOINT + '?fields%5Bmock%5D=test'

    assert expectation == url_util.build_url(
        MOCK_URL_ENDPOINT,
        fields={'mock': ['test']},
    )


def test_build_url_generates_url_with_includes():
    MOCK_INCLUDES = ['mock', 'include']
    expectation = MOCK_URL_ENDPOINT + '?include=' + '%2C'.join(MOCK_INCLUDES)

    assert expectation == url_util.build_url(
        MOCK_URL_ENDPOINT,
        includes=MOCK_INCLUDES,
    )


def test_build_url_generates_url_with_ampersand_if_query_params_present():
    MOCK_INCLUDES = ['example']
    expectation = MOCK_URL_ENDPOINT + '?mock=this&include=example'

    assert expectation == url_util.build_url(
        MOCK_URL_ENDPOINT + '?mock=this',
        includes=MOCK_INCLUDES,
    )
