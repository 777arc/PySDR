from patreon.version_compatibility import utc_timezone


def test_utc_timezone_returns_timezone_with_expected_tzname():
    assert utc_timezone.utc_timezone().tzname(None) in ['UTC', 'UTC+00:00']
