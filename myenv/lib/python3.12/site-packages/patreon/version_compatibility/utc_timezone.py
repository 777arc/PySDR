import datetime


def has_timezone_support():
    return hasattr(datetime, 'timezone')


def utc_timezone():
    if has_timezone_support():
        return datetime.timezone.utc

    # Fallback to a mock Python 3 timezone for Python 2
    class UTCTimezone(datetime.tzinfo):
        def __init__(self):
            pass

        def utcoffset(self, dt):
            return datetime.timedelta(0)

        def dst(self, dt):
            return datetime.timedelta(0)

        def tzname(self, dt):
            return "UTC"

    return UTCTimezone()
