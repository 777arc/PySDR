try:
    # python2
    from urllib import urlencode
    from urlparse import urlparse, parse_qs
except ImportError:
    # python3
    from urllib.parse import urlencode, urlparse, parse_qs
