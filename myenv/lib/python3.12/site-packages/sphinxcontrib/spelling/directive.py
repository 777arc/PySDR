#
# Copyright (c) 2010 Doug Hellmann.  All rights reserved.
#
"""Spelling checker extension for Sphinx.
"""

import collections

from docutils.parsers import rst
from sphinx.util import logging

logger = logging.getLogger(__name__)


def add_good_words_to_document(env, docname, good_words):
    # Initialize the per-document good words list
    if not hasattr(env, 'spelling_document_words'):
        env.spelling_document_words = collections.defaultdict(list)
    logger.debug('Extending local dictionary for %s with %s',
                 env.docname, good_words)
    env.spelling_document_words[env.docname].extend(good_words)


class SpellingDirective(rst.Directive):
    """Custom directive for passing instructions to the spelling checker.

    .. spelling::

       word1
       word2

    """

    has_content = True

    def run(self):
        env = self.state.document.settings.env

        good_words = []
        for entry in self.content:
            if not entry:
                continue
            good_words.extend(entry.split())
        if good_words:
            add_good_words_to_document(env, env.docname, good_words)

        return []


class LegacySpellingDirective(SpellingDirective):

    def run(self):
        logger.info('direct use of the spelling directive is deprecated, '
                    'replace ".. spelling::" with ".. spelling:word-list::"')
        return super().run()
