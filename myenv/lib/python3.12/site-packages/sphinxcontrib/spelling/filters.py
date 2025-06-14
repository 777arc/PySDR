#
# Copyright (c) 2010 Doug Hellmann.  All rights reserved.
#
"""Spelling checker extension for Sphinx.
"""

# TODO - Words with multiple uppercase letters treated as classes and ignored

import builtins
import importlib
import subprocess
import sys
from xmlrpc import client as xmlrpc_client

from enchant.tokenize import Filter, get_tokenizer, tokenize, unit_tokenize
from sphinx.util import logging

logger = logging.getLogger(__name__)


class AcronymFilter(Filter):
    """If a word looks like an acronym (all upper case letters),
    ignore it.
    """
    def _skip(self, word):
        return (
            word.isupper() or  # all caps
            # pluralized acronym ("URLs")
            (word[-1].lower() == 's' and word[:-1].isupper())
        )


class list_tokenize(tokenize):

    def __init__(self, words):
        super().__init__('')
        self._words = words

    def next(self):
        if not self._words:
            raise StopIteration()
        word = self._words.pop(0)
        return (word, 0)


class ContractionFilter(Filter):
    """Strip common contractions from words.
    """
    splits = {
        "aren't": ["are", "not"],
        "can't": ["can", "not"],
        "could've": ["could", "have"],
        "couldn't": ["could", "not"],
        "didn't": ["did", "not"],
        "doesn't": ["does", "not"],
        "don't": ["do", "not"],
        "hadn't": ["had", "not"],
        "hasn't": ["has", "not"],
        "haven't": ["have", "not"],
        "he'd": ["he", "would"],
        "he'll": ["he", "will"],
        "he's": ["he", "is"],
        "how'd": ["how", "would"],
        "how'll": ["how", "will"],
        "how's": ["how", "is"],
        "i'd": ["I", "would"],
        "i'll": ["I", "will"],
        "i'm": ["I", "am"],
        "i've": ["I", "have"],
        "isn't": ["is", "not"],
        "it'd": ["it", "would"],
        "it'll": ["it", "will"],
        "it's": ["it", "is"],
        "ma'am": ["madam"],
        "might've": ["might", "have"],
        "mightn't": ["might", "not"],
        "must've": ["must", "have"],
        "mustn't": ["must", "not"],
        "o'": ["of"],
        "o'clock": ["of", "the", "clock"],
        "she'd": ["she", "would"],
        "she'll": ["she", "will"],
        "she's": ["she", "is"],
        "should've": ["should", "have"],
        "shouldn't": ["should", "not"],
        "that'd": ["that", "would"],
        "that'll": ["that", "will"],
        "that's": ["that", "is"],
        "they'd": ["they", "would"],
        "they'll": ["they", "will"],
        "they're": ["they", "are"],
        "they've": ["they", "have"],
        "wasn't": ["was", "not"],
        "we'd": ["we", "would"],
        "we'll": ["we", "will"],
        "we're": ["we", "are"],
        "we've": ["we", "have"],
        "weren't": ["were", "not"],
        "what'd": ["what", "would"],
        "what'll": ["what", "will"],
        "what're": ["what", "are"],
        "what's": ["what", "is"],
        "when'd": ["when", "would"],
        "when'll": ["when", "will"],
        "when's": ["when", "is"],
        "where'd": ["where", "would"],
        "where'll": ["where", "will"],
        "where's": ["where", "is"],
        "who'd": ["who", "would"],
        "who'll": ["who", "will"],
        "who's": ["who", "is"],
        "why'd": ["why", "would"],
        "why'll": ["why", "will"],
        "why's": ["why", "is"],
        "won't": ["will", "not"],
        "would've": ["would", "have"],
        "wouldn't": ["would", "not"],
        "you'd": ["you", "would"],
        "you'll": ["you", "will"],
        "you're": ["you", "are"],
        "you've": ["you", "have"],
    }

    def _split(self, word):
        # Fixed responses
        if word.lower() in self.splits:
            return list_tokenize(self.splits[word.lower()])

        # Possessive
        if word.lower().endswith("'s"):
            return unit_tokenize(word[:-2])

        # * not
        if word.lower().endswith("n't"):
            return unit_tokenize(word[:-3])

        return unit_tokenize(word)


class IgnoreWordsFilter(Filter):
    """Given a set of words, ignore them all.
    """

    def __init__(self, tokenizer, word_set):
        self.word_set = set(word_set)
        super().__init__(tokenizer)

    def _skip(self, word):
        return word in self.word_set


class IgnoreWordsFilterFactory:

    def __init__(self, words):
        self.words = words

    def __call__(self, tokenizer):
        return IgnoreWordsFilter(tokenizer, self.words)


class PyPIFilterFactory(IgnoreWordsFilterFactory):
    """Build an IgnoreWordsFilter for all of the names of packages on PyPI.
    """
    def __init__(self):
        client = xmlrpc_client.ServerProxy('https://pypi.python.org/pypi')
        super().__init__(client.list_packages())


class PythonBuiltinsFilter(Filter):
    """Ignore names of built-in Python symbols.
    """
    def _skip(self, word):
        return hasattr(builtins, word)


class ImportableModuleFilter(Filter):
    """Ignore names of modules that we could import.
    """
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.found_modules = set(sys.builtin_module_names)
        self.sought_modules = self.found_modules.copy()
        # By adding __main__ to the list of sought modules but not
        # found modules we ensure that it is never recognized as a
        # valid module, which is consistent with the behavior before
        # version 7.3.1.  See
        # https://github.com/sphinx-contrib/spelling/issues/141
        self.sought_modules.add('__main__')

    def _skip(self, word):
        # If the word looks like a python module filename, strip the
        # extension to avoid the side-effect of actually importing the
        # module. This prevents, for example, 'setup.py' triggering an
        # import of the setup module during a doc build, which makes
        # it look like Sphinx is complaining about a commandline
        # argument. See
        # https://github.com/sphinx-contrib/spelling/issues/142
        if word.endswith('.py'):
            logger.debug(
                'removing .py extension from %r before searching for module',
                word)
            word = word[:-3]

        valid_module_name = all(n.isidentifier() for n in word.split('.'))
        if not valid_module_name:
            return False

        if word not in self.sought_modules:
            self.sought_modules.add(word)
            try:
                mod = importlib.util.find_spec(word)
            except BaseException as err:
                # This could be an ImportError, SystemExit, some more detailed
                # error out of distutils, or something else triggered
                # by failing to be able to import a parent package to
                # use the metadata to search for a subpackage.
                logger.debug('find_spec(%r) failed, invalid module name: %s',
                             word, err)
            else:
                if mod is not None:
                    self.found_modules.add(word)

        return word in self.found_modules


class ContributorFilter(IgnoreWordsFilter):
    """Accept information about contributors as spelled correctly.

    Look in the git history for authors and committers and accept
    tokens that are in the set.
    """

    _pretty_format = (
        '%(trailers:key=Co-Authored-By,separator=%x0A)%x0A%an%x0A%cn'
    )

    def __init__(self, tokenizer):
        contributors = self._get_contributors()
        super().__init__(tokenizer, contributors)

    def _get_contributors(self):
        logger.info('Scanning contributors')
        cmd = ['git', 'log', '--quiet', '--no-color',
               f'--pretty=format:{self._pretty_format}']

        try:
            p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
        except subprocess.CalledProcessError as err:
            logger.warning('Called: %s', ' '.join(cmd))
            logger.warning('Failed to scan contributors: %s', err)
            return set()
        output = p.stdout.decode('utf-8')
        tokenizer = get_tokenizer('en_US', filters=[])
        return {word for word, pos in tokenizer(output)}
