#
# Copyright (c) 2010 Doug Hellmann.  All rights reserved.
#
"""Spelling checker extension for Sphinx.
"""

try:
    import enchant
    from enchant.tokenize import get_tokenizer
except ImportError as imp_exc:
    enchant_import_error = imp_exc
else:
    enchant_import_error = None


class SpellingChecker:
    """Checks the spelling of blocks of text.

    Uses options defined in the sphinx configuration file to control
    the checking and filtering behavior.
    """

    def __init__(self, lang, suggest, word_list_filename,
                 tokenizer_lang='en_US', filters=None, context_line=False):
        if enchant_import_error is not None:
            raise RuntimeError(
                'Cannot instantiate SpellingChecker '
                'without PyEnchant installed',
            ) from enchant_import_error
        if filters is None:
            filters = []
        self.dictionary = enchant.DictWithPWL(lang, word_list_filename)
        self.tokenizer = get_tokenizer(tokenizer_lang, filters=filters)
        self.original_tokenizer = self.tokenizer
        self.suggest = suggest
        self.context_line = context_line

    def push_filters(self, new_filters):
        """Add a filter to the tokenizer chain.
        """
        t = self.tokenizer
        for f in new_filters:
            t = f(t)
        self.tokenizer = t

    def pop_filters(self):
        """Remove the filters pushed during the last call to push_filters().
        """
        self.tokenizer = self.original_tokenizer

    def check(self, text):
        """Yields bad words and suggested alternate spellings.
        """
        for word, pos in self.tokenizer(text):
            correct = self.dictionary.check(word)
            if correct:
                continue

            suggestions = self.dictionary.suggest(word) if self.suggest else []
            line = line_of_index(text, pos) if self.context_line else ""
            line_offset = text.count("\n", 0, pos)

            yield word, suggestions, line, line_offset


def line_of_index(text, index):
    try:
        line_start = text.rindex("\n", 0, index) + 1
    except ValueError:
        line_start = 0
    try:
        line_end = text.index("\n", index)
    except ValueError:
        line_end = len(text)

    return text[line_start:line_end]
