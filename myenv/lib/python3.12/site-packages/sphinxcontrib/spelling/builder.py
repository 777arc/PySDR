#
# Copyright (c) 2010 Doug Hellmann.  All rights reserved.
#
"""Spelling checker extension for Sphinx.
"""

import collections
import importlib
import os
import tempfile

import docutils.nodes
import docutils.utils
from sphinx.builders import Builder
from sphinx.util import logging, osutil
from sphinx.util.console import red
from sphinx.util.matching import Matcher
from sphinx.util.osutil import ensuredir

try:
    from enchant.tokenize import EmailFilter, WikiWordFilter
except ImportError as imp_exc:
    enchant_import_error = imp_exc
else:
    enchant_import_error = None

from . import checker, filters

logger = logging.getLogger(__name__)

# TODO - Words with multiple uppercase letters treated as classes and ignored


class SpellingBuilder(Builder):
    """
    Spell checks a document
    """
    name = 'spelling'

    def init(self):
        if enchant_import_error is not None:
            raise RuntimeError(
                'Cannot initialize spelling builder '
                'without PyEnchant installed') from enchant_import_error
        self.misspelling_count = 0

        self.env.settings["smart_quotes"] = False
        # Initialize the per-document filters
        if not hasattr(self.env, 'spelling_document_words'):
            self.env.spelling_document_words = collections.defaultdict(list)

        # Initialize the global filters
        f = [
            filters.ContractionFilter,
            EmailFilter,
        ]
        if self.config.spelling_ignore_wiki_words:
            logger.info('Ignoring wiki words')
            f.append(WikiWordFilter)
        if self.config.spelling_ignore_acronyms:
            logger.info('Ignoring acronyms')
            f.append(filters.AcronymFilter)
        if self.config.spelling_ignore_pypi_package_names:
            logger.info('Adding package names from PyPI to local dictionary…')
            f.append(filters.PyPIFilterFactory())
        if self.config.spelling_ignore_python_builtins:
            logger.info('Ignoring Python builtins')
            f.append(filters.PythonBuiltinsFilter)
        if self.config.spelling_ignore_importable_modules:
            logger.info('Ignoring importable module names')
            f.append(filters.ImportableModuleFilter)
        if self.config.spelling_ignore_contributor_names:
            logger.info('Ignoring contributor names')
            f.append(filters.ContributorFilter)
        f.extend(self._load_filter_classes(self.config.spelling_filters))

        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        word_list = self.get_wordlist_filename()
        logger.info('Looking for custom word list in %s', word_list)

        self.checker = checker.SpellingChecker(
            lang=self.config.spelling_lang,
            tokenizer_lang=self.config.tokenizer_lang,
            suggest=self.config.spelling_show_suggestions,
            word_list_filename=word_list,
            filters=f,
            context_line=self.config.spelling_show_whole_line,
        )

    def _load_filter_classes(self, filters):
        # Filters may be expressed in the configuration file using
        # names, so look through them and import the referenced class
        # and use that in the checker.
        for filter_ in filters:
            if not isinstance(filter_, str):
                yield filter_
                continue
            module_name, _, class_name = filter_.rpartition('.')
            mod = importlib.import_module(module_name)
            yield getattr(mod, class_name)

    def get_configured_wordlist_filenames(self):
        "Returns the configured wordlist filenames."
        word_list = self.config.spelling_word_list_filename
        if word_list is None:
            word_list = ['spelling_wordlist.txt']

        if isinstance(word_list, str):
            # Wordlist is a string. Split on comma in case it came
            # from the command line, via -D, and has multiple values.
            word_list = word_list.split(',')

        return [
            os.path.join(self.srcdir, p)
            for p in word_list
        ]

    def get_wordlist_filename(self):
        "Returns the filename of the wordlist to use when checking content."
        filenames = self.get_configured_wordlist_filenames()
        if len(filenames) == 1:
            return filenames[0]
        # In case the user has multiple word lists, we combine them
        # into one large list that we pass on to the checker.
        return self._build_combined_wordlist()

    def _build_combined_wordlist(self):
        # If we have a list, the combined list is the first list plus all words
        # from the other lists. Otherwise, word_list is assumed to just be a
        # string.
        temp_dir = tempfile.mkdtemp()
        combined_word_list = os.path.join(temp_dir,
                                          'spelling_wordlist.txt')

        with open(combined_word_list, 'w', encoding='UTF-8') as outfile:
            for word_file in self.get_configured_wordlist_filenames():
                # Paths are relative
                long_word_file = os.path.join(self.srcdir, word_file)
                logger.info('Adding contents of %s to custom word list',
                            long_word_file)
                with open(long_word_file, encoding='UTF-8') as infile:
                    infile_contents = infile.readlines()
                outfile.writelines(infile_contents)

                # Check for newline, and add one if not present
                if infile_contents and not infile_contents[-1].endswith('\n'):
                    outfile.write('\n')

        return combined_word_list

    def get_outdated_docs(self):
        return 'all documents'

    def prepare_writing(self, docnames):
        return

    def get_target_uri(self, docname, typ=None):
        return ''

    def get_suggestions_to_show(self, suggestions):
        if not self.config.spelling_show_suggestions or not suggestions:
            return []
        to_show = suggestions
        try:
            n_to_show = int(self.config.spelling_suggestion_limit)
        except ValueError:
            n_to_show = 0
        if n_to_show > 0:
            to_show = suggestions[:n_to_show]
        return to_show

    def format_suggestions(self, suggestions):
        to_show = self.get_suggestions_to_show(suggestions)
        if not to_show:
            return ''
        return '[' + ', '.join('"%s"' % s for s in to_show) + ']'

    TEXT_NODES = {
        'block_quote',
        'caption',
        'paragraph',
        'list_item',
        'term',
        'definition_list_item',
        'title',
    }

    def write_doc(self, docname, doctree):
        lines = list(self._find_misspellings(docname, doctree))
        self.misspelling_count += len(lines)
        if lines:
            output_filename = os.path.join(self.outdir, f'{docname}.spelling')
            logger.info('Writing %s', output_filename)
            ensuredir(os.path.dirname(output_filename))
            with open(output_filename, 'w', encoding='UTF-8') as output:
                output.writelines(lines)

    def _find_misspellings(self, docname, doctree):

        excluded = Matcher(self.config.spelling_exclude_patterns)
        if excluded(self.env.doc2path(docname, None)):
            return
        # Build the document-specific word filter based on any good
        # words listed in spelling directives. If we have no such
        # words, we want to push an empty list of filters so that we
        # can always safely pop the filter stack when we are done with
        # this document.
        doc_filters = []
        good_words = self.env.spelling_document_words.get(docname)
        if good_words:
            logger.debug('Extending local dictionary for %s', docname)
            doc_filters.append(filters.IgnoreWordsFilterFactory(good_words))
        self.checker.push_filters(doc_filters)

        # Set up a filter for the types of nodes to ignore during
        # traversal.
        def filter(n):
            if n.tagname != '#text':
                return False
            if (n.parent and n.parent.tagname not in self.TEXT_NODES):
                return False
            # Nodes marked by the spelling:ignore role
            if hasattr(n, "spellingIgnore"):
                return False
            return True

        for node in doctree.findall(filter):
            # Get the location of the text being checked so we can
            # report it in the output file. Nodes from text that
            # comes in via an 'include' directive does not include
            # the full path, so convert all to relative path
            # for consistency.
            source, node_lineno = docutils.utils.get_source_line(node)
            source = osutil.relpath(source)

            # Check the text of the node.
            misspellings = self.checker.check(node.astext())
            for (
                word,
                suggestions,
                context_line,
                line_offset
            ) in misspellings:

                # Avoid TypeError on nodes lacking a line number
                # This happens for some node originating from docstrings
                lineno = node_lineno
                if lineno is not None:
                    lineno += line_offset

                msg_parts = [
                    f'{source}:{lineno}: ',
                    'Spell check',
                    red(word),
                ]
                if self.format_suggestions(suggestions) != '':
                    msg_parts.append(self.format_suggestions(suggestions))
                msg_parts.append(context_line)
                msg = ': '.join(msg_parts) + '.'
                if self.config.spelling_warning:
                    logger.warning(msg)
                elif self.config.spelling_verbose:
                    logger.info(msg)
                yield "%s:%s: (%s) %s %s\n" % (
                    source, lineno, word,
                    self.format_suggestions(suggestions),
                    context_line,
                )

        self.checker.pop_filters()
        return

    def finish(self):
        if self.misspelling_count:
            logger.warning('Found %d misspelled words',
                           self.misspelling_count)
