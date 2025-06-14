try:
    # For Python 3.8 and later
    import importlib.metadata as importlib_metadata
except ImportError:
    # For everyone else
    import importlib_metadata

from sphinx.util import logging

from . import asset, builder, directive, domain

logger = logging.getLogger(__name__)


def setup(app):
    version = importlib_metadata.version('sphinxcontrib-spelling')
    logger.info('Initializing Spelling Checker %s', version)
    app.add_builder(builder.SpellingBuilder)
    # Register the 'spelling' directive for setting parameters within
    # a document
    app.add_directive('spelling', directive.LegacySpellingDirective)
    app.add_domain(domain.SpellingDomain)
    # Register an environment collector to merge data gathered by the
    # directive in parallel builds
    app.add_env_collector(asset.SpellingCollector)
    # Report guesses about correct spelling
    app.add_config_value('spelling_show_suggestions', False, 'env')
    # Limit the number of suggestions output
    app.add_config_value('spelling_suggestion_limit', 0, 'env')
    # Report the whole line that has the error
    app.add_config_value('spelling_show_whole_line', True, 'env')
    # Emit misspelling as a sphinx warning instead of info message
    app.add_config_value('spelling_warning', False, 'env')
    # Set the language for the text
    app.add_config_value('spelling_lang', 'en_US', 'env')
    # Set the language for the tokenizer
    app.add_config_value('tokenizer_lang', 'en_US', 'env')
    # Set a user-provided list of words known to be spelled properly
    app.add_config_value('spelling_word_list_filename', None, 'env')
    # Assume anything that looks like a PyPI package name is spelled properly
    app.add_config_value('spelling_ignore_pypi_package_names', False, 'env')
    # Assume words that look like wiki page names are spelled properly
    app.add_config_value('spelling_ignore_wiki_words', True, 'env')
    # Assume words that are all caps, or all caps with trailing s, are
    # spelled properly
    app.add_config_value('spelling_ignore_acronyms', True, 'env')
    # Assume words that are part of __builtins__ are spelled properly
    app.add_config_value('spelling_ignore_python_builtins', True, 'env')
    # Assume words that look like the names of importable modules are
    # spelled properly
    app.add_config_value('spelling_ignore_importable_modules', True, 'env')
    # Treat contributor names from git history as spelled correctly
    app.add_config_value('spelling_ignore_contributor_names', True, 'env')
    # Add any user-defined filter classes
    app.add_config_value('spelling_filters', [], 'env')
    # Set a user-provided list of files to ignore
    app.add_config_value('spelling_exclude_patterns', [], 'env')
    # Choose whether or not the misspelled output should be displayed
    # in the terminal
    app.add_config_value('spelling_verbose', True, 'env')
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
        "version": version,
    }
