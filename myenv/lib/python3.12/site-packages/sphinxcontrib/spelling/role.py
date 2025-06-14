from docutils import nodes

from . import directive


def spelling_word(role, rawtext, text, lineno, inliner,
                  options={}, content=[]):
    """Let the user indicate that inline text is spelled correctly."""
    env = inliner.document.settings.env
    docname = env.docname
    good_words = text.split()
    directive.add_good_words_to_document(env, docname, good_words)
    node = nodes.Text(text)
    return [node], []


def spelling_ignore(role, rawtext, text, lineno, inliner,
                    options={}, content=[]):
    """Let the user indicate that inline text is to not be spellchecked."""
    node = nodes.Text(text)
    setattr(node, "spellingIgnore", True)
    return [node], []
