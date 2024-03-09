# -*- coding: utf-8 -*-
# textbook documentation build configuration file, created by sphinx-quickstart on Thu Nov 15 16:36:13 2018.

import os
from datetime import datetime
import patreon

################
# Patreon part #
################
creator_id = os.environ.get('CREATOR_ID') # Creator's Access Token from https://www.patreon.com/portal/registration/register-clients
if creator_id:
    api_client = patreon.API(creator_id)
    #data = api_client.fetch_campaign_and_patrons().json_data
    #patron_count = data['data'][0]['attributes']['patron_count']
    #print("patron count:", patron_count)

    # Get list of all patrons
    campaign_id = api_client.fetch_campaign().data()[0].id()
    pledges_response = api_client.fetch_page_of_pledges(campaign_id, 50) # 2nd arg is number of pledges per page
    names = []
    for pledge in pledges_response.data():
        patron_id = pledge.relationship('patron').id()
        patron = pledges_response.find_resource_by_type_and_id('user', patron_id)
        full_name = patron.attribute('full_name')
        # Manual substitutions to make it look nicer
        full_name = full_name.replace("Jon Kraft, Analog Devices", "Jon Kraft")
        full_name = full_name.replace("vince baker", "Vince Baker")
        if full_name == "Дмитрий Ступаков":
            continue
        names.append(full_name) # there's also 'first_name' which might be better for a public display name
    # Patreon Supporters
    html_string = ''
    html_string += '<div style="font-size: 120%; margin-top: 5px;">A big thanks to all PySDR<br><a href="https://www.patreon.com/PySDR" target="_blank">Patreon</a> supporters:</div>'
    html_string += '<div style="font-size: 120%; margin-bottom: 80px; margin-top: 0px;">'
    for name in names:
        html_string += '&#9900; ' + name + "<br />"
    # Organizations that are sponsoring (Manually added to get logo included)
    html_string += '<div style="margin-top: 5px;">and organization-level supporters:</div>'
    html_string += '<img width="12px" height="12px" src="https://pysdr.org/_static/adi.svg">' + ' <a style="border-bottom: 0;" target="_blank" href="https://www.analog.com/en/design-center/reference-designs/circuits-from-the-lab/cn0566.html">Analog Devices, Inc.</a>' + "<br />"
    html_string += "</div>"
    with open("_templates/patrons.html", "w") as patron_file:
        patron_file.write(html_string)
else:
    print("\n=====================================================")
    print("Warning- CREATOR_ID wasn't set, skipping patron list")
    print("=====================================================\n")
    with open("_templates/patrons.html", "w") as patron_file:
        patron_file.write('')


###############################
# -- General configuration ----
###############################

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.imgmath',
    #'sphinx.ext.autosectionlabel', #added for dutch
    #'sphinxcontrib.tikz', #added for dutch
]
imgmath_image_format = 'svg' # way better looking than pngs (its vectorized after all!)
imgmath_embed = True #turned this on since latest update broke html formula output, generated wrong svg src links.
imgmath_font_size = 14 # default is 12 and it looked a bit small

# Additional LaTeX code to put into the preamble of the LaTeX files used to translate the math snippets. This is left empty by default. Use it e.g. to add packages which modify the fonts used for math, such as '\\usepackage{newtxsf}' for sans-serif fonts, or '\\usepackage{fouriernc}' for serif fonts. Indeed, the default LaTeX math fonts have rather thin glyphs which (in HTML output) often do not match well with the font for text.
# The code below makes math equations not right-align which was so ugly
imgmath_latex_preamble = r'''
\makeatletter
\renewenvironment{aligned}[1][c]{%
    \alignedspace@left
    \if #1t\vtop \else \if#1b\vbox \else \vcenter \fi\fi \bgroup
        \Let@ \chardef\dspbrk@context\@ne \restore@math@cr
        \spread@equation
        \ialign\bgroup
            \hfil\strut@$\m@th\displaystyle##$\hfil
            \crcr
}{%
  \crcr\egroup
  \restorecolumn@
  \egroup
}
\makeatother
'''

# dvisvgm is the program used to generate the svg equations, and these are its settings:
# https://dvisvgm.de/Manpage/
#imgmath_dvisvgm_args = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document
master_doc = 'index'

# General information about the project.
project = u'PySDR: A Guide to SDR and DSP using Python'
year = str(datetime.now().year)
copyright = year + u', Marc Lichtman'
author = u'Marc Lichtman'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u'0.1'
# The full version, including alpha/beta/rc tags.
release = u'0.1'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'index-fr.rst', 'content-fr/*', 'index-nl.rst', 'content-nl/*', 'index-ukraine.rst', 'content-ukraine/*', 'index-zh.rst', 'content-zh/*']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
#todo_include_todos = False

# Auto add Figure X to captions
numfig = True
numfig_format = {'figure': '%s', 
                 'table': 'Table %s', 
                 'code-block': 'Listing %s',
                 'section': 'Section %s'}

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'alabaster'

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
        'patrons.html', # a custom template, has to be in the _templates dir, but for pysdr it gets auto-generated by python build script
        #'relations.html', # doesnt seem to show anything
    ]
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {'description':'By <a href="https://pysdr.org/content/about_author.html">Dr. Marc Lichtman</a>',
                      'logo': 'logo.svg',
                      'logo_name': True, # used if the logo doesn't contain the project name itself
                      'fixed_sidebar': True, # on smaller screens you can't see the whole sidebar, and it won't scroll
                      'page_width': 'auto', # makes body width fill the window nicely, but for some reason there's a max around 1000px
                      'sidebar_width': '290px', # width of sidebar
                      'show_powered_by': False,
                      'show_relbars': True} # previous and next links at top and bottom (can also use show_relbar_bottom)


# This code makes external links show up in new tabs
from sphinx.writers.html import HTMLTranslator
from docutils import nodes
from docutils.nodes import Element
class PatchedHTMLTranslator(HTMLTranslator):
    def visit_reference(self, node: Element) -> None:
        atts = {'class': 'reference'}
        if node.get('internal') or 'refuri' not in node:
            atts['class'] += ' internal'
        else:
            atts['class'] += ' external'
            # Customize behavior (open in new tab, secure linking site)
            atts['target'] = '_blank'
            atts['rel'] = 'noopener noreferrer'
        if 'refuri' in node:
            atts['href'] = node['refuri'] or '#'
            if self.settings.cloak_email_addresses and atts['href'].startswith('mailto:'):
                atts['href'] = self.cloak_mailto(atts['href'])
                self.in_mailto = True
        else:
            assert 'refid' in node, \
                   'References must have "refuri" or "refid" attribute.'
            atts['href'] = '#' + node['refid']
        if not isinstance(node.parent, nodes.TextElement):
            assert len(node) == 1 and isinstance(node[0], nodes.image)
            atts['class'] += ' image-reference'
        if 'reftitle' in node:
            atts['title'] = node['reftitle']
        if 'target' in node:
            atts['target'] = node['target']
        self.body.append(self.starttag(node, 'a', '', **atts))

        if node.get('secnumber'):
            self.body.append(('%s' + self.secnumber_suffix) %
                             '.'.join(map(str, node['secnumber'])))
def setup(app):
    app.set_translator('html', PatchedHTMLTranslator)
    
# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents which ends up being the browser tab names.  If None, it defaults to "<project> v<release> documentation".
html_title = project # so it leaves out version and "documentation", didn't want to make it any shorter and risk SEO issues

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path or fully qualified paths (eg. https://...)
html_css_files = ['custom.css',]

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True


# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'textbookdoc'

# -- Options for LaTeX output ---------------------------------------------
tikz_latex_preamble = r'\usepackage{circuitikz}'
#imgmath_latex_preamble = r'\usepackage{siunitx}'


latex_elements = {
#    'extrapackages': r'\usepackage{siunitx}',
#    'passoptionstopackages': r'\PassOptionsToPackage{svgnames}{xcolor}',
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',

# Latex figure (float) alignment
#'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'textbook.tex', u'textbook Documentation',
     u'Marc Lichtman', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'textbook', u'textbook Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'textbook', u'textbook Documentation',
     author, 'textbook', 'One line description of project.',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False

# I needed to add this in order for the custom.css to be processed
rst_prolog = """
.. include:: <s5defs.txt>
.. default-role::

"""
