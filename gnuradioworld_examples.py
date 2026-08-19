"""Collect every GNU Radio World embed in the textbook and build the list shown on the homepage.

Each chapter embeds GNU Radio World flowgraphs with a raw HTML iframe whose src looks like:

    https://gnuradioworld.com/?embed=1&zoom=60%#example=filter/filters_lowpass_interference

This script scans content/*.rst for those iframes, sorts them by the chapter order in
index.rst, and writes _templates/homepage_generated.html, which is homepage.html with the
GNU RADIO WORLD EXAMPLE LIST marker replaced by a collapsed <details> list.  index.rst
includes the generated file, so homepage.html itself is never modified by a build.

It is called from conf.py so it runs on every build/deploy, the same way scrape_patreon does.
"""

import html
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
CONTENT_DIR = os.path.join(ROOT, 'content')
INDEX_RST = os.path.join(ROOT, 'index.rst')
HOMEPAGE_IN = os.path.join(ROOT, '_templates', 'homepage.html')
HOMEPAGE_OUT = os.path.join(ROOT, '_templates', 'homepage_generated.html')

MARKER = '<!-- GNU RADIO WORLD EXAMPLE LIST -->'
EXAMPLE_URL = 'https://gnuradioworld.com/#example={}'

IFRAME_PATTERN = re.compile(
    r'<iframe\b[^>]*?src="https://gnuradioworld\.com/[^"]*?#example=(?P<example>[^"&]+)"'
    r'(?P<rest>[^>]*?)>',
    re.DOTALL,
)
TITLE_PATTERN = re.compile(r'title="(?P<title>[^"]*)"')


def _chapter_order():
    """Chapter file names (without extension) in the order they appear in index.rst's toctree."""
    with open(INDEX_RST, encoding='utf-8') as f:
        index_text = f.read()
    return [os.path.basename(m) for m in re.findall(r'^\s+content/(\S+)\s*$', index_text, re.MULTILINE)]


def _chapter_title(rst_text, fallback):
    """The chapter's title, i.e., the line wrapped in a ###### overline/underline."""
    match = re.search(r'^#{3,}\s*\n(?P<title>.+?)\n#{3,}\s*$', rst_text, re.MULTILINE)
    if match:
        return match.group('title').strip()
    return fallback.replace('_', ' ').title()


def _collect_examples():
    """Every GNU Radio World embed in the textbook, in chapter order, then order of appearance."""
    order = _chapter_order()
    rst_files = sorted(f for f in os.listdir(CONTENT_DIR) if f.endswith('.rst'))
    rst_files.sort(key=lambda f: order.index(f[:-4]) if f[:-4] in order else len(order))

    examples = []
    seen = set()
    for rst_file in rst_files:
        with open(os.path.join(CONTENT_DIR, rst_file), encoding='utf-8') as f:
            rst_text = f.read()
        if 'gnuradioworld.com' not in rst_text:
            continue
        chapter = _chapter_title(rst_text, rst_file[:-4])
        for match in IFRAME_PATTERN.finditer(rst_text):
            example = match.group('example')
            if example in seen:  # the same flowgraph embedded twice only gets listed once
                continue
            seen.add(example)
            title_match = TITLE_PATTERN.search(match.group('rest'))
            title = title_match.group('title').strip() if title_match else example
            title = re.sub(r'^PySDR:\s*', '', title)  # the iframe titles are all prefixed with "PySDR: "
            examples.append((chapter, title, example))
    return examples


def _list_html(examples):
    lines = [
        MARKER,
        '<details style="margin: 10px 0 20px 0;">',
        '  <summary style="cursor: pointer;"><h4 style="display: inline;">Expand for a list of all GNU Radio World examples in PySDR</h4></summary>',
        '  <ul style="margin-top: 10px;">',
    ]
    for chapter, title, example in examples:
        lines.append(
            '    <li><a class="reference external" href="{}" rel="noopener noreferrer" target="_blank">{}</a> '
            '<span style="color: #777;">({})</span></li>'.format(
                html.escape(EXAMPLE_URL.format(example)), html.escape(title), html.escape(chapter)
            )
        )
    lines += ['  </ul>', '</details>']
    return '\n'.join(lines)


def generate_gnuradioworld_list():
    with open(HOMEPAGE_IN, encoding='utf-8') as f:
        homepage = f.read()

    examples = _collect_examples()
    print('Found {} GNU Radio World examples throughout the textbook'.format(len(examples)))

    if MARKER not in homepage:
        # Don't break the build over it, the homepage just won't have the list
        print('WARNING: {} not found in {}, skipping the example list'.format(MARKER, HOMEPAGE_IN))
        generated = homepage
    elif examples:
        generated = homepage.replace(MARKER, _list_html(examples))
    else:
        generated = homepage  # no examples found, leave the marker comment in place

    with open(HOMEPAGE_OUT, 'w', encoding='utf-8') as f:
        f.write(generated)


if __name__ == '__main__':
    generate_gnuradioworld_list()
