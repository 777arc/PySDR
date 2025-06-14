# -*- coding: utf-8 -*-

# Copyright (c) 2012-2024 by Christoph Reller. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.

#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY CHRISTOPH RELLER ''AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL CHRISTOPH RELLER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Christoph Reller.

"""
    sphinxcontrib.tikz
    ~~~~~~~~~~~~~~~~~~

    Draw pictures with the TikZ/PGF LaTeX package.

    See README.rst file for details

    Author: Christoph Reller <christoph.reller@gmail.com>
"""

__version__ = '0.4.20'

import contextlib
import tempfile
import posixpath
import shutil
import sys
import codecs
import os
import re

from errno import ENOENT
from string import Template
from subprocess import Popen, PIPE
try:
    from hashlib import sha1 as sha
except ImportError:
    from sha import sha

from docutils import nodes, utils
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import ViewList

from sphinx.errors import SphinxError
try:
    from sphinx.util.osutil import ensuredir
except ImportError:
    from sphinx.util import ensuredir

from sphinx.util.fileutil import copy_asset_file

from glob import glob

_Win_ = sys.platform[0:3] == 'win'

# TODO: Check existence of executables with subprocess.check_call


@contextlib.contextmanager
def changedir(directory):
    """Context to temporary change directory"""
    curdir = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(curdir)


def system(command, builder, outfile=None, offending=None):
    """Perform a system call, handling errors.

    :param list command: System command to run.
    :param builder: Sphinx builder object performing the call.
    :param str outfile: Output file in which to store command output.
    """
    binary = command[0]
    try:
        t = False if command[0] == 'pnmtopng' else True
        process = Popen(command, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=t)
        stdout, stderr = process.communicate()
    except OSError as err:
        if err.errno != ENOENT:   # No such file or directory
            raise
        raise TikzExtError('!%s command cannot be run' % binary)
    if process.returncode != 0:
        builder._tikz_warned = True
        message = 'Error (tikz extension):'
        if offending:
            message += f'\n\nOffending input:\n\n{offending}\n'
        message += f'\n{binary} exited with error code {process.returncode}'
        if stderr and stdout:
            message += f'\n\nstderr:\n{stderr}\n\nstdout{stdout}\n'
        else:
            message += f'\n\n{stderr}{stdout}'
        raise TikzExtError(message)
    if outfile is not None:
        with open(outfile, 'wb') as f:
            f.write(stdout)


class TikzExtError(SphinxError):
    category = 'Tikz extension error'


class tikzinline(nodes.Inline, nodes.Element):
    pass


def tikz_role(role, rawtext, text, lineno, inliner, option={}, content=[]):
    tikz = utils.unescape(text, restore_backslashes=True)
    return [tikzinline(tikz=tikz)], []


class tikz(nodes.Part, nodes.Element):
    pass


DOC_HEAD = r'''
\documentclass[12pt,tikz]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{pgfplots}
\usetikzlibrary{%s}
\pagestyle{empty}
'''

DOC_BODY = r'''
\begin{document}
%s
\end{document}
'''

OUT_EXTENSION = {
    'GhostScript': 'png',
    'ImageMagick': 'png',
    'Netpbm': 'png',
    'pdf2svg': 'svg',
    }

LATEX_ALIGN = {
    'center': 'centering',
    'left': 'raggedright',
    'right': 'raggedleft',
    }

LATEX_ALIGN_ENV = {
    'center': 'center',
    'left': 'flushleft',
    'right': 'flushright',
    }


class TikzDirective(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {'libs': directives.unchanged,
                   'alt': directives.unchanged,
                   'align': directives.unchanged,
                   'stringsubst': directives.flag,
                   'xscale': directives.unchanged,
                   'include': directives.unchanged}

    def run(self):
        node = tikz()
        captionstr = ''

        if 'include' in self.options:
            node['include'] = self.options['include']
            env = self.state.document.settings.env
            rel_filename, filename = env.relfn2path(node['include'])
            env.note_dependency(rel_filename)
            try:
                with codecs.open(filename, 'r', 'utf-8') as f:
                    node['tikz'] = '\n' + f.read() + '\n'
            except (IOError, OSError):
                return [self.state.document.reporter.warning(
                    'External Tikz file %r not found or reading '
                    'it failed' % filename, line=self.lineno)]
            if self.arguments:
                captionstr = '\n'.join(self.arguments)
        else:
            if not self.content:
                node['tikz'] = '\n'.join(self.arguments)
            else:
                node['tikz'] = '\n'.join(self.content)
                captionstr = '\n'.join(self.arguments)

        node['libs'] = self.options.get('libs', '')
        node['alt'] = self.options.get('alt', 'Figure made with TikZ')
        node['align'] = self.options.get('align', 'center')
        node['xscale'] = self.options.get('xscale', '')
        if 'stringsubst' in self.options:
            node['stringsubst'] = True
        else:
            node['stringsubst'] = False
        if node['tikz'] == '':
            return [self.state_machine.reporter.warning(
                    'Ignoring "tikz" directive without content.',
                    line=self.lineno)]

        # If we have a caption, add it as a child node.
        if captionstr:
            parsed = nodes.Element()
            self.state.nested_parse(ViewList([captionstr], source=''),
                                    self.content_offset, parsed)
            first_node = parsed[0]
            caption_node = nodes.caption(first_node.rawsource, '',
                                         *first_node.children)
            caption_node.source = first_node.source
            caption_node.line = first_node.line
            node += caption_node

        return [node]


def cleanup_tikzcode(self, node):
    tikz = node['tikz']
    tikz = tikz.replace('\r\n', '\n')
    tikz = re.sub(r'^\s*%.*$\n', '', tikz, 0, re.MULTILINE)
    tikz = re.sub(r'^\s*$\n', '', tikz, 0, re.MULTILINE)
    begin_tikz = '\\begin{tikz'
    if not tikz.startswith(begin_tikz):
        if begin_tikz not in tikz:
            tikz = '\\begin{tikzpicture}\n' + tikz + '\n\\end{tikzpicture}'

    if 'stringsubst' in node:
        tikz = Template(tikz).safe_substitute(wd=os.getcwd().replace('\\', '/'))
    return tikz


def render_tikz(self, node, libs='', stringsubst=False):
    config = self.builder.config
    # must use unique filenames for all tmpfiles to support sphinx -j
    tikz = cleanup_tikzcode(self, node)
    shasum = sha(tikz.encode('utf-8')).hexdigest()
    fname = 'tikz-%s.%s' % (shasum, OUT_EXTENSION[config.tikz_proc_suite])
    relfn = posixpath.join(self.builder.imgpath, fname)
    outfn = os.path.join(self.builder.outdir, '_images', fname)

    if os.path.isfile(outfn):
        return relfn

    if hasattr(self.builder, '_tikz_warned'):
        return None

    ensuredir(os.path.dirname(outfn))

    latex = DOC_HEAD % libs
    latex += self.builder._tikz_preamble
    if config.tikz_latex_preamble:
        latex += config.tikz_latex_preamble
    elif 'preamble' in config.latex_elements:
        latex += config.latex_elements['preamble']
    latex += DOC_BODY % tikz
    latex = latex.encode('utf-8')

    with changedir(self.builder._tikz_tempdir):

        with open('tikz-%s.tex' % shasum, 'wb') as f:
            f.write(latex)

        system([config.latex_engine, '--interaction=nonstopmode',
                'tikz-%s.tex' % shasum],
               self.builder, None, latex.decode())

        resolution = str(config.tikz_resolution)

        if config.tikz_proc_suite in ['ImageMagick', 'Netpbm']:

            system(['pdftoppm', '-r', resolution, 'tikz-%s.pdf' % shasum,
                    'tikz-%s' % shasum], self.builder)
            ppmfilename = glob('tikz-%s*.ppm' % shasum)[0]

            if config.tikz_proc_suite == "ImageMagick":
                if config.tikz_transparent:
                    convert_args = ['-fuzz', '2%', '-transparent', 'white']
                else:
                    convert_args = []

                system([which('convert'), '-trim'] + convert_args +
                       [ppmfilename, outfn], self.builder)

            elif config.tikz_proc_suite == "Netpbm":
                if config.tikz_transparent:
                    pnm_args = ['-transparent', 'rgb:ff/ff/ff']
                else:
                    pnm_args = []
                system(['pnmtopng'] + pnm_args + [ppmfilename], self.builder,
                       outfile=outfn)

        elif config.tikz_proc_suite == "GhostScript":
            ghostscript = which('ghostscript') or which('gs') or \
                which('gswin64')
            if config.tikz_transparent:
                device = "pngalpha"
            else:
                device = "png256"
            system([ghostscript, '-dBATCH', '-dNOPAUSE',
                    '-sDEVICE=%s' % device, '-sOutputFile=%s' % outfn,
                    '-r' + resolution + 'x' + resolution,
                    '-f', 'tikz-%s.pdf' % shasum], self.builder)
        elif config.tikz_proc_suite == "pdf2svg":
            system(['pdf2svg', 'tikz-%s.pdf' % shasum, outfn], self.builder)
        else:
            self.builder._tikz_warned = True
            raise TikzExtError('Error (tikz extension): Invalid configuration '
                               'value for tikz_proc_suite')

        return relfn


def html_visit_tikzinline(self, node):
    libs = self.builder.config.tikz_tikzlibraries
    libs = libs.replace(' ', '').replace('\t', '').strip(', ')
    try:
        fname = render_tikz(self, node, libs)
    except TikzExtError as exc:
        self.document.reporter.error(str(exc))
    else:
        self.body.append('<img src="%s"/>' % fname)
    raise nodes.SkipNode


def html_visit_tikz(self, node):
    libs = self.builder.config.tikz_tikzlibraries + ',' + node['libs']
    libs = libs.replace(' ', '').replace('\t', '').strip(', ')
    try:
        fname = render_tikz(self, node, libs, node['stringsubst'])
    except TikzExtError as exc:
        self.document.reporter.error(str(exc))
    else:
        # If scaling option is set, add 'width' attribute
        scale = ''
        if node['xscale']:
            scale = 'width="%s%%"' % (node['xscale'])
        style = 'text-align: %s' % self.encode(node['align'])
        tag = self.starttag(node, 'div', CLASS='figure', STYLE=style)
        self.body.append(tag.strip())
        self.body.append('<p>')
        self.body.append('<img %s src="%s" alt="%s" /></p>\n' %
                         (scale, fname, self.encode(node['alt']).strip()))


def html_depart_tikz(self, node):
    self.body.append('</div>')


def latex_visit_tikzinline(self, node):
    tikz = node['tikz']
    if tikz[0] == '[':
        cnt, pos = 1, 1
        while cnt > 0 and cnt < len(tikz):
            if tikz[pos] == '[':
                cnt = cnt + 1
            if tikz[pos] == ']':
                cnt = cnt - 1
            pos = pos + 1
        tikz = tikz[:pos] + '{' + tikz[pos:]
    else:
        tikz = '{' + tikz
    self.body.append('\\tikz' + tikz + '}')
    raise nodes.SkipNode


def latex_visit_tikz(self, node):
    tikz = cleanup_tikzcode(self, node)

    align = self.elements['figure_align']
    if self.no_latex_floats:
        align = "H"

    # If scaling option is set, enclose in resizebox
    scale_start = r""
    scale_end = r""
    scale = 0
    if node['xscale']:
        scale = int(node['xscale']) * 0.01
        scale_start = r"\resizebox{" + str(scale) + r"\columnwidth}{!}{"
        scale_end = r"}"
    tikz = scale_start + tikz + scale_end

    # Have a caption: enclose in a figure environment.
    if any(isinstance(child, nodes.caption) for child in node.children):
        self.body.append('\\begin{figure}[%s]\\%s\\capstart' %
                         (align, LATEX_ALIGN[node['align']]) + tikz)

    # No caption: place in a center environment.
    else:
        env = LATEX_ALIGN_ENV[node['align']]
        self.body.append('\\begin{%s}' % env + tikz + '\\end{%s}' % env)


def latex_depart_tikz(self, node):
    # If we have a caption, we need to add a label for any cross-referencing
    # and then close the figure environment.
    if any(isinstance(child, nodes.caption) for child in node.children):
        self.body.append(self.hypertarget_to(node))
        self.body.append('\\end{figure}')


def depart_tikzinline(self, node):
    pass


def cleanup_tempdir(app, exc):
    if exc:
        return
    if not hasattr(app.builder, '_tikz_tempdir'):
        return
    try:
        shutil.rmtree(app.builder._tikz_tempdir)
    except Exception:
        pass


def builder_inited(app):
    app.builder._tikz_tempdir = tempfile.mkdtemp()
    app.builder._tikz_preamble = ''
    config = app.builder.config
    extgfxpath = config.tikz_includegraphics_path
    if not extgfxpath == '':
        if isinstance(extgfxpath, str):
            extgfxpath = [extgfxpath]
        app.builder._tikz_preamble = "\\graphicspath{"
        for s in extgfxpath:
            if s[-1] != "/":
                s += "/"
            app.builder._tikz_preamble += '{"%s/%s"}' % (app.srcdir, s)
        app.builder._tikz_preamble += "}\n"

    if app.builder.name == "latex":
        sty_path = os.path.join(app.builder._tikz_tempdir,
                                "sphinxcontribtikz.sty")
        with open(sty_path, mode="w") as sty:
            sty.write(r"\RequirePackage{tikz}" + "\n")
            sty.write(r"\RequirePackage{amsmath}" + "\n")
            sty.write(r"\RequirePackage{amsfonts}" + "\n")
            sty.write(r"\RequirePackage{pgfplots}" + "\n")
            tikzlibs = config.tikz_tikzlibraries
            tikzlibs = tikzlibs.replace(' ', '')
            tikzlibs = tikzlibs.replace('\t', '')
            tikzlibs = tikzlibs.strip(', ')
            sty.write("\\usetikzlibrary{%s}\n" % tikzlibs)
            sty.write(app.builder._tikz_preamble)
            if config.tikz_latex_preamble:
                sty.write(config.tikz_latex_preamble + "\n")
            elif 'preamble' in config.latex_elements:
                sty.write(config.latex_elements['preamble'] + "\n")
        sty.close()

        config.latex_additional_files.append(sty_path)
        config.latex_additional_files.extend(
            config.tikz_additional_files)
        app.add_latex_package("sphinxcontribtikz")

    if app.builder.name == "html":
        additional_files = config.tikz_additional_files \
            if config.tikz_additional_files \
            else config.latex_additional_files
        if additional_files:
            for filename in additional_files:
                copy_asset_file(os.path.join(app.builder.confdir, filename),
                                app.builder._tikz_tempdir)


def which(program):
    if sys.platform == "win32" and not program.endswith(".exe"):
        program += ".exe"

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for p in os.environ["PATH"].split(os.pathsep):
            p = p.strip('"')
            exe_file = os.path.join(p, program)
            if is_exe(exe_file):
                return exe_file

    return None


def setup(app):
    app.add_enumerable_node(tikz, 'figure',
                            html=(html_visit_tikz, html_depart_tikz),
                            latex=(latex_visit_tikz, latex_depart_tikz))
    app.add_node(tikzinline,
                 html=(html_visit_tikzinline, depart_tikzinline),
                 latex=(latex_visit_tikzinline, depart_tikzinline))
    app.add_role('tikz', tikz_role)
    app.add_directive('tikz', TikzDirective)
    app.add_config_value('tikz_latex_preamble', '', 'env')
    app.add_config_value('tikz_additional_files', [], 'env')
    app.add_config_value('tikz_tikzlibraries', '', 'env')
    app.add_config_value('tikz_transparent', True, 'html')
    app.add_config_value('tikz_includegraphics_path', '', 'env')

    # fallback to another value depending what is on the system
    suite = 'pdf2svg'
    if not which('pdf2svg'):
        suite = 'GhostScript'
        if not (which('ghostscript') or which('gs') or which('gswin64')):
            suite = 'ImageMagick'
            if not which('pnmcrop'):
                suite = 'Netpbm'
    app.add_config_value('tikz_proc_suite', suite, 'html')
    app.add_config_value('tikz_resolution', 184, 'html')

    app.connect('build-finished', cleanup_tempdir)
    app.connect('builder-inited', builder_inited)

    return {
        'version': __version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True
    }
