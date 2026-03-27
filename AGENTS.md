# Repository Guidance for AI Contributors

This repo is the source for the PySDR textbook, created using Sphinx.  

## Human Notes

This section is for human reference only. It is not an instruction for agents and should not change behavior.

Marc uses AI to help create the JavaScript mini-apps and solve issues like when certain things are not rendered correctly, he doesn't use it to write the actual content, other than spelling/grammar edits and scanning for bugs/incorrectness. 

## What to edit

- Edit the `.rst` files under `content/` and the Sphinx config/templates under the repo root.
- Treat `_build/` as generated output. Do not edit it directly.
- If you change a page, also check whether image assets, scripts, or config in `_static/`, `_images/`, `conf.py`, or `Makefile` need matching updates.

## How to build locally

- Activate the project virtual environment:

```bash
source /home/marc/venvs/pysdr/bin/activate
```

- Build the site using:

```bash
make fast-html
```

- The rendered HTML site will be in `_build/`.
- Open `_build/index.html` for the main site, or the relevant page under `_build/content/`.

## Practical workflow

- Make the source change.
- Run `make fast-html`.
- Inspect the generated HTML in `_build/` to verify the result.
- Keep changes minimal and aligned with the existing textbook style.

## Writing Style

Note to humans- This guidance is provided to AI to help edit, not actually write material, Marc's writes everything himself then uses AI for catching grammar issues and such.

PySDR's prose is intentionally instructional, conversational, and example-driven. When editing chapter text, match these patterns:

- Start with intuition before formalism. Explain the idea in plain language first, then introduce equations or code.
- Prefer short, direct paragraphs. Long sections are acceptable, but they should be broken up with transitions, examples, or figures.
- Use first-person plural sparingly but naturally (`we`, `let's`) to guide the reader through the material.
- Keep the tone approachable and lightly informal, but not casual or chatty.
- Use rhetorical questions and plain-English restatements when they help clarify a concept.
- Explain why a step matters, not just what the step is.
- Preserve the textbook’s teaching rhythm: concept, example, code, result, takeaway.
- Keep technical terminology precise, but avoid sounding overly academic or formal.
- When a section already has figures or code, make the surrounding prose point the reader to them and explain what they should notice.
- Avoid hype, filler, and motivational fluff.

When in doubt, read the surrounding chapter aloud mentally and aim for the same cadence and readability rather than a generic documentation voice.

## Notes

- This project uses Sphinx and a custom `conf.py`.
- The repository already contains scripts and generated assets for many figures, so prefer reusing existing conventions instead of introducing new build patterns.
