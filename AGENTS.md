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

## Notes

- This project uses Sphinx and a custom `conf.py`.
- The repository already contains scripts and generated assets for many figures, so prefer reusing existing conventions instead of introducing new build patterns.
