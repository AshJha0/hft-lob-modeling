# Publishing the documentation site (GitHub Pages)

This folder contains a self-contained landing page (`index.html`) for
**hft-lob-modeling**, styled identically to the sister projects
([intraday-alpha-platform](https://ashjha0.github.io/intraday-alpha-platform/),
[quant-portfolio](https://ashjha0.github.io/quant-portfolio/)).

## One-time setup

1. Push `main` to GitHub.
2. Repository **Settings → Pages**.
3. Under *Build and deployment*, choose **Source: Deploy from a branch**,
   **Branch: `main`**, **Folder: `/docs`**, then **Save**.
4. After the first build (about a minute) the site is served at
   `https://ashjha0.github.io/hft-lob-modeling/`.

## What is published

- `docs/index.html` — the landing page (links into `LEARN.md`, `COOKBOOK.md`,
  `API_SPEC.md`, `docs/ARCHITECTURE.md`, `docs/LATENCY.md` and the research
  results on GitHub).
- Everything else under `docs/` is reachable by its path, e.g.
  `https://ashjha0.github.io/hft-lob-modeling/ARCHITECTURE.md` (served raw).

## Keeping it accurate

The "Checked in CI" block on the landing page quotes the golden-case count and
the bundled-data figures stated in `README.md`. When those change, update both
places in the same commit. The page is static HTML and has no build step. Test
counts are deliberately left to CI on both pages.
