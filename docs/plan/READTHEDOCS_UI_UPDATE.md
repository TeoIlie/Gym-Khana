# Plan: Improve ReadTheDocs Visual Appeal (Low-Time Fixes)

## Context

The documentation at https://gym-khana.readthedocs.io uses `sphinx_rtd_theme` with custom CSS copied from the Godot engine docs. While functional, it looks generic and has several visual issues:
- Custom CSS still references "Godot's visual identity" in comments
- Dark mode is commented out
- Landing page is a plain bullet list with no visual hierarchy
- Dense content pages (configuration, API) lack visual breaks
- The `max-width: 1100px` centering creates an awkward floating layout on wide screens
- Custom JS uses deprecated jQuery patterns

## Recommended Approach: Switch to Furo Theme + Landing Page Refresh

### 1. Switch theme from `sphinx_rtd_theme` to `furo` (~15 min)

Furo is a modern, clean Sphinx theme with built-in dark mode, responsive design, and good typography out of the box. This is the single highest-impact change.

**Files to modify:**
- `docs/conf.py` — change `html_theme`, remove RTD-specific options, update theme config
- `docs/requirements.txt` — replace `sphinx_rtd_theme>=2.0` with `furo`
- `.readthedocs.yaml` — no changes needed (Sphinx config stays the same)

**conf.py changes:**
- Set `html_theme = "furo"`
- Remove `html_theme_options` RTD-specific keys (`logo_only`, `collapse_navigation`, `prev_next_buttons_location`)
- Add Furo-specific options: light/dark color scheme, sidebar behavior
- Keep `html_logo`, `html_favicon`, `html_context` (GitHub integration works with Furo too)
- Remove `"sphinx_rtd_theme"` from `extensions` list (Furo doesn't need it)

### 2. Replace custom CSS (~10 min)

The 652-line custom CSS is almost entirely RTD-theme overrides. With Furo, most of it is unnecessary.

**File:** `docs/_static/css/custom.css`

Replace with a minimal file (~30 lines) that only customizes:
- Brand colors (navbar accent, link colors) via Furo CSS variables
- Font stack (keep the system font stack)
- Any spacing tweaks specific to the project

### 3. Remove custom JS (~2 min)

**File:** `docs/_static/js/custom.js`

The scroll-based logo hiding is an RTD-theme workaround. Furo handles the sidebar natively — delete this file and remove `html_js_files` from `conf.py`.

### 4. Add `sphinx-design` for landing page cards (~10 min)

**Files to modify:**
- `docs/requirements.txt` — add `sphinx-design`
- `docs/conf.py` — add `"sphinx_design"` to `extensions`
- `docs/index.rst` — replace the plain bullet list with a card grid

**Landing page improvements:**
- Add a card grid for the 3 main sections (Getting Started, User Guide, API Reference) with brief descriptions
- Convert the features bullet list into a 2-column grid of feature cards with short descriptions
- Keep the demo GIF prominently placed

### 5. Add admonitions to key content pages (~10 min)

Add `.. tip::`, `.. note::`, and `.. warning::` directives to break up dense text:

- `docs/installation.rst` — tip about virtual environments, note about MPC being optional
- `docs/quickstart.rst` — tip about the default config
- `docs/configuration.rst` — notes on which options affect rewards, warnings about incompatible combinations
- `docs/known_issues.rst` — use `.. warning::` instead of plain text

### 6. Minor content polish (~5 min)

- `docs/conf.py`: update copyright, clean up commented-out options
- `docs/index.rst`: add a brief "Get started in 3 commands" snippet before the toctrees

## Files to Modify (Summary)

| File | Change |
|------|--------|
| `docs/conf.py` | Switch theme, add sphinx-design, remove JS, update options |
| `docs/requirements.txt` | `furo`, `sphinx-design` instead of `sphinx_rtd_theme` |
| `docs/_static/css/custom.css` | Replace with minimal Furo customization |
| `docs/_static/js/custom.js` | Delete |
| `docs/index.rst` | Card grid for features and sections |
| `docs/installation.rst` | Add admonitions |
| `docs/quickstart.rst` | Add admonitions |
| `docs/configuration.rst` | Add admonitions |
| `docs/known_issues.rst` | Use warning admonitions |

## Verification

1. Build docs locally: `cd docs && make html` (or `sphinx-build -b html . _build/html`)
2. Open `docs/_build/html/index.html` in browser
3. Check: dark mode toggle works, cards render, sidebar navigation works, mobile responsive
4. Verify all pages render without Sphinx warnings
5. Push to trigger ReadTheDocs build and verify live site
