"""Assemble the Sentry project website for GitHub Pages."""

import argparse
from pathlib import Path
from shutil import copyfile


ROOT = Path(__file__).resolve().parents[1]
SITE_FILES = (
    "index.html",
    "styles.css",
    "main.js",
    "assets/demo-poster.webp",
    "assets/recovery-art.webp",
    "assets/sentry-demo-branded.mp4",
)
PROJECT_ASSETS = (
    "assets/sentry_logo.png",
    "assets/sentry_demo.mp4",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "dist/site",
        help="Site output directory (default: dist/site in the repository)",
    )
    output = parser.parse_args().output
    sources = {name: ROOT / "site" / name for name in SITE_FILES}
    sources.update({name: ROOT / name for name in PROJECT_ASSETS})

    for source in sources.values():
        if not source.is_file():
            parser.error(f"Missing site input: {source.relative_to(ROOT)}")

    for name, source in sources.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)

    print(f"Site ready at {output.resolve()}")


if __name__ == "__main__":
    main()
