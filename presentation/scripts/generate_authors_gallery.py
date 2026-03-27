#!/usr/bin/env python3
"""Generate authors_gallery.qmd from _authors.yaml (only if changed)."""

import hashlib
import yaml
from pathlib import Path

PRESENTATION_DIR = Path(__file__).resolve().parent.parent
AUTHORS_YAML = PRESENTATION_DIR / "_authors.yaml"
OUTPUT_QMD = PRESENTATION_DIR / "authors_gallery.qmd"
HASH_CACHE = PRESENTATION_DIR / ".authors_yaml.md5"

CARD_TEMPLATE = """\
    <div style="flex: 1 1 calc(25% - 10px); margin: 5px; box-sizing: border-box; text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center; max-width: 200px; flex-shrink: 1;">
        <a href="{link}" target="_blank" style="flex-grow: 1;">
            <img src="{image}" style="width: 100%; height: auto; max-width: 100px; object-fit: cover; margin-bottom: 0; filter: grayscale(100%);">
        </a>
        <p style="font-size: 1vw; margin-top: 5px;"><strong>{name}</strong></p>
    </div>"""


def main():
    current_hash = hashlib.md5(AUTHORS_YAML.read_bytes()).hexdigest()
    if HASH_CACHE.exists() and OUTPUT_QMD.exists():
        if HASH_CACHE.read_text().strip() == current_hash:
            print(f"authors_gallery.qmd is up to date, skipping.")
            return

    with open(AUTHORS_YAML) as f:
        data = yaml.safe_load(f)

    contributors = []
    supervisors = []
    for author in data["authors"]:
        image = author.get("image", "")
        if not image:
            continue
        card = CARD_TEMPLATE.format(
            link=author.get("link", "#"),
            image=image,
            name=author["name"],
        )
        if author.get("supervisor"):
            supervisors.append(card)
        else:
            contributors.append(card)

    cards = contributors + supervisors
    gallery_html = '```{{=html}}\n\n<div style="display: flex; flex-wrap: wrap; justify-content: center; width: 100%; align-items: stretch;">\n\n{cards}\n    </div>\n```'.format(
        cards="\n    \n".join(cards)
    )

    OUTPUT_QMD.write_text(gallery_html + "\n")
    HASH_CACHE.write_text(current_hash + "\n")
    print(f"Generated {OUTPUT_QMD}")


if __name__ == "__main__":
    main()
