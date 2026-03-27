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
      <div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
        <a href="{link}" target="_blank">
                    <img src="{image}" style="width: clamp(128px, 11vw, 180px); height: clamp(128px, 11vw, 180px); object-fit: cover; border-radius: 50%; filter: grayscale(100%);">
        </a>
                <p style="margin: 8px 0 0; font-size: 0.66em; line-height: 1.2;"><strong>{name}</strong></p>
      </div>"""


def make_card(author):
    return CARD_TEMPLATE.format(
        link=author.get("link", "#"),
        image=author["image"],
        name=author["name"],
    )


def main():
    current_hash = hashlib.md5(AUTHORS_YAML.read_bytes()).hexdigest()
    if HASH_CACHE.exists() and OUTPUT_QMD.exists():
        if HASH_CACHE.read_text().strip() == current_hash:
            print("authors_gallery.qmd is up to date, skipping.")
            return

    with open(AUTHORS_YAML) as f:
        data = yaml.safe_load(f)

    contributors = []
    supervisors = []
    for author in data["authors"]:
        if not author.get("image"):
            continue
        if author.get("supervisor"):
            supervisors.append(author)
        else:
            contributors.append(author)

    contrib_cards = "\n".join(make_card(a) for a in contributors)
    super_cards = "\n".join(make_card(a) for a in supervisors)

    lines = [
        '```{=html}',
        '<div style="display: flex; flex-direction: column; justify-content: space-between; align-items: stretch; width: 100%; height: 100%; min-height: 58vh; box-sizing: border-box; padding: 1.2vh 4vw 1.8vh;">',
        '  <div style="flex: 1; display: flex; flex-wrap: nowrap; justify-content: space-between; align-items: center; gap: 12px;">',

        contrib_cards,
        '  </div>',
        '  <div style="height: 1.5vh;"></div>',
        '  <div style="flex: 1; display: flex; flex-wrap: nowrap; justify-content: space-between; align-items: center; gap: 12px;">',

        super_cards,
        '  </div>',
        '</div>',
        '```',
    ]
    gallery_html = "\n".join(lines)

    OUTPUT_QMD.write_text(gallery_html + "\n")
    HASH_CACHE.write_text(current_hash + "\n")
    print(f"Generated {OUTPUT_QMD}")


if __name__ == "__main__":
    main()
