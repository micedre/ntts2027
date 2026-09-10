"""Derive ``reference.docx`` from the untouched NTTS template.

Pandoc's docx writer looks for a table style named ``Table`` and paragraph
styles ``Table Caption`` / ``Image Caption`` in the reference document. The
NTTS template defines none of them, so tables come out border-less and
captions fall back to defaults. This script copies the template and adds
those three styles as clones of the template's own ``Table Grid`` and
``caption`` styles. The template itself is never modified.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TEMPLATE = HERE / "NTTS2027_Abstract_template (1).docx"
OUT = HERE / "reference.docx"


def clone_style(styles_xml: str, src_id: str, new_id: str, new_name: str) -> str:
    m = re.search(rf'<w:style [^>]*w:styleId="{src_id}".*?</w:style>', styles_xml, flags=re.S)
    if not m:
        raise SystemExit(f"style {src_id} not found in template")
    if re.search(rf'w:styleId="{new_id}"', styles_xml):
        return styles_xml
    clone = m.group(0)
    clone = re.sub(r'w:styleId="[^"]+"', f'w:styleId="{new_id}"', clone, count=1)
    clone = re.sub(r'<w:name w:val="[^"]+"/>', f'<w:name w:val="{new_name}"/>', clone, count=1)
    return styles_xml.replace("</w:styles>", clone + "</w:styles>")


def add_style(styles_xml: str, xml: str) -> str:
    sid = re.search(r'w:styleId="([^"]+)"', xml).group(1)
    if re.search(rf'w:styleId="{sid}"', styles_xml):
        return styles_xml
    return styles_xml.replace("</w:styles>", xml + "</w:styles>")


# Paragraph styles pandoc uses inside tables and after headings. They are not
# in the template, and pandoc does not add them, so Word would fall back to
# Normal (12 pt after each paragraph = very tall table rows).
EXTRA_PARAGRAPH_STYLES = [
    '<w:style w:type="paragraph" w:styleId="FirstParagraph"><w:name w:val="First Paragraph"/>'
    '<w:basedOn w:val="BodyText"/><w:next w:val="BodyText"/><w:qFormat/></w:style>',
    '<w:style w:type="paragraph" w:styleId="Compact"><w:name w:val="Compact"/>'
    '<w:basedOn w:val="BodyText"/><w:qFormat/>'
    '<w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr>'
    '<w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>',
]

# Bold header row + small cell margins on the cloned Table style.
TABLE_STYLE_EXTRAS = (
    '<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
)
TABLE_FIRST_ROW = '<w:tblStylePr w:type="firstRow"><w:rPr><w:b/></w:rPr></w:tblStylePr>'


def main() -> None:
    tmp = OUT.with_suffix(".tmp.docx")
    with zipfile.ZipFile(TEMPLATE) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                xml = data.decode("utf-8")
                xml = clone_style(xml, "TableGrid", "Table", "Table")
                xml = clone_style(xml, "Caption", "TableCaption", "Table Caption")
                xml = clone_style(xml, "Caption", "ImageCaption", "Image Caption")
                for extra in EXTRA_PARAGRAPH_STYLES:
                    xml = add_style(xml, extra)
                # enrich the cloned Table style
                m = re.search(r'<w:style [^>]*w:styleId="Table".*?</w:style>', xml, flags=re.S)
                tbl = m.group(0)
                if "tblCellMar" not in tbl:
                    tbl2 = tbl.replace("</w:tblBorders>", "</w:tblBorders>" + TABLE_STYLE_EXTRAS, 1)
                    tbl2 = tbl2.replace("</w:style>", TABLE_FIRST_ROW + "</w:style>", 1)
                    xml = xml.replace(tbl, tbl2)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
