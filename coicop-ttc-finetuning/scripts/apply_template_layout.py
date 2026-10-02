"""Give ``abstract.docx`` the page layout of the untouched NTTS template.

Pandoc drops the template's section properties, so the rendered file loses
the A4 page size, the margins and the headers and footers. This script
copies them from the template: the final ``sectPr``, ``header*.xml`` and
``footer*.xml`` with their relationships and content types. Idempotent.

Usage: python scripts/apply_template_layout.py [abstract.docx]
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TEMPLATE = HERE / "NTTS2027_Abstract_template (1).docx"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PART_RE = re.compile(r"word/(header|footer)\d+\.xml$")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "abstract.docx"
    with zipfile.ZipFile(TEMPLATE) as t:
        tdoc = t.read("word/document.xml").decode("utf-8")
        trels = t.read("word/_rels/document.xml.rels").decode("utf-8")
        ttypes = t.read("[Content_Types].xml").decode("utf-8")
        parts = {n: t.read(n) for n in t.namelist() if PART_RE.match(n)}
        part_rels = {n: t.read(n) for n in t.namelist()
                     if re.match(r"word/_rels/(header|footer)\d+\.xml\.rels$", n)}
        media = {n: t.read(n) for n in t.namelist() if n.startswith("word/media/")}
    sect = re.findall(r"<w:sectPr\b.*?</w:sectPr>", tdoc, flags=re.S)[-1]
    # relationships of the template that the sectPr uses
    rel_ids = set(re.findall(r'r:id="(rId\d+)"', sect))
    rels = [r for r in re.findall(r"<Relationship [^>]*/>", trels)
            if re.search(r'Id="(%s)"' % "|".join(rel_ids), r)]
    # prefix the ids so they cannot clash with pandoc's
    sect = re.sub(r'r:id="rId(\d+)"', r'r:id="rIdTpl\1"', sect)
    rels = [re.sub(r'Id="rId(\d+)"', r'Id="rIdTpl\1"', r) for r in rels]
    override = "".join(
        f'<Override PartName="/{n}" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.'
        f'{PART_RE.match(n).group(1)}+xml"/>' for n in parts)

    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        names = set(zin.namelist())
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                xml = re.sub(r"<w:sectPr\b.*?</w:sectPr>", lambda m: sect, xml, flags=re.S)
                if "<w:sectPr" not in xml:
                    xml = xml.replace("</w:body>", sect + "</w:body>")
                # the template sectPr must declare the r: namespace
                if 'xmlns:r=' not in xml.split(">", 2)[1] + xml[:1000]:
                    xml = xml.replace("<w:document ", f'<w:document xmlns:r="{REL_NS}" ', 1)
                data = xml.encode("utf-8")
            elif item.filename == "word/_rels/document.xml.rels":
                xml = re.sub(r'<Relationship [^>]*Id="rIdTpl\d+"[^>]*/>', "", data.decode("utf-8"))
                data = xml.replace("</Relationships>", "".join(rels) + "</Relationships>").encode("utf-8")
            elif item.filename == "[Content_Types].xml":
                xml = data.decode("utf-8")
                for n in parts:
                    xml = re.sub(rf'<Override PartName="/{re.escape(n)}"[^>]*/>', "", xml)
                data = xml.replace("</Types>", override + "</Types>").encode("utf-8")
            elif PART_RE.match(item.filename) or item.filename in part_rels:
                continue
            zout.writestr(item, data)
        for n, b in {**parts, **part_rels}.items():
            zout.writestr(n, b)
        for n, b in media.items():
            if n not in names:
                zout.writestr(n, b)
    shutil.move(tmp, path)
    print(f"applied template layout to {path}")


if __name__ == "__main__":
    main()
