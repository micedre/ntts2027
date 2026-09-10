"""Post-process ``abstract.docx`` so its tables render the same everywhere.

Pandoc writes tables with a column grid but empty cell properties
(``<w:tcPr/>``) and a percentage table width. Word usually copes, other
viewers (LibreOffice, OnlyOffice, Google Docs, previewers) often do not and
collapse or overflow the columns. This script rewrites every pandoc table
(style ``Table``) with an absolute width equal to the grid, explicit cell
widths taken from the grid, and no indent. Idempotent.

Usage: python scripts/fix_docx_tables.py [abstract.docx]
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def fix_table(tbl: str) -> str:
    grid = [int(w) for w in re.findall(r'<w:gridCol w:w="(\d+)"', tbl)]
    if not grid:
        return tbl
    total = sum(grid)
    # absolute table width, no indent, fixed layout
    tbl = re.sub(r'<w:tblW [^>]*/>', f'<w:tblW w:type="dxa" w:w="{total}" />', tbl, count=1)
    if "<w:tblInd" not in tbl:
        tbl = tbl.replace("</w:tblPr>", '<w:tblInd w:w="0" w:type="dxa" /></w:tblPr>', 1)
    # explicit width on every cell, column by column
    rows = re.findall(r'<w:tr>.*?</w:tr>|<w:tr .*?</w:tr>', tbl, flags=re.S)
    for row in rows:
        cells = re.findall(r'<w:tc>.*?</w:tc>', row, flags=re.S)
        if len(cells) != len(grid):
            continue
        new_row = row
        for cell, width in zip(cells, grid):
            tcw = f'<w:tcW w:w="{width}" w:type="dxa" />'
            if "<w:tcW" in cell:
                new_cell = re.sub(r'<w:tcW [^>]*/>', tcw, cell, count=1)
            elif "<w:tcPr />" in cell or "<w:tcPr/>" in cell:
                new_cell = re.sub(r'<w:tcPr ?/>', f"<w:tcPr>{tcw}</w:tcPr>", cell, count=1)
            elif "<w:tcPr>" in cell:
                new_cell = cell.replace("<w:tcPr>", f"<w:tcPr>{tcw}", 1)
            else:
                new_cell = cell.replace("<w:tc>", f"<w:tc><w:tcPr>{tcw}</w:tcPr>", 1)
            new_row = new_row.replace(cell, new_cell, 1)
        tbl = tbl.replace(row, new_row, 1)
    return tbl


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "abstract.docx"
    tmp = path.with_suffix(".tmp.docx")
    n = 0
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")

                def repl(m: re.Match) -> str:
                    nonlocal n
                    t = m.group(0)
                    if '<w:tblStyle w:val="Table" />' not in t and '<w:tblStyle w:val="Table"/>' not in t:
                        return t
                    n += 1
                    return fix_table(t)

                # innermost tables only (no nested <w:tbl> inside the match)
                xml = re.sub(r"<w:tbl>(?:(?!<w:tbl>).)*?</w:tbl>", repl, xml, flags=re.S)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)
    print(f"fixed {n} table(s) in {path}")


if __name__ == "__main__":
    main()
