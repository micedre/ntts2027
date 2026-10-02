"""Distribution of ECOICOP level-4 codes in the training data (read-only).

Sources (all on S3, the first two encrypted with the project's parquet key):
  * raw scanner extraction  data/ddc_raw/ddc_raw_20260904.parquet          (output of extract-ddc)
  * stage-1 corpus          data/ddc_raw/ddc_train_20260930-full.parquet   (DDC + synthetic)
  * annotated survey set    data/workflow_runs/2026-09-30/train-ttc-57c27/build-datasets/annotations_full.parquet

The parquet key is read from the environment variable ``DDC_ENCRYPTION_KEY``
(same key as the codif-ttc scripts). It is never printed nor written anywhere.

Outputs: results/code_distribution.csv  (dataset, l4, n — one row per code)
         results/code_distribution_summary.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent.parent
OUT_CSV = HERE / "results" / "code_distribution.csv"
OUT_JSON = HERE / "results" / "code_distribution_summary.json"

BUCKET = "s3://projet-budget-famille/data/"
RAW_DDC = [BUCKET + "ddc_raw/ddc_raw_20260904.parquet"]
CORPUS = BUCKET + "ddc_raw/ddc_train_20260930-full.parquet"
ANNOTATED = BUCKET + "workflow_runs/2026-09-30/train-ttc-57c27/build-datasets/annotations_full.parquet"
PRUNED_NOMENCLATURE = BUCKET + "coicop-2018_envoi_rmes_20251022_prunned_lvl4.parquet"

L4 = "array_to_string(list_slice(string_split({col}, '.'), 1, 4), '.')"


def connect() -> duckdb.DuckDBPyConnection:
    key = os.environ.get("DDC_ENCRYPTION_KEY")
    if not key:
        sys.exit("DDC_ENCRYPTION_KEY is not set (parquet key of the DDC files); nothing done.")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        CREATE SECRET s3_secret (
            TYPE S3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
            SESSION_TOKEN '{os.environ.get("AWS_SESSION_TOKEN", "")}',
            REGION 'us-east-1', URL_STYLE 'path', SCOPE 's3://'
        );
    """)
    if not re.fullmatch(r"[0-9a-fA-F]{32,64}", key):
        sys.exit("DDC_ENCRYPTION_KEY does not look like a hex parquet key; nothing done.")
    con.execute(f"PRAGMA add_parquet_key('k', '{key}');")  # PRAGMA cannot take bound parameters
    return con


def enc(path: str) -> str:
    return f"read_parquet('{path}', encryption_config={{footer_key: 'k'}})"


def main() -> None:
    con = connect()
    raw_union = " UNION ALL ".join(
        f"SELECT description_ean, {L4.format(col='coicop_code')} AS l4 FROM {enc(p)}" for p in RAW_DDC
    )
    con.execute(f"CREATE TABLE raw AS {raw_union}")
    con.execute(f"CREATE TABLE corpus AS SELECT source, code8 AS l4 FROM {enc(CORPUS)}")
    con.execute(f"CREATE TABLE annot AS SELECT {L4.format(col='code')} AS l4 FROM read_parquet('{ANNOTATED}')")
    con.execute(f"""
        CREATE TABLE leaves AS
        SELECT code AS l4 FROM read_parquet('{PRUNED_NOMENCLATURE}')
        WHERE code NOT IN (SELECT parent FROM read_parquet('{PRUNED_NOMENCLATURE}') WHERE parent IS NOT NULL)
    """)

    # one row per (dataset, code)
    con.execute("""
        CREATE TABLE dist AS
        SELECT 'raw_ddc' AS dataset, l4, count(*) AS n FROM raw GROUP BY l4
        UNION ALL SELECT 'corpus_ddc', l4, count(*) FROM corpus WHERE source = 'ddc' GROUP BY l4
        UNION ALL SELECT 'corpus_synthetic', l4, count(*) FROM corpus WHERE source = 'synthetic' GROUP BY l4
        UNION ALL SELECT 'corpus_all', l4, count(*) FROM corpus GROUP BY l4
        UNION ALL SELECT 'annotated', l4, count(*) FROM annot GROUP BY l4
    """)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM dist ORDER BY dataset, n DESC) TO '{OUT_CSV}' (HEADER)")

    def summary(dataset: str) -> dict:
        row = con.execute(f"""
            WITH r AS (SELECT *, row_number() OVER (ORDER BY n DESC) rn, sum(n) OVER () tot
                       FROM dist WHERE dataset = ?)
            SELECT count(*), sum(n), median(n), max(n),
                   sum(n) FILTER (WHERE rn <= 10) / max(tot),
                   sum(n) FILTER (WHERE l4 LIKE '01.%') / max(tot),
                   count(*) FILTER (WHERE n < 100),
                   count(*) FILTER (WHERE n < 1000),
                   count(*) FILTER (WHERE l4 IN (SELECT l4 FROM leaves)),
                   count(*) FILTER (WHERE l4 LIKE '98%' OR l4 LIKE '99%')
            FROM r
        """, [dataset]).fetchone()
        keys = ["n_codes", "rows", "median_per_code", "max_per_code", "top10_share", "food_share",
                "codes_lt_100", "codes_lt_1000", "codes_among_pruned_leaves", "technical_codes"]
        return {k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in zip(keys, row)}

    out = {d: summary(d) for d in ["raw_ddc", "corpus_ddc", "corpus_synthetic", "corpus_all", "annotated"]}
    out["raw_ddc"]["distinct_descriptions"] = int(con.execute("SELECT count(DISTINCT description_ean) FROM raw").fetchone()[0])
    out["corpus_coverage"] = {
        k: {"codes": int(c), "rows": int(r)}
        for k, c, r in con.execute("""
            SELECT CASE WHEN d AND s THEN 'both' WHEN d THEN 'ddc_only' ELSE 'synthetic_only' END, count(*), sum(n)
            FROM (SELECT l4, bool_or(source='ddc') d, bool_or(source='synthetic') s, count(*) n FROM corpus GROUP BY l4)
            GROUP BY 1
        """).fetchall()
    }
    out["pruned_leaves"] = int(con.execute("SELECT count(*) FROM leaves").fetchone()[0])
    out["pruned_leaves_without_scanner_data"] = int(
        con.execute("SELECT count(*) FROM leaves WHERE l4 NOT IN (SELECT l4 FROM raw)").fetchone()[0])
    out["annotated_outside_corpus_label_space"] = dict(zip(
        ["codes", "rows"],
        [int(x) for x in con.execute("SELECT count(DISTINCT l4), count(*) FROM annot WHERE l4 NOT IN (SELECT l4 FROM corpus)").fetchone()],
    ))
    out["division_share_pct"] = {
        d: {"raw_ddc": r, "corpus_all": c, "annotated": a}
        for d, r, c, a in con.execute("""
            SELECT d, round(sum(r)*100/(SELECT count(*) FROM raw),1), round(sum(c)*100/(SELECT count(*) FROM corpus),1),
                   round(sum(a)*100/(SELECT count(*) FROM annot),1)
            FROM (SELECT left(l4,2) d, 1 r, 0 c, 0 a FROM raw
                  UNION ALL SELECT left(l4,2), 0, 1, 0 FROM corpus
                  UNION ALL SELECT left(l4,2), 0, 0, 1 FROM annot)
            GROUP BY 1 ORDER BY 1
        """).fetchall()
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "division_share_pct"}, indent=1))
    print(f"wrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
