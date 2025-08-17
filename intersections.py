# -*- coding: utf-8 -*-
"""
Intersections parcelle (EPSG:4326) ↔ couches PostGIS (Supabase, pooler eu-west-3)
-------------------------------------------------------------------------------
- Aucune requête WFS côté parcelle : coords lon/lat (EPSG:4326) en entrée
- Découverte auto des couches via geometry_columns + geography_columns
- Reprojection de la parcelle vers le SRID de chaque couche avant ST_Intersects
- Détection heuristique des colonnes "de valeur" (libellé, code, zone, type, …)
- Export JSON exhaustif :
    * count des entités intersectantes
    * values (distinctes) par attribut pertinent
    * coverage : % de parcelle couvert par chaque valeur d’attribut (UA 74%, UE 26%, …)
- Timeouts locaux pour éviter les requêtes trop longues

Dépendances : sqlalchemy, psycopg2-binary, python-dotenv
"""

import os
import re
import json
import logging
import socket
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("intersections_export_values")

# ======================= Connexion (pooler eu-west-3) ======================= #
def connect_direct() -> Engine:
    # ⚠️ en prod, passe par .env / secrets
    project_ref = "odlkagfeqkbrruajlcxm"
    region = "eu-west-3"
    host = f"aws-0-{region}.pooler.supabase.com"
    port = 5432
    user = f"postgres.{project_ref}"
    password = "Kerelia123+"  # ⚠️ à sécuriser (env var)
    db = "postgres"

    dsn = f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{db}?sslmode=require"
    eng = create_engine(
        dsn,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"connect_timeout": 20},
    )

    # test rapide
    with eng.begin() as con:
        con.execute(text("select 1"))
    log.info(f"✅ Connexion OK via pooler → {host}:{port} (DB={db})")
    return eng

# ======================= Utils identifiants & encodage ====================== #
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")

def _safe_ident(s: str) -> bool:
    return bool(IDENT_RE.fullmatch(s or ""))

def _q_ident(ident: str) -> str:
    return ".".join(f'"{p}"' for p in ident.split("."))

def _fix_utf8_mojibake(x):
    # Corrige "arrÃªtÃ©" -> "arrêté" quand la source a été mal encodée
    if isinstance(x, str):
        try:
            return x.encode('latin1').decode('utf-8')
        except Exception:
            return x
    if isinstance(x, list):
        return [_fix_utf8_mojibake(v) for v in x]
    if isinstance(x, dict):
        return {k: _fix_utf8_mojibake(v) for k, v in x.items()}
    return x

# ============================ Découverte des couches ========================= #
SQL_LIST_LAYERS = """
SELECT
  f_table_schema      AS "schema",
  f_table_name        AS "table",
  f_geometry_column   AS geom_col,
  srid,
  type,
  'geometry'::text    AS gkind
FROM geometry_columns
UNION ALL
SELECT
  f_table_schema      AS "schema",
  f_table_name        AS "table",
  f_geography_column  AS geom_col,
  srid,
  type,
  'geography'::text   AS gkind
FROM geography_columns
ORDER BY "schema", "table", geom_col;
"""

def list_layers(eng: Engine) -> List[Dict[str, Any]]:
    with eng.begin() as con:
        rows = con.execute(text(SQL_LIST_LAYERS)).mappings().all()
    layers = []
    for r in rows:
        sch, tbl, geom, srid, gkind = r["schema"], r["table"], r["geom_col"], int(r["srid"] or 0), r["gkind"]
        if not (sch and tbl and geom) or srid == 0:
            continue
        if tbl == "ingest_log":
            continue
        if not all(_safe_ident(x) for x in [sch, tbl, geom]):
            continue
        layers.append({"schema": sch, "table": tbl, "geom_col": geom, "srid": srid, "gkind": gkind})
    return layers

# ========================= Choix des colonnes "valeur" ====================== #
CANDIDATE_NAME_PATTERNS = [
    r"libell?e", r"label", r"nom", r"name", r"intitul[eé]", r"designation", r"titre", r"title",
    r"zone", r"secteur", r"categorie", r"cat(egorie)?", r"type", r"nature", r"statut", r"classe?",
    r"code(_?(insee|zone|sup|plu))?", r"id(_?(zone|sup|plu))?", r"regime", r"niveau",
    r"servitude", r"risq?ue", r"enjeu", r"znieff", r"ppri", r"ppr(if|l|sm|mt)?", r"plu",
    r"zps|sic|natura", r"epci", r"departement|commune"
]
CANDIDATE_NAME_RE = re.compile("|".join(CANDIDATE_NAME_PATTERNS), re.IGNORECASE)
TEXT_TYPES = {"text", "varchar", "character varying", "char", "bpchar", "citext"}

def get_non_geom_columns(eng: Engine, schema: str, table: str, geom_cols_known: List[str]) -> List[Tuple[str, str]]:
    q_cols = """
    SELECT column_name, data_type, udt_name
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table
    ORDER BY ordinal_position;
    """
    q_geom_extra = """
    SELECT f_geometry_column AS g FROM geometry_columns
    WHERE f_table_schema = :schema AND f_table_name = :table
    UNION ALL
    SELECT f_geography_column AS g FROM geography_columns
    WHERE f_table_schema = :schema AND f_table_name = :table;
    """
    with eng.begin() as con:
        cols = con.execute(text(q_cols), {"schema": schema, "table": table}).mappings().all()
        gextra = con.execute(text(q_geom_extra), {"schema": schema, "table": table}).mappings().all()
    geom_set = {*(geom_cols_known or []), *(g["g"] for g in gextra if g["g"])}

    out = []
    for c in cols:
        name = c["column_name"]
        dt = (c["data_type"] or "").lower()
        udt = (c["udt_name"] or "").lower()
        dtype = udt if udt in TEXT_TYPES else dt
        if name in geom_set:
            continue
        out.append((name, dtype))
    return out

def choose_candidate_columns(columns: List[Tuple[str, str]], max_fallback: int = 3) -> List[str]:
    text_cols = [(n, t) for (n, t) in columns if t in TEXT_TYPES]
    by_name = [n for (n, _) in text_cols if CANDIDATE_NAME_RE.search(n)]
    if by_name:
        return by_name
    return [n for (n, _) in text_cols[:max_fallback]]

# =================== Intersections + valeurs + couverture % ================= #
def count_intersections(eng: Engine, schema: str, table: str, geom: str, gkind: str,
                        parcel_wkt_4326: str, l_srid: int) -> int:
    geom_expr = f't.{_q_ident(geom)}::geometry' if gkind == "geography" else f't.{_q_ident(geom)}'
    q = f"""
    SET LOCAL statement_timeout = '30s';
    WITH parcelle AS (
      SELECT ST_Transform(ST_GeomFromText(:wkt, 4326), :lsrid) AS g
    )
    SELECT COUNT(*)::bigint
    FROM {_q_ident(schema)}.{_q_ident(table)} t, parcelle p
    WHERE {geom_expr} IS NOT NULL
      AND ST_Intersects({geom_expr}, p.g);
    """
    with eng.begin() as con:
        return int(con.execute(text(q), {"wkt": parcel_wkt_4326, "lsrid": l_srid}).scalar() or 0)

def distinct_values_for_column(eng: Engine, schema: str, table: str, geom: str, gkind: str,
                               parcel_wkt_4326: str, l_srid: int, col: str,
                               max_values: int = 50) -> List[Any]:
    if not all(_safe_ident(x) for x in [schema, table, geom, col]):
        return []
    geom_expr = f't.{_q_ident(geom)}::geometry' if gkind == "geography" else f't.{_q_ident(geom)}'
    q = f"""
    SET LOCAL statement_timeout = '30s';
    WITH parcelle AS (
      SELECT ST_Transform(ST_GeomFromText(:wkt, 4326), :lsrid) AS g
    )
    SELECT DISTINCT t.{_q_ident(col)} AS v
    FROM {_q_ident(schema)}.{_q_ident(table)} t, parcelle p
    WHERE {geom_expr} IS NOT NULL
      AND ST_Intersects({geom_expr}, p.g)
      AND t.{_q_ident(col)} IS NOT NULL
    LIMIT :lim;
    """
    with eng.begin() as con:
        rows = con.execute(text(q), {"wkt": parcel_wkt_4326, "lsrid": l_srid, "lim": int(max_values)}).all()
    return [r[0] for r in rows]

def values_map_for_layer(eng: Engine, layer: Dict[str, Any], parcel_wkt_4326: str,
                         per_col_value_limit: int = 50) -> Dict[str, List[Any]]:
    schema, table, geom, gkind, srid = layer["schema"], layer["table"], layer["geom_col"], layer["gkind"], layer["srid"]
    cols = get_non_geom_columns(eng, schema, table, [geom])
    cand = choose_candidate_columns(cols)
    if not cand:
        return {}
    out: Dict[str, List[Any]] = {}
    for col in cand:
        vals = distinct_values_for_column(eng, schema, table, geom, gkind, parcel_wkt_4326, srid, col,
                                          max_values=per_col_value_limit)
        if vals:
            out[col] = vals
    return out

def coverage_by_attribute(eng: Engine, layer, parcel_wkt_4326: str, attr: str, min_pct: float = 0.0):
    schema, table, geom, gkind, l_srid = layer["schema"], layer["table"], layer["geom_col"], layer["gkind"], layer["srid"]
    if not all(_safe_ident(x) for x in [schema, table, geom, attr]):
        return []
    geom_expr = f't.{_q_ident(geom)}::geometry' if gkind == "geography" else f't.{_q_ident(geom)}'

    q = f"""
    SET LOCAL statement_timeout = '30s';
    WITH
      p AS (
        SELECT ST_Transform(ST_GeomFromText(:wkt, 4326), 2154) AS g_l93,
               ST_Area(ST_Transform(ST_GeomFromText(:wkt, 4326), 2154)) AS a_parcel
      ),
      f AS (  -- features candidates (filtrées au SRID couche)
        SELECT
          ST_Transform({geom_expr}, 2154) AS g_l93,
          t.{_q_ident(attr)} AS v
        FROM {_q_ident(schema)}.{_q_ident(table)} t, p
        WHERE {geom_expr} IS NOT NULL
          AND t.{_q_ident(attr)} IS NOT NULL
          AND ST_Intersects({geom_expr}, ST_Transform(p.g_l93, :lsrid))
      ),
      i AS (  -- intersection stricte en L93
        SELECT v, ST_Intersection(f.g_l93, p.g_l93) AS gi
        FROM f, p
        WHERE f.g_l93 && p.g_l93
      ),
      d AS (  -- AGRÉGATION correcte : collect → unaryunion
        SELECT v, ST_UnaryUnion(ST_Collect(gi)) AS gu
        FROM i
        WHERE gi IS NOT NULL AND NOT ST_IsEmpty(gi)
        GROUP BY v
      ),
      s AS (  -- surfaces par valeur
        SELECT v, ST_Area(gu) AS area_m2
        FROM d
        WHERE gu IS NOT NULL AND NOT ST_IsEmpty(gu)
      ),
      pa AS (SELECT a_parcel FROM p)
    SELECT v::text AS v, area_m2,
           CASE WHEN pa.a_parcel > 0 THEN area_m2 / pa.a_parcel ELSE 0 END AS pct
    FROM s, pa
    WHERE area_m2 IS NOT NULL
    ORDER BY pct DESC NULLS LAST;
    """
    with eng.begin() as con:
        rows = con.execute(text(q), {"wkt": parcel_wkt_4326, "lsrid": l_srid}).mappings().all()

    out = []
    for r in rows:
        if r["v"] is None:
            continue
        out.append({
            "value": _fix_utf8_mojibake(r["v"]),
            "area_m2": float(r["area_m2"] or 0.0),
            "parcel_pct": float(r["pct"] or 0.0),
        })
    # filtre optionnel
    if min_pct > 0:
        out = [x for x in out if x["parcel_pct"] * 100 >= min_pct]
    return out

# ------------------------------- Filet filtre verification SRID ------------------------------- #

def effective_srid_for_layer(eng: Engine, layer: Dict[str, Any]) -> Optional[int]:
    """
    Retourne le SRID majoritaire observé dans les données (via ST_SRID),
    ou None si indéterminable. Marche aussi pour geography (cast -> geometry).
    """
    schema, table, geom, gkind = layer["schema"], layer["table"], layer["geom_col"], layer["gkind"]
    if not all(_safe_ident(x) for x in [schema, table, geom]):
        return None
    geom_expr = f't.{_q_ident(geom)}::geometry' if gkind == "geography" else f't.{_q_ident(geom)}'
    q = f"""
    SELECT srid, cnt FROM (
      SELECT ST_SRID({geom_expr})::int AS srid, count(*) AS cnt
      FROM {_q_ident(schema)}.{_q_ident(table)} t
      WHERE {geom_expr} IS NOT NULL
      GROUP BY 1
    ) s
    WHERE srid IS NOT NULL AND srid > 0
    ORDER BY cnt DESC
    LIMIT 1;
    """
    with eng.begin() as con:
        r = con.execute(text(q)).mappings().first()
    return int(r["srid"]) if r else None



# ================================ Orchestration ============================= #
def run(parcel_coords_lonlat_4326: List[List[float]],
        per_layer_attr_values_limit: int = 50,
        save_json: Optional[str] = None,
        schema_whitelist: Optional[List[str]] = None) -> Dict[str, Any]:

    eng = connect_direct()

    layers = list_layers(eng)
    if schema_whitelist:
        layers = [l for l in layers if l["schema"] in schema_whitelist]
    log.info(f"🗺️ Couches candidates: {len(layers)}")

    # WKT POLYGON (lon lat). Ferme la boucle si besoin.
    coords = parcel_coords_lonlat_4326
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    ring = ", ".join(f"{lon} {lat}" for lon, lat in coords)
    parcel_wkt = f"POLYGON(({ring}))"

    results = []
    hit_layers = 0

    for i, lyr in enumerate(layers, 1):
        name = f"{lyr['schema']}.{lyr['table']}"
        try:
            # Vérifie le SRID observé dans les données
            lyr_srid = lyr["srid"]
            obs = effective_srid_for_layer(eng, lyr)
            if obs and obs != lyr_srid:
                log.warning(f"🔎 {name}: SRID déclaré {lyr_srid} ≠ observé {obs} → utilisation {obs}")
                lyr_srid = obs

            # Comptage d'intersections
            n = count_intersections(
                eng, lyr["schema"], lyr["table"], lyr["geom_col"],
                lyr["gkind"], parcel_wkt, lyr_srid
            )

            if n > 0:
                hit_layers += 1

                # Valeurs attributaires distinctes
                vals_map = values_map_for_layer(
                    eng, {**lyr, "srid": lyr_srid}, parcel_wkt,
                    per_col_value_limit=per_layer_attr_values_limit
                )
                vals_map = {k: _fix_utf8_mojibake(v) for k, v in vals_map.items()}

                # Couverture % par valeur
                coverage = {}
                for col in vals_map.keys():
                    cov = coverage_by_attribute(
                        eng, {**lyr, "srid": lyr_srid}, parcel_wkt, col, min_pct=0.1
                    )
                    if cov:
                        coverage[col] = cov

                results.append({
                    "schema": lyr["schema"],
                    "table": lyr["table"],
                    "srid": lyr_srid,  # on garde le SRID corrigé
                    "geom_col": lyr["geom_col"],
                    "count": n,
                    "value_attributes": vals_map,  # {col -> [values]}
                    "coverage": coverage           # {col -> [{value, area_m2, parcel_pct}, ...]}
                })
                log.info(f"✅ {i}/{len(layers)} {name}: {n} intersect(s) | attrs: {', '.join(vals_map.keys()) or '—'}")
            else:
                log.info(f"—  {i}/{len(layers)} {name}: 0")

        except Exception as e:
            log.error(f"❌ {i}/{len(layers)} {name}: {e}")

    out = {
        "parcel": {
            "srid": 4326,
            "coords_lonlat": parcel_coords_lonlat_4326,
            "wkt": parcel_wkt
        },
        "layers_with_hits": hit_layers,
        "results": results
    }

    if save_json:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        log.info(f"💾 JSON écrit: {save_json}")

    log.info("—" * 60)
    log.info(f"✅ Terminé. Couches intersectantes: {hit_layers}")

    return out
