# -*- coding: utf-8 -*-
"""
App Streamlit — Analyse d'intersections urbanistiques
- INSEE via CSV local (v_commune_2025.csv)
- Géométrie parcelle via WFS IGN (EPSG:4326), sans GeoPandas
- Résultats persistés avec st.session_state (pas de disparition après le run)
- Classement des couches par origine (WFS / SHP / Autres) via layer_registry
"""

import json
from typing import List, Dict, Any, Optional
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

# --- module calcul existant ---
from intersections import run as run_intersections
from intersections import list_layers, connect_direct

st.set_page_config(page_title="Intersections urbanistiques", layout="wide")

# ======================= Constantes & fallback =======================
DEFAULT_PARCEL_COORDS_LONLAT: List[List[float]] = [
    [-0.49049403, 44.78599768], [-0.49049645, 44.78601013], [-0.49046629, 44.78601153],
    [-0.49046708, 44.78602620], [-0.49043873, 44.78602826], [-0.49043636, 44.78601257],
    [-0.49042420, 44.78601232], [-0.49041418, 44.78594874], [-0.49040172, 44.78587046],
    [-0.49039878, 44.78585153], [-0.49044849, 44.78584916], [-0.49045140, 44.78585943],
    [-0.49045161, 44.78586276], [-0.49047401, 44.78587710], [-0.49049403, 44.78599768]
]
SCHEMA_WHITELIST_DEFAULT = ["public"]

IGN_WFS = "https://data.geopf.fr/wfs/ows"
LAYER_PARCELLE = "CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle"

# ======================= Helpers géométrie & UI =======================
def lonlat_to_latlon(coords_lonlat: List[List[float]]) -> List[List[float]]:
    return [[lat, lon] for lon, lat in coords_lonlat]

def centroid_lonlat(coords_lonlat: List[List[float]]):
    arr = np.array(coords_lonlat)
    lon_c = float(arr[:, 0].mean()); lat_c = float(arr[:, 1].mean())
    return lat_c, lon_c  # (lat, lon) pour folium

def ensure_ring_closed(coords_lonlat: List[List[float]]) -> List[List[float]]:
    if not coords_lonlat:
        return coords_lonlat
    if coords_lonlat[0] != coords_lonlat[-1]:
        return coords_lonlat + [coords_lonlat[0]]
    return coords_lonlat

def geojson_polygon_to_lonlat_ring(geom: Dict[str, Any]) -> Optional[List[List[float]]]:
    if not geom or "type" not in geom or "coordinates" not in geom:
        return None
    t = geom["type"]; coords = geom["coordinates"]
    if t == "Polygon":
        if not coords or not coords[0]:
            return None
        ring = [[float(x), float(y)] for (x, y) in coords[0]]
        return ensure_ring_closed(ring)
    if t == "MultiPolygon":
        rings = []
        for poly in coords:
            if poly and poly[0]:
                ring = [[float(x), float(y)] for (x, y) in poly[0]]
                rings.append(ensure_ring_closed(ring))
        return max(rings, key=lambda r: len(r)) if rings else None
    return None

# ======================= INSEE via CSV local =======================
@st.cache_data(show_spinner=False)
def load_communes_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, sep=",", dtype=str)
    except Exception:
        return None

def normalize_string(text: str) -> str:
    if pd.isna(text):
        return ""
    s = text.lower()
    s = (s.replace('é','e').replace('è','e').replace('ê','e')
           .replace('à','a').replace('â','a')
           .replace('ô','o')
           .replace('ù','u').replace('û','u')
           .replace('ç','c')
           .replace('-', ' ').replace("'", ' '))
    return ' '.join(s.split()).strip()

def get_insee_from_csv(df_communes: pd.DataFrame, commune_name: str, department_code: Optional[str]) -> Optional[str]:
    # Colonnes présentes: TYPECOM, COM, REG, DEP, CTCD, ARR, TNCC, NCC, NCCENR, LIBELLE, CAN, COMPARENT
    if df_communes is None or df_communes.empty or "COM" not in df_communes.columns:
        return None
    df = df_communes.copy()
    have_label = "LIBELLE" in df.columns
    have_nccenr = "NCCENR" in df.columns
    if have_label: df["LIBELLE_NORM"] = df["LIBELLE"].apply(normalize_string)
    if have_nccenr: df["NCCENR_NORM"] = df["NCCENR"].apply(normalize_string)

    name_norm = normalize_string(commune_name)
    mask = False
    if have_label: mask = (df["LIBELLE_NORM"] == name_norm)
    if have_nccenr: mask = mask | (df["NCCENR_NORM"] == name_norm)
    df = df[mask]
    if df.empty: return None

    if department_code and "DEP" in df.columns:
        dep = str(department_code).upper()
        if dep.isdigit() and len(dep) == 1: dep = dep.zfill(2)  # ex: "3" -> "03"
        df = df[df["DEP"].str.upper() == dep]
        if df.empty: return None

    if len(df) != 1: return None
    return df.iloc[0]["COM"]  # code INSEE

# ======================= WFS Parcellaire (sans GeoPandas) =======================
def build_wfs_url(params: dict) -> str:
    from urllib.parse import urlencode
    return f"{IGN_WFS}?{urlencode(params)}"

def locate_parcel_no_gpd(insee_code: str, parcels_str: str) -> dict:
    """
    Résout une ou plusieurs parcelles par (INSEE, SECTION, NUMÉRO) et renvoie un GeoJSON.
    - NUMÉRO padding 4 (ex: 12 -> '0012')
    - Géométrie en EPSG:4326 (srsName)
    """
    features = []
    for raw in parcels_str.split(","):
        parts = raw.strip().split()
        if len(parts) != 2:
            continue
        section, numero = parts
        numero_padded = str(numero).zfill(4)

        cql = (
            f"code_insee='{insee_code}' AND "
            f"section='{section}' AND "
            f"numero='{numero_padded}'"
        )
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": LAYER_PARCELLE,
            "outputFormat": "application/json",
            "count": 1,
            "cql_filter": cql,
            "srsName": "EPSG:4326",
        }
        try:
            r = requests.get(build_wfs_url(params), timeout=30)
            r.raise_for_status()
            data = r.json()
            feats = data.get("features") or []
            if feats:
                features.append(feats[0])
        except Exception:
            pass
    return {"type": "FeatureCollection", "features": features}

# ======================= UI (sidebar) =======================
st.sidebar.title("Paramètres")

schema_whitelist = st.sidebar.multiselect(
    "Limiter aux schémas",
    options=SCHEMA_WHITELIST_DEFAULT,
    default=SCHEMA_WHITELIST_DEFAULT,
)

limit_values = st.sidebar.slider(
    "Nb max de valeurs distinctes par attribut",
    min_value=10, max_value=200, value=50, step=10
)

with st.sidebar.expander("Recherche parcelle", expanded=True):
    mode_coords = st.toggle("Utiliser les coordonnées en dur (WGS84)", value=False)
    commune_name = st.text_input("Commune (nom exact)", value="Latresne", disabled=mode_coords)
    department_code = st.text_input("Code département (ex: 33, 2A, 2B)", value="33", disabled=mode_coords)
    parcel_str = st.text_input("Parcelle(s) (ex: 'AB 0123' ou 'AB 0123, AC 0042')", value="AL 27", disabled=mode_coords)
    csv_path = st.text_input("Chemin CSV communes (v_commune_2025.csv)", value="v_commune_2025.csv", disabled=mode_coords)

colA, colB = st.sidebar.columns([1, 1])
run_btn = colA.button("🔎 Lancer l'analyse", type="primary")
reset_btn = colB.button("♻️ Réinitialiser")

# ======================= Session state =======================
if reset_btn:
    for k in ("parcel_coords", "parcel_label", "results", "schema_filter"):
        st.session_state.pop(k, None)
    st.rerun()

@st.cache_data(show_spinner=False)
def load_all_layers(schema_filter: List[str]) -> List[Dict[str, Any]]:
    eng = connect_direct()
    layers_all = list_layers(eng)
    if schema_filter:
        layers_all = [l for l in layers_all if l["schema"] in schema_filter]
    for l in layers_all:
        l["full_name"] = f"{l['schema']}.{l['table']}"
    return layers_all

def summarize_coverage(result_item: Dict[str, Any], top_n: int = 3) -> str:
    cov = result_item.get("coverage") or {}
    parts = []
    for attr, rows in cov.items():
        rows_sorted = sorted(rows, key=lambda r: r.get("parcel_pct", 0), reverse=True)
        segs = []
        for r in rows_sorted[:top_n]:
            pct = 100.0 * float(r.get("parcel_pct") or 0.0)
            segs.append(f"{str(r.get('value'))} {pct:.1f}%")
        if segs: parts.append(f"{attr} → " + ", ".join(segs))
    return " ; ".join(parts)

def preview_attrs(result_item: Dict[str, Any], max_cols: int = 3) -> str:
    attrs = list((result_item.get("value_attributes") or {}).keys())
    if not attrs: return "—"
    return ", ".join(attrs[:max_cols]) + (f" (+{len(attrs)-max_cols})" if len(attrs) > max_cols else "")

# === Nouveaux helpers d'affichage d'origine ===
def pretty_origin(o: Optional[str]) -> str:
    o = (o or "").lower()
    if o == "wfs":
        return "🌐 WFS"
    if o in ("shp", "shape", "shapefile", "file"):
        return "🗂️ SHP"
    return "❓ Autre"

def render_outputs():
    """Affiche la carte + résultats depuis st.session_state."""
    parcel_coords = st.session_state.get("parcel_coords")
    label = st.session_state.get("parcel_label", "Parcelle")
    results = st.session_state.get("results")
    if not parcel_coords or not results:
        st.info("Renseigne les champs puis clique **Lancer l’analyse**.")
        return

    # Carte
    st.markdown("## Parcelle")
    lat_c, lon_c = centroid_lonlat(parcel_coords)
    m = folium.Map(location=[lat_c, lon_c], zoom_start=19, tiles="OpenStreetMap")
    folium.Polygon(
        locations=lonlat_to_latlon(parcel_coords),
        tooltip=label, popup=label,
        weight=3, color="#2c7fb8", fill=True, fill_opacity=0.25,
    ).add_to(m)
    st_folium(m, height=420, use_container_width=True)

    # Résumé
    layers_all = load_all_layers(st.session_state.get("schema_filter") or SCHEMA_WHITELIST_DEFAULT)
    hit_by_fullname = {f"{r['schema']}.{r['table']}": r for r in results.get("results", [])}

    st.markdown("## Résultats de superposition")
    st.caption(
        f"Parcelle SRID **{results['parcel']['srid']}** — "
        f"Couches intersectantes: **{results.get('layers_with_hits', 0)}** / **{len(layers_all)}**"
    )

    # Tableau principal + onglets par origine
    rows = []
    for lyr in layers_all:
        name = lyr["full_name"]
        r = hit_by_fullname.get(name)
        rows.append({
            "couche": name,
            "source": pretty_origin(lyr.get("origin")),
            "intersecte": "✅" if r else "❌",        # affichage lisible
            "origin_raw": (lyr.get("origin") or "").lower(),   # pour filtrage
            "source_path": lyr.get("source_path") or "—",
            "geom_col": lyr["geom_col"],
            "srid": lyr["srid"],
            "count": (r or {}).get("count", 0),
            "attributs": preview_attrs(r or {}),
            "couverture (aperçu)": summarize_coverage(r or {}),
        })
    df = pd.DataFrame(rows)

    tab_all, tab_wfs, tab_shp, tab_other = st.tabs(["Toutes", "WFS", "Fichiers (SHP)", "Autres"])

    def show_df(filtered: pd.DataFrame):
        if filtered.empty:
            st.info("Aucune couche.")
        else:
            filtered = filtered.sort_values(["intersecte", "couche"], ascending=[False, True])
            st.dataframe(
                filtered.drop(columns=["origin_raw"]),  # colonne technique pour le filtre
                use_container_width=True,
                height=420
            )

    with tab_all:
        show_df(df)
    with tab_wfs:
        show_df(df[df["origin_raw"] == "wfs"])
    with tab_shp:
        show_df(df[df["origin_raw"].isin(["shp", "shape", "shapefile", "file"])])
    with tab_other:
        show_df(df[~df["origin_raw"].isin(["wfs", "shp", "shape", "shapefile", "file"])])

    # Détails
    st.markdown("### Détails par couche intersectante")
    if not hit_by_fullname:
        st.info("Aucune intersection trouvée.")
    else:
        for name, r in sorted(hit_by_fullname.items(), key=lambda kv: kv[0]):
            lyr = next((x for x in layers_all if x["full_name"] == name), {})
            origin_label = pretty_origin(lyr.get("origin"))
            spath = lyr.get("source_path") or "—"

            with st.expander(f"{name} — {r.get('count', 0)} entité(s)"):
                st.markdown(f"**Source**: {origin_label}  \n**source_path**: `{spath}`")
                st.markdown(f"**SRID**: `{r.get('srid')}` — **geom**: `{r.get('geom_col')}`")
                cov = r.get("coverage") or {}
                if cov:
                    cov_rows = []
                    for attr, rows_cov in cov.items():
                        for c in rows_cov:
                            cov_rows.append({
                                "attribut": attr,
                                "valeur": c.get("value"),
                                "surface_m²": round(float(c.get("area_m2") or 0.0), 2),
                                "% parcelle": round(100.0 * float(c.get("parcel_pct") or 0.0), 2),
                            })
                    st.dataframe(pd.DataFrame(cov_rows), use_container_width=True)
                vals = r.get("value_attributes") or {}
                if vals:
                    st.markdown("**Valeurs distinctes (échantillon)**")
                    for attr, arr in vals.items():
                        preview = ", ".join(map(str, arr[:12])) + (f" … (+{len(arr)-12})" if len(arr) > 12 else "")
                        st.write(f"- **{attr}**: {preview}")

    st.download_button(
        "💾 Télécharger le JSON brut",
        data=json.dumps(results, ensure_ascii=False, indent=2),
        file_name="intersections_result.json",
        mime="application/json",
    )

# ======================= Action au clic =======================
if run_btn:
    try:
        if mode_coords:
            parcel_coords = DEFAULT_PARCEL_COORDS_LONLAT
            parcel_label = "Parcelle (coords en dur)"
        else:
            df_communes = load_communes_csv(csv_path)
            insee = get_insee_from_csv(df_communes, commune_name.strip(), department_code.strip() or None)
            if not insee:
                st.error("Impossible de déterminer un code INSEE unique via le CSV. Vérifie le fichier/les champs.")
                parcel_coords = None
            else:
                gj = locate_parcel_no_gpd(insee, parcel_str.strip())
                if gj.get("features"):
                    ring = geojson_polygon_to_lonlat_ring(gj["features"][0].get("geometry"))
                    if ring:
                        parcel_coords = ring
                        parcel_label = f"Parcelle (WFS IGN, INSEE {insee})"
                    else:
                        st.error("La géométrie retournée n’est pas un Polygon/MultiPolygon exploitable.")
                        parcel_coords = None
                else:
                    st.error(f"Aucune géométrie trouvée pour '{parcel_str}' (INSEE {insee}).")
                    parcel_coords = None

        if parcel_coords:
            with st.spinner("Analyse en cours…"):
                results = run_intersections(
                    parcel_coords_lonlat_4326=parcel_coords,
                    per_layer_attr_values_limit=limit_values,
                    save_json=None,
                    schema_whitelist=schema_whitelist or None
                )
            # Persistance en session
            st.session_state["parcel_coords"] = parcel_coords
            st.session_state["parcel_label"] = parcel_label
            st.session_state["results"] = results
            st.session_state["schema_filter"] = schema_whitelist or SCHEMA_WHITELIST_DEFAULT
    except requests.HTTPError as e:
        st.error(f"Requête WFS échouée ({e.response.status_code})")
    except Exception as e:
        st.error(f"Erreur: {e}")

# ======================= Affichage (persistant) =======================
render_outputs()
