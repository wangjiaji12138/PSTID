"""Restore map coordinates from cached UrbanGPT and official city geography.

Requires numpy, shapely and pyshp. Sources are cached in
figs/urban_demand_maps/sources; see geography_sources.md for provenance.
Run before experiments.visualize_urban_demand_map and plot_urban_demand_heatmap.
"""
import io
import json
import zipfile

import numpy as np
import shapefile
from shapely.geometry import shape, box, mapping

from utils.paths import PROJECT_ROOT

SOURCES = PROJECT_ROOT / 'figs/urban_demand_maps/sources'
ORIGINAL = 'https://huggingface.co/datasets/bjdwh/UrbanGPT_ori_stdata/resolve/main/st_data.zip'


def export(dataset, regions, evidence, sources, note, status):
    positions, geometries = [], []
    for node, (region_id, name, polygon) in enumerate(regions):
        west, south, east, north = polygon.bounds
        center = polygon.centroid
        positions.append(dict(grid_id=node, region_id=int(region_id), region_name=name,
                              lng_min=west, lng_max=east, lat_min=south, lat_max=north,
                              lng_center=center.x, lat_center=center.y))
        geometries.append(mapping(polygon))
    assert len(positions) == len({p['region_id'] for p in positions})
    for p in positions:
        assert -90 < p['lng_center'] < -70 and 40 < p['lat_center'] < 43
    result = dict(status=status, evidence=evidence, sources=sources,
                  coordinate_note=note, positions=positions, geometries=geometries)
    path = SOURCES / f'{dataset}_mapping.json'
    path.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    print(path, len(positions), flush=True)


def taxi():
    # The newer GeoJSON merges LocationIDs 57, 104, 105 into other IDs.
    # Match its unchanged area AND perimeter to the historical TLC shapefile
    # so all 263 original IDs retain their own geometry.
    features = json.loads((SOURCES / 'nyc_zones.geojson').read_text())['features']
    with zipfile.ZipFile(SOURCES / 'nyc_zones.zip') as archive:
        reader = shapefile.Reader(
            shp=io.BytesIO(archive.read('taxi_zones/taxi_zones.shp')),
            dbf=io.BytesIO(archive.read('taxi_zones/taxi_zones.dbf')))
        regions, matched = [], set()
        for record in reader.records():
            candidates = [i for i, f in enumerate(features)
                          if abs(float(f['properties']['shape_area']) - record['Shape_Area']) < 1e-14
                          and abs(float(f['properties']['shape_leng']) - record['Shape_Leng']) < 1e-12]
            assert len(candidates) == 1, record
            i = candidates[0]
            assert i not in matched
            matched.add(i)
            regions.append((record['LocationID'], record['zone'], shape(features[i]['geometry'])))
    regions.sort(key=lambda r: r[0])
    assert [r[0] for r in regions] == list(range(1, 264))
    export('nyc_taxi', regions,
           'Raw NPZ CRC matches UrbanGPT and OpenCity. UrbanGPT region_decode_ori uses node+1 '
           'for taxi zone IDs. All 263 historical TLC shapes match official WGS84 GeoJSON '
           'uniquely by area and perimeter, including historical IDs 57, 104 and 105.',
           [ORIGINAL, 'https://github.com/HKUDS/UrbanGPT/blob/main/instruction_generate/instruction_generate.py',
            'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip',
            'https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/8meu-9t5y'],
           'TLC 官方区域边界（WGS84）；节点编号按 UrbanGPT 源码对应 LocationID = 节点 + 1。',
           'source_code_and_official_boundary_mapping')

    features = json.loads((SOURCES / 'chicago_official.geojson').read_text())['features']
    regions = sorted([(int(f['properties']['area_numbe']), f['properties']['community'],
                       shape(f['geometry'])) for f in features], key=lambda r: r[0])
    assert [r[0] for r in regions] == list(range(1, 78))
    # Corroborate numbering against independent source adjacency. This graph
    # also includes nearby non-touching regions, so equality is not expected.
    adjacent = np.array([[p.distance(q) < 1e-7 for _, _, q in regions] for _, _, p in regions])
    original = np.load(PROJECT_ROOT / 'data/raw/CHI_TAXI/CHI_TAXI_rn_adj.npy') > 0
    recall = float((adjacent & original).sum() / adjacent.sum())
    assert recall > .98, recall
    export('chi_taxi', regions,
           f'Raw NPZ CRC matches UrbanGPT and OpenCity. Community Area IDs 1–77 map to '
           f'nodes 0–76; ordering corroborated by UrbanGPT indexed POI descriptions and '
           f'{recall:.4%} of official touching-region pairs present in source adjacency. '
           'Original trip aggregation script is not published; ordering is source-corroborated.',
           [ORIGINAL, 'https://github.com/HKUDS/UrbanGPT/blob/main/instruction_generate/poi/CHI_taxi_POIs.json',
            'https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6'],
           '芝加哥官方社区边界（WGS84）；按社区编号排列，已用源 POI 与邻接关系交叉核对。',
           'official_boundaries_source_corroborated_order')


def bike():
    identity = json.loads((SOURCES / 'nyc_bike_geography_identity.json').read_text())
    assert identity['status'] == 'exact_all_values_match'
    meta = json.loads((SOURCES / 'zone_region_map_meta.json').read_text())
    anchors = np.array([[r['rid'] // 47, r['rid'] % 47, *r['lat_lon']]
                        for group in meta.values() for r in group])
    lat_step, lat_origin = np.polyfit(anchors[:, 0], anchors[:, 2], 1)
    lon_step, lon_origin = np.polyfit(anchors[:, 1], anchors[:, 3], 1)
    np.testing.assert_allclose(anchors[:, 0] * lat_step + lat_origin, anchors[:, 2], rtol=0, atol=1e-10)
    np.testing.assert_allclose(anchors[:, 1] * lon_step + lon_origin, anchors[:, 3], rtol=0, atol=1e-10)
    indices = np.load(PROJECT_ROOT / 'data/processed/nyc_bike/node_indices.npy')
    ids = indices + identity['opencity_slice_start']
    np.testing.assert_array_equal(ids, identity['selected_original_grid_ids'])
    regions = []
    for rid in ids:
        lat = lat_origin + (rid // 47) * lat_step
        lon = lon_origin + (rid % 47) * lon_step
        regions.append((rid, f'UrbanGPT grid {rid}',
                        box(lon-lon_step/2, lat-lat_step/2, lon+lon_step/2, lat+lat_step/2)))
    export('nyc_bike', regions,
           'All 105216 time steps and both channels of the 540-node local/OpenCity array '
           'exactly match UrbanGPT flattened 46x47 grid slice [1077:1617]. The 72 retained '
           'nodes use saved node_indices.npy. Grid coordinates reconstructed from all '
           '806 published rid/lat_lon anchors with residual below 1e-10 degrees.',
           [ORIGINAL, 'https://arxiv.org/html/2403.00813v1#A1.SS1.SSS1',
            'https://huggingface.co/datasets/hkuds/OpenCity-dataset/blob/main/NYC_BIKE-3.zip'],
           '坐标来自 UrbanGPT 原始网格经纬度；OpenCity 子集和本项目 72 节点索引已逐值核实。',
           'verified_against_original_grid_and_all_values')


if __name__ == '__main__':
    taxi()
    bike()
