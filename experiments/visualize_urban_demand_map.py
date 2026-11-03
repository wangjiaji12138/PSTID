"""Audit node geography and map raw channel-0 demand without window duplication.

Run: python -m experiments.visualize_urban_demand_map
HTML maps use Leaflet/OpenStreetMap and need internet when opened.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.grid import DataProcessor
from utils.paths import PROJECT_ROOT


DATASETS = ('cq', 'hz', 'nyc_taxi', 'chi_taxi', 'nyc_bike')


def read_header(stream):
    version = np.lib.format.read_magic(stream)
    shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    if fortran or dtype.hasobject:
        raise ValueError('Only numeric, C-order arrays are supported')
    return shape, dtype


def read_timeline(path):
    """Reconstruct stride-1 X/y timeline, streaming compressed arrays in chunks.

    Consecutive input windows must overlap exactly. This validates the stride
    assumption rather than silently applying it to other preprocessing schemes.
    """
    starts = []
    previous = None
    with zipfile.ZipFile(path) as archive:
        with archive.open('X.npy') as stream:
            shape, dtype = read_header(stream)
            samples, window, nodes, channels = shape
            sample_bytes = int(np.prod(shape[1:])) * dtype.itemsize
            for offset in range(0, samples, 256):
                count = min(256, samples - offset)
                chunk = np.frombuffer(stream.read(count * sample_bytes), dtype=dtype)
                chunk = chunk.reshape(count, window, nodes, channels)[..., 0]
                if previous is not None:
                    np.testing.assert_array_equal(previous[1:], chunk[0, :-1])
                if count > 1:
                    np.testing.assert_array_equal(chunk[:-1, 1:], chunk[1:, :-1])
                starts.append(chunk[:, 0].copy())
                previous = chunk[-1].copy()
        with archive.open('y.npy') as stream:
            yshape, ydtype = read_header(stream)
            if yshape[0] != samples or yshape[2:] != (nodes, 1):
                raise ValueError(f'Unexpected target shape: {yshape}')
            sample_bytes = int(np.prod(yshape[1:])) * ydtype.itemsize
            stream.seek((samples - 1) * sample_bytes, 1)
            last_target = np.frombuffer(stream.read(sample_bytes), dtype=ydtype)
            last_target = last_target.reshape(yshape[1:])[..., 0].copy()
    values = np.concatenate([*starts, previous[1:], last_target])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Demand contains nonfinite or negative values')
    return values


def validate_chinese_geography(dataset, metadata, values):
    """Rebuild the exact raw-CSV node mapping and verify every demand value."""
    processor = DataProcessor(city=dataset, grid_size=metadata['grid_size'])
    processor._preprocess_data()
    positions = []
    for node, west, east, south, north in processor.gridSplitter.valid_grids:
        positions.append(dict(grid_id=node, lng_min=west, lng_max=east,
                              lat_min=south, lat_max=north,
                              lng_center=(west + east) / 2,
                              lat_center=(south + north) / 2))
    if len(positions) != values.shape[1]:
        raise ValueError('Raw CSV node count differs from processed data')
    times = pd.date_range(metadata['time_range']['start'],
                          metadata['time_range']['end'], freq=metadata['time_size'])
    times = times[(times.hour >= processor.day_start_hour) &
                  (times.hour <= processor.day_end_hour)]
    frame = processor.processed_data
    counts = frame.groupby([frame.index.floor(metadata['time_size']), 'grid_id']).size()
    expected = counts.unstack(fill_value=0).reindex(
        index=times, columns=range(len(positions)), fill_value=0).fillna(0).to_numpy()
    np.testing.assert_array_equal(values, expected)
    saved = metadata.get('node_positions', [])
    if saved:
        for actual, original in zip(positions, saved):
            for key in actual:
                np.testing.assert_allclose(actual[key], original[key], atol=1e-9)
        if len(saved) != len(positions):
            raise ValueError('Saved coordinate count differs from raw CSV')
    return positions


def write_map(path, dataset, positions, demand, description, geometries=None,
              coordinate_note='坐标沿用原始 CSV，坐标系未标注，未作偏移转换。'):
    """Grid colors retain exact node values; smooth heat layer is optional."""
    records = [dict(p, demand=float(v)) for p, v in zip(positions, demand)]
    if geometries is not None:
        for record, geometry in zip(records, geometries):
            record['geometry'] = geometry
    template = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>UrbanDemand — __DATASET__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#map{height:100%;margin:0;font:14px sans-serif}.info{background:white;padding:12px;max-width:350px;box-shadow:0 1px 6px #888;line-height:1.6}.bar{height:14px;background:linear-gradient(to right,#ffffcc,#fd8d3c,#800026)}</style>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<script>
const rows=__ROWS__, max=Math.max(...rows.map(r=>r.demand));
const map=L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',maxZoom:19}).addTo(map);
const grids=L.featureGroup().addTo(map), colors=['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026'];
const boundaries=L.featureGroup().addTo(map);
L.control.scale({imperial:false,position:'bottomright'}).addTo(map);
function color(v){return colors[Math.min(8,Math.floor(9*Math.log1p(v)/Math.max(Math.log1p(max),1e-12)))];}
for(const r of rows){const style={color:'#555',weight:.5,fillColor:color(r.demand),fillOpacity:.7};
const layer=r.geometry ? L.geoJSON(r.geometry,{style:style}) : L.rectangle([[r.lat_min,r.lng_min],[r.lat_max,r.lng_max]],style);
const rectangular=!r.geometry || String(r.region_name||'').startsWith('UrbanGPT grid');
const size=rectangular ? '<br>网格约 '+((r.lng_max-r.lng_min)*111.32*Math.cos(r.lat_center*Math.PI/180)).toFixed(2)+' × '+((r.lat_max-r.lat_min)*111.32).toFixed(2)+' km（东西 × 南北）' : '<br>实际区域多边形，尺寸不固定';
const activity=r.nonzero_timesteps===undefined ? '' : '<br>有需求时间步: '+r.nonzero_timesteps+' / '+r.total_timesteps+'<br>总需求: '+r.demand_total;
const tip='节点 '+r.grid_id+(r.region_name ? ' · '+r.region_name : '')+'<br>UrbanDemand: '+r.demand.toPrecision(6)+'<br>中心: '+r.lat_center.toFixed(5)+', '+r.lng_center.toFixed(5)+size+activity;
layer.bindTooltip(tip).addTo(grids);
const outlineStyle={color:'#202735',weight:1.2,opacity:.9,fill:false};
const outline=r.geometry ? L.geoJSON(r.geometry,{style:outlineStyle}) : L.rectangle([[r.lat_min,r.lng_min],[r.lat_max,r.lng_max]],outlineStyle);
outline.bindTooltip(tip).addTo(boundaries);}
const heat=L.heatLayer(rows.map(r=>[r.lat_center,r.lng_center,r.demand/Math.max(max,1e-12)]),{radius:25,blur:18,max:1});
L.control.layers({}, {'区域需求值':grids,'实际节点边界':boundaries,'平滑热力图（随缩放变化）':heat},{collapsed:false}).addTo(map);
map.fitBounds(grids.getBounds(),{padding:[25,25]});
const info=L.control({position:'bottomleft'});info.onAdd=()=>{const d=L.DomUtil.create('div','info');d.innerHTML='<b>__DATASET__ · UrbanDemand</b><br>__DESCRIPTION__<br>颜色：log(1 + 需求值)<div class="bar"></div>0 → '+max.toFixed(4)+'<br>悬停查看原始数值。各数据集独立色标。<br>__COORDINATE_NOTE__<br>底图和脚本需要联网加载。';return d;};info.addTo(map);
</script></html>'''
    path.write_text(template.replace('__ROWS__', json.dumps(records))
                    .replace('__DATASET__', dataset.upper())
                    .replace('__DESCRIPTION__', description)
                    .replace('__COORDINATE_NOTE__', coordinate_note), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--aggregation', choices=('mean', 'sum', 'snapshot'), default='mean')
    parser.add_argument('--time-index', type=int, default=0, help='Deduplicated timeline index for snapshot')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'figs/urban_demand_maps')
    parser.add_argument('--geography-dir', type=Path,
                        default=PROJECT_ROOT / 'figs/urban_demand_maps/sources')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {}
    for dataset in args.datasets:
        print(f'Processing {dataset}', flush=True)
        folder = PROJECT_ROOT / 'data/processed' / dataset
        values = read_timeline(folder / 'all.npz')
        metadata_path = folder / 'metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if args.aggregation == 'snapshot':
            if not 0 <= args.time_index < len(values):
                raise ValueError(f'time-index must be in [0, {len(values)})')
            demand = values[args.time_index]
            description = f'第一个通道，去重时间索引 {args.time_index}'
        else:
            demand = getattr(values.astype(np.float64), args.aggregation)(axis=0)
            description = f'第一个通道，全部 {len(values)} 个唯一时间步的' + ('平均值' if args.aggregation == 'mean' else '总和')
        entry = dict(nodes=values.shape[1], unique_timesteps=len(values), channel=0,
                     aggregation=args.aggregation, time_index=args.time_index if args.aggregation == 'snapshot' else None,
                     min=float(demand.min()), max=float(demand.max()))
        positions = None
        geometries = None
        coordinate_note = '坐标沿用原始 CSV，坐标系未标注，未作偏移转换。'
        mapping_path = args.geography_dir / f'{dataset}_mapping.json'
        if dataset in ('cq', 'hz'):
            positions = validate_chinese_geography(dataset, metadata, values)
            entry.update(status='verified_against_raw_csv',
                         evidence='Rebuilt node grids and matched all channel-0 values against raw CSV counts.',
                         crs='Unspecified in source CSV; coordinates used unchanged.')
        elif dataset == 'nyc_taxi':
            entry.update(status='invalid_coordinates',
                         reason='Existing positions have identical longitude and latitude extends to 53.625; no verified node-to-region mapping in raw NPZ.')
        elif dataset == 'chi_taxi':
            entry.update(status='unverified_coordinates',
                         reason='Metadata contains a regular 11 by 7 grid, but raw files contain only demand and adjacency; node-to-region correspondence lacks source evidence.')
        else:
            indices = np.load(folder / 'node_indices.npy')
            if len(indices) != values.shape[1]:
                raise ValueError('NYC-Bike node index count mismatch')
            entry.update(status='missing_coordinates', original_node_indices=indices.tolist(),
                         reason='Selected 72 of 540 nodes; indices alone do not specify latitude/longitude or grid origin and ordering.')
        if dataset in ('nyc_taxi', 'chi_taxi', 'nyc_bike') and mapping_path.exists():
            mapping = json.loads(mapping_path.read_text())
            if len(mapping['positions']) != values.shape[1]:
                raise ValueError('External geography node count mismatch')
            if len(mapping['geometries']) != values.shape[1]:
                raise ValueError('External geometry count mismatch')
            positions, geometries = mapping['positions'], mapping['geometries']
            entry.update(status=mapping['status'], evidence=mapping['evidence'],
                         sources=mapping['sources'], crs='EPSG:4326')
            entry.pop('reason', None)
            coordinate_note = mapping.get('coordinate_note',
                '官方区域边界（WGS84）；节点顺序由公开源码和邻接关系推定，尚无原始制表脚本验证。')
        csv_path = args.output_dir / f'{dataset}_{args.aggregation}.csv'
        with csv_path.open('w', newline='') as stream:
            fields = ['node_id', 'demand'] + (list(positions[0]) if positions else [])
            if dataset == 'nyc_bike':
                fields.append('original_node_index')
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for node, value in enumerate(demand):
                row = dict(node_id=node, demand=float(value))
                if positions:
                    row.update(positions[node])
                if dataset == 'nyc_bike':
                    row['original_node_index'] = int(indices[node])
                writer.writerow(row)
        entry['csv'] = str(csv_path)
        if positions:
            map_path = args.output_dir / f'{dataset}_{args.aggregation}.html'
            write_map(map_path, dataset, positions, demand, description, geometries,
                      coordinate_note)
            entry['map'] = str(map_path)
        audit[dataset] = entry
        print(f'  {entry["status"]}: {values.shape}; {csv_path}', flush=True)
    report = args.output_dir / 'geography_audit.json'
    if report.exists():
        audit = {**json.loads(report.read_text()), **audit}
    report.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(f'Audit: {report}')


if __name__ == '__main__':
    main()
