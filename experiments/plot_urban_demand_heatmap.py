"""Export paper-style demand heatmaps over cached OpenStreetMap raster tiles.

First run ``python -m experiments.visualize_urban_demand_map`` to export means.
Then run ``python -m experiments.plot_urban_demand_heatmap``.
Only CQ/HZ have raw-CSV-verified node geography. No generated imagery is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import FuncNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.cm import ScalarMappable
from shapely.geometry import shape
import numpy as np
import pandas as pd
from PIL import Image
import requests
from scipy.ndimage import gaussian_filter

from utils.paths import PROJECT_ROOT


NAMES = {'cq': 'Chongqing', 'hz': 'Hangzhou', 'nyc_taxi': 'NYC-Taxi',
         'chi_taxi': 'CHI-Taxi', 'nyc_bike': 'NYC-Bike'}
EARTH_KM = 40075.016686


def pixels(longitude, latitude, zoom):
    size = 256 * 2 ** zoom
    lat = np.radians(np.clip(latitude, -85.0511, 85.0511))
    return ((np.asarray(longitude) + 180) / 360 * size,
            (1 - np.arcsinh(np.tan(lat)) / np.pi) / 2 * size)


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return np.interp(quantiles, cumulative / cumulative[-1], values[order])


def bounds(frame, extent, width, height, padding=1.65):
    lon, lat, weights = (frame[k].to_numpy() for k in ('lng_center', 'lat_center', 'demand'))
    if weights.sum() <= 0:
        raise ValueError('Cannot plot a dataset with zero total demand')
    if extent == 'core':
        peak = int(np.argmax(weights))
        cx, cy = pixels(lon[peak], lat[peak], 0)
        # Explicit 80 km city-core view centered on the highest-demand grid.
        dx = 80 / (EARTH_KM * np.cos(np.radians(lat[peak]))) * 256
        dy = dx * height / width
        zoom = int(np.clip(math.floor(math.log2(width / dx)), 3, 15))
        return zoom, np.array([cx - dx / 2, cy - dy / 2,
                               cx + dx / 2, cy + dy / 2]) * 2 ** zoom
    if extent == 'all':
        west, east = frame.lng_min.min(), frame.lng_max.max()
        south, north = frame.lat_min.min(), frame.lat_max.max()
    else:
        west, east = weighted_quantile(lon, weights, [.005, .995])
        south, north = weighted_quantile(lat, weights, [.005, .995])
    x, y = pixels(np.array([west, east]), np.array([north, south]), 0)
    cx, cy = x.mean(), y.mean()
    # Padding leaves geographic context visible around the demand footprint.
    dx = max(x[1] - x[0], .15) * padding
    dy = max(y[1] - y[0], .15) * padding
    dx = max(dx, dy * width / height)
    dy = dx * height / width
    zoom = int(np.clip(math.floor(math.log2(width / dx)), 3, 15))
    scale = 2 ** zoom
    return zoom, np.array([cx - dx / 2, cy - dy / 2,
                           cx + dx / 2, cy + dy / 2]) * scale


def basemap(zoom, box, cache):
    x0, y0, x1, y1 = np.round(box).astype(int)
    tx0, ty0, tx1, ty1 = x0 // 256, y0 // 256, (x1 - 1) // 256, (y1 - 1) // 256
    jobs = [(x, y) for x in range(tx0, tx1 + 1) for y in range(ty0, ty1 + 1)]

    def tile(job):
        x, y = job
        path = cache / str(zoom) / str(x) / f'{y}.png'
        if not path.exists():
            for attempt in range(3):
                try:
                    response = requests.get(f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png',
                                            headers={'User-Agent': 'PSTID-UrbanDemand-Research/1.0'},
                                            timeout=(10, 25))
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
            # Validate before caching so failed downloads never become map tiles.
            Image.open(BytesIO(response.content)).verify()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return x, y, Image.open(path).convert('RGB')

    canvas = Image.new('RGB', ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for x, y, image in executor.map(tile, jobs):
            canvas.paste(image, ((x - tx0) * 256, (y - ty0) * 256))
    return canvas.crop((x0 - tx0 * 256, y0 - ty0 * 256,
                        x1 - tx0 * 256, y1 - ty0 * 256)), (x0, y0, x1, y1)


def draw_node_boundaries(ax, dataset, frame, args, zoom, pixel_box):
    """Project actual node polygons, including all multipart rings and holes."""
    x0, y0, x1, y1 = pixel_box
    mapping_path = args.input_dir / 'sources' / f'{dataset}_mapping.json'
    if dataset in ('nyc_taxi', 'chi_taxi', 'nyc_bike'):
        source = json.loads(mapping_path.read_text())
        assert len(source['positions']) == len(frame) == len(source['geometries'])
        np.testing.assert_array_equal([p['grid_id'] for p in source['positions']], frame.node_id)
        geometries = source['geometries']
    else:
        geometries = [dict(type='Polygon', coordinates=[[
            [r.lng_min, r.lat_min], [r.lng_max, r.lat_min],
            [r.lng_max, r.lat_max], [r.lng_min, r.lat_max],
            [r.lng_min, r.lat_min]]]) for r in frame.itertuples()]
    lines = []
    norm = FuncNorm((np.log1p, np.expm1), vmin=0, vmax=max(float(frame.demand.max()), 1e-12))
    cmap = plt.get_cmap('YlOrRd')
    labeled_ids = []
    for node, geometry in enumerate(geometries):
        assert geometry['type'] in ('Polygon', 'MultiPolygon')
        polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
        vertices, codes = [], []
        for polygon in polygons:
            for ring_index, ring in enumerate(polygon):
                coords = np.asarray(ring)
                px, py = pixels(coords[:, 0], coords[:, 1], zoom)
                projected = np.column_stack((px - x0, py - y0))
                lines.append(projected)
                # Opposite winding for holes, independent of source ring orientation.
                signed_area = np.sum(projected[:-1, 0]*projected[1:, 1] - projected[1:, 0]*projected[:-1, 1])
                if (signed_area > 0) != (ring_index == 0):
                    projected = projected[::-1]
                vertices.extend(projected)
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO]*(len(projected)-2) + [MplPath.CLOSEPOLY])
        if args.layer == 'cells':
            value = float(frame.demand.iloc[node])
            color = '#9b9b9b' if not np.isfinite(value) else ('#ffffff' if value == 0 else cmap(norm(value)))
            ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor='none', zorder=3))
        point = shape(geometry).representative_point()
        label_x, label_y = pixels(point.x, point.y, zoom)
        if x0 <= label_x <= x1 and y0 <= label_y <= y1:
            node_id = int(frame.node_id.iloc[node])
            labeled_ids.append(node_id)
            ax.text(label_x-x0, label_y-y0, str(node_id), ha='center', va='center', fontsize=3.3,
                    color='white' if norm(frame.demand.iloc[node]) > .65 and args.layer == 'cells' else '#111',
                    zorder=5.5, clip_on=True)
    # A narrow white casing keeps black boundaries legible over both heat and roads.
    ax.add_collection(LineCollection(lines, colors='white', linewidths=1.05, alpha=.85, zorder=4))
    ax.add_collection(LineCollection(lines, colors='#202735', linewidths=.55, alpha=.95, zorder=5))
    rectangular = dataset in ('cq', 'hz', 'nyc_bike')
    if rectangular:
        widths = (frame.lng_max-frame.lng_min) * EARTH_KM / 360 * np.cos(np.radians(frame.lat_center))
        heights = (frame.lat_max-frame.lat_min) * EARTH_KM / 360
        note = (f'{len(labeled_ids)}/{len(frame)} nodes shown (IDs 0–{len(frame)-1})\n'
                f'Cell size ≈ {widths.median():.2f} × {heights.median():.2f} km (E–W × N–S)')
    else:
        note = f'{len(labeled_ids)}/{len(frame)} nodes shown (IDs 0–{len(frame)-1})\nIrregular regions; a node may have multiple pieces'
    if args.extent != 'all':
        note += '\nCity-core view; some nodes outside frame' if args.extent == 'core' else '\nDemand-focused view; some nodes may be outside frame'
    ax.text(.025, .975, note, transform=ax.transAxes, va='top', fontsize=7.5, zorder=6,
            bbox=dict(facecolor='white', alpha=.92, edgecolor='#777', linewidth=.4, pad=4))
    if args.layer == 'cells':
        cax = ax.inset_axes([.61, .07, .34, .025], zorder=8)
        colorbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        ticks = np.expm1(np.linspace(0, np.log1p(frame.demand.max()), 4))
        colorbar.set_ticks(ticks, labels=[f'{v:.2g}' for v in ticks])
        colorbar.ax.tick_params(labelsize=5, pad=1)
        cax.set_title('Per-node mean demand / slot (log scale)', fontsize=5.5,
                      bbox=dict(facecolor='white', alpha=.95, edgecolor='none', pad=2))
        ax.text(.975, .17, 'Pale yellow = low positive demand\nWhite = zero; gray = missing',
                transform=ax.transAxes, ha='right', fontsize=5.5, zorder=8,
                bbox=dict(facecolor='white', alpha=.95, edgecolor='none', pad=2))
    # Local Web Mercator scale at the latitude of the scale bar itself.
    width, height = x1-x0, y1-y0
    y = .94 * height
    latitude = np.arctan(np.sinh(np.pi*(1-2*(y0+y)/(256*2**zoom))))
    km_per_pixel = EARTH_KM * np.cos(latitude) / (256*2**zoom)
    target_km = width * .18 * km_per_pixel
    choices = np.array([1, 2, 5])[:, None] * 10. ** np.arange(-3, 5)[None, :]
    length_km = float(choices[choices <= target_km].max())
    start, end = width*.04, width*.04 + length_km/km_per_pixel
    ax.plot([start, end], [y, y], color='white', linewidth=5, zorder=6)
    ax.plot([start, end], [y, y], color='#111', linewidth=2, zorder=7)
    ax.plot([start, start, end, end], [y-4, y+4, y+4, y-4],
            linestyle='none', marker='|', color='#111', markersize=7, zorder=7)
    ax.text((start+end)/2, y-9, f'{length_km:g} km', ha='center', va='bottom', fontsize=8,
            zorder=7, bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=1))
    assert len(labeled_ids) == len(set(labeled_ids))
    if args.extent == 'all':
        assert sorted(labeled_ids) == sorted(frame.node_id.tolist())
    return dict(boundary_nodes=len(geometries), boundary_rings=len(lines), labeled_node_ids=labeled_ids,
                boundary_type='rectangular grid' if rectangular else 'official region polygons',
                scale_bar_km=length_km,
                grid_width_km_median=float(widths.median()) if rectangular else None,
                grid_height_km_median=float(heights.median()) if rectangular else None)


def render_panel(ax, dataset, frame, args):
    zoom, box = bounds(frame, args.extent, 900, 680, getattr(args, 'extent_padding', 1.65))
    background, box = basemap(zoom, box, getattr(args, 'tile_cache', args.output_dir / 'tile_cache'))
    x0, y0, x1, y1 = box
    px, py = pixels(frame.lng_center.to_numpy(), frame.lat_center.to_numpy(), zoom)
    width, height = background.size
    km_per_pixel = EARTH_KM * np.cos(np.radians(frame.lat_center.mean())) / (256 * 2 ** zoom)
    sigma = args.bandwidth_km / km_per_pixel
    pad = math.ceil(4 * sigma)
    density, _, _ = np.histogram2d(py, px, bins=(height + 2 * pad, width + 2 * pad),
                                  range=((y0 - pad, y1 + pad), (x0 - pad, x1 + pad)),
                                  weights=frame.demand)
    density = gaussian_filter(density, sigma, mode='constant')[pad:-pad, pad:-pad] / km_per_pixel ** 2
    normalized = density / max(float(density.max()), 1e-12)
    # Nonlinear display reveals weaker local hotspots; the saved field remains linear.
    rgba = plt.get_cmap('jet')(normalized ** .35)
    rgba[..., 3] = .74 * np.clip((normalized - .008) / .045, 0, 1)
    ax.imshow(background, extent=(0, width, height, 0))
    if args.layer == 'smooth':
        ax.imshow(rgba, extent=(0, width, height, 0), interpolation='bilinear')
    boundary_info = (draw_node_boundaries(ax, dataset, frame, args, zoom, box)
                     if not getattr(args, 'paper_three', False) else dict(boundaries_shown=False))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
    if not np.isfinite(density).all() or density.max() <= 0:
        raise ValueError(f'Invalid density for {dataset}')
    if not getattr(args, 'paper_three', False):
        np.savez_compressed(args.output_dir / f'{dataset}_density.npz', density=density,
                            pixel_bounds=np.array(box), zoom=zoom)
    return dict(dataset=dataset, **boundary_info, aggregation='mean of channel 0 over unique timesteps',
                bandwidth_km=args.bandwidth_km, density_units='mean demand per time slot per km^2',
                layer=args.layer,
                color_transform='log1p(node mean demand)' if args.layer == 'cells' else '(density / panel_max)^0.35',
                color_note='Independent color scale per dataset. Cell colors use exact node means; no interpolation.' if args.layer == 'cells' else 'Smoothed density; not individual node values.',
                density_max=float(density.max()), zoom=zoom, extent_mode=args.extent,
                displayed_demand_fraction=float(frame.demand[inside].sum() / frame.demand.sum()),
                displayed_nodes=int(inside.sum()), total_nodes=len(frame),
                basemap='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                geography=args.geography_audit.get(dataset, {}).get('evidence', 'See geography_audit.json'),
                crs_note=args.geography_audit.get(dataset, {}).get('crs', 'Unspecified'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(NAMES), default=list(NAMES))
    parser.add_argument('--input-dir', type=Path, default=PROJECT_ROOT / 'figs/urban_demand_maps')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'figs/urban_demand_maps/static')
    parser.add_argument('--bandwidth-km', type=float, default=3.0)
    parser.add_argument('--layer', choices=('cells', 'smooth'), default='cells')
    parser.add_argument('--paper-three', action='store_true',
                        help='Export only a combined smooth PDF: CQ urban core, CHI-Taxi, NYC-Bike; no node overlays')
    parser.add_argument('--extent', choices=('core', 'demand', 'all'), default='all',
                        help='Core: 80 km view around the peak grid; demand: padded 0.5–99.5%% weighted range; all: every node')
    args = parser.parse_args()
    if args.paper_three:
        args.datasets = ['cq', 'chi_taxi', 'nyc_bike']
        args.layer = 'smooth'
    if args.bandwidth_km <= 0:
        parser.error('--bandwidth-km must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.input_dir / 'geography_audit.json'
    args.geography_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    font_dir = Path.home() / '.local/share/fonts/msttcorefonts'
    for font in font_dir.glob('*.[tT][tT][fF]'):
        font_manager.fontManager.addfont(str(font))
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman', weight='bold'),
                          fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold',
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, len(args.datasets), figsize=(5.1 * len(args.datasets), 4.35),
                             squeeze=False)
    manifest = []
    for i, (dataset, ax) in enumerate(zip(args.datasets, axes[0])):
        print(f'Rendering {dataset}', flush=True)
        frame = pd.read_csv(args.input_dir / f'{dataset}_mean.csv')
        if args.paper_three:
            args.extent = 'core' if dataset == 'cq' else 'demand'
            args.bandwidth_km = {'cq': 3.0, 'chi_taxi': 1.5, 'nyc_bike': 0.5}[dataset]
            args.extent_padding = 1.15 if dataset == 'chi_taxi' else 1.65
            args.tile_cache = args.input_dir / 'static' / ('tile_cache' if dataset == 'cq' else 'other_three/tile_cache')
        manifest.append(render_panel(ax, dataset, frame, args))
        label = 'CQ (Urban Core)' if args.paper_three and dataset == 'cq' else NAMES[dataset]
        ax.text(.5, -.09, f'({chr(97 + i)}) {label}', transform=ax.transAxes,
                ha='center', va='top', fontsize=23, fontweight='bold')
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.17, wspace=.06)
    stem = args.output_dir / 'urban_demand_heatmaps'
    if args.paper_three:
        fig.savefig(stem.with_suffix('.pdf'), dpi=350, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        print(stem.with_suffix('.pdf'))
        return
    for extension in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.' + extension), dpi=350, facecolor='white', bbox_inches='tight')
    # Export each panel from the same rendered figure, without fetching tiles again.
    renderer = fig.canvas.get_renderer()
    for dataset, ax in zip(args.datasets, axes[0]):
        box = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(.08)
        for extension in ('png', 'pdf'):
            fig.savefig(args.output_dir / f'{dataset}_heatmap.{extension}', dpi=350,
                        facecolor='white', bbox_inches=box)
    plt.close(fig)
    for dataset in args.datasets:
        density = np.load(args.output_dir / f'{dataset}_density.npz')['density']
        if not np.isfinite(density).all() or density.max() <= 0:
            raise ValueError(f'Invalid density for {dataset}')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
