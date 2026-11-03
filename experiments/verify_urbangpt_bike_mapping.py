"""Verify the OpenCity bike subset against the original UrbanGPT data.

Input: cached original ZIP member, downloaded by sources/download_urbangpt_bike.py.
The outer member is CRC-checked before decompression; every demand value in both
channels is compared, including zero-only nodes after determining the slice.
"""
from pathlib import Path
import struct,zlib,zipfile,numpy as np,json
p=Path('figs/urban_demand_maps/sources');b=(p/'urbangpt_bike_zip_entry.bin').read_bytes();h=struct.unpack_from('<4s5H3I2H',b);offset=30+h[-2]+h[-1]
out=p/'urbangpt_all_bike.npz'
if not out.exists():
 d=zlib.decompressobj(-15);crc=0;size=0
 with out.open('wb') as f:
  for i in range(offset,len(b),65536):
   block=d.decompress(b[i:i+65536]);f.write(block);crc=zlib.crc32(block,crc);size+=len(block)
 assert crc==h[6] and size==h[8],(crc,size,h)
 print('Original ZIP entry CRC verified',flush=True)
with zipfile.ZipFile(out) as z:
 info=z.getinfo('data.npy');assert info.compress_type==0
 with out.open('rb') as f:
  f.seek(info.header_offset);h=struct.unpack('<4s5H3I2H',f.read(30));f.seek(h[-2]+h[-1],1)
  version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version);data_offset=f.tell()
 assert not fortran
 source=np.memmap(out,dtype=dtype,mode='r',offset=data_offset,shape=shape).reshape(-1,105216,2)
 local=np.load('data/raw/NYC_BIKE/NYC_BIKE.npz')['data'].transpose(1,0,2)
 # Match full time series fingerprints for all nonzero nodes; use these to identify a common slice.
 means=source.mean(axis=(1,2));localmeans=local.mean(axis=(1,2));offsets=[]
 for i in np.flatnonzero(localmeans):
  candidates=np.flatnonzero(np.isclose(means,localmeans[i],rtol=0,atol=1e-12))
  candidates=[int(j) for j in candidates if np.array_equal(source[j],local[i])]
  assert len(candidates)==1,(i,candidates)
  offsets.append(candidates[0]-int(i))
 assert len(set(offsets))==1,set(offsets)
 start=offsets[0];np.testing.assert_array_equal(source[start:start+len(local)],local)
 indices=np.load('data/processed/nyc_bike/node_indices.npy')
 report=dict(status='exact_all_values_match',source_url='https://huggingface.co/datasets/bjdwh/UrbanGPT_ori_stdata/resolve/main/st_data.zip',source_member='st_data/all_nyc_bike_46x47x105216x2.npz',source_shape=list(shape),opencity_slice_start=start,opencity_slice_stop=start+len(local),timesteps=105216,channels=2,matched_nonzero_nodes=len(offsets),selected_original_grid_ids=(indices+start).tolist())
 (p/'nyc_bike_geography_identity.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
