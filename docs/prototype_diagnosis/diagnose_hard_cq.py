import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.PROPHET.model import PROPHET
from data_pipeline.dataset import load_dataset, StandardScaler
from data_pipeline.normalization import normalize_input
from utils.metrics import compute_all_metrics_torch

torch.set_num_threads(2)
root=Path('results/cq/PROPHET_20261019_144706')
ck=torch.load(root/'checkpoint/best_model.pth',map_location='cpu',weights_only=False)
c=ck['config']; d=load_dataset(c['data_dir'],c['data'],seed=c['seed'])
m=PROPHET.from_args(c,d['test']['X'].shape[2],None,torch.device('cpu')).eval()
m.load_state_dict(ck['model_state_dict'],strict=True)
sc=StandardScaler(ck['scaler_mean'],ck['scaler_std'])
s=m.proto_module.spatial_codebook; t=m.proto_module.temporal_basis
ks,kt=s.num_protos,t.num_protos
result={'experiment':str(root),'seed':c['seed'],'num_nodes':d['test']['X'].shape[2], 'best_epoch_zero_based':ck['epoch'], 'temperature':{'spatial':s.temperature,'temporal':float(t.temp.detach())}, 'max_possible_forecast_templates':ks*kt}
with torch.no_grad():
 bank=(s.prototypes[:,None,:]+t.prototypes[None,:,:]).reshape(ks*kt,-1)
 bank=m.proto_module.proto_out_proj(bank).T[None,:,:,None]
 bank=sc.inverse_transform(m.regression_layer(m.encoder(bank))).squeeze(0).squeeze(-1).T
 for split in ['train','test']:
  ids=[]; predictions=[]; largest_error=0.0; spatial_counts=torch.zeros(ks,dtype=torch.long)
  for start in range(0,len(d[split]['X']),c['batch_size']):
   x=normalize_input(torch.from_numpy(d[split]['X'][start:start+c['batch_size']]).float(),sc)
   h,tid,diw,node,adj=m._get_embeddings({'X':x})
   xs=adj if m.use_adj else node
   if m.use_his: xs=torch.cat([xs,h.mean(1).mean(0)],-1)
   qs=F.normalize(s.proj(xs),dim=-1); ps=F.normalize(s.prototypes,dim=-1)
   si=(qs@ps.T).argmax(-1)
   xt=torch.cat(([h[:,-1]] if m.use_his else [])+[tid[:,-1],diw[:,-1]],-1)
   ti=t.allocator(xt).argmax(-1)
   code=si[None,:]*kt+ti
   spatial_counts+=torch.bincount(si,minlength=ks)*x.shape[0]
   actual=sc.inverse_transform(m({'X':x})).squeeze(-1).permute(0,2,1)
   reconstructed=bank[code]
   error=(actual-reconstructed).abs().max().item()
   largest_error=max(largest_error,error)
   torch.testing.assert_close(actual,reconstructed,rtol=2e-5,atol=2e-5)
   ids.append(code.flatten());predictions.append(actual)
  ids=torch.cat(ids); pred=torch.cat(predictions)
  counts=torch.bincount(ids,minlength=ks*kt)
  target=torch.from_numpy(d[split]['y']).float().squeeze(-1).permute(0,2,1)
  most=int(counts.argmax()); mask=ids==most
  means=target.mean(-1).flatten()[mask]
  p=counts.float()/counts.sum()
  metrics=compute_all_metrics_torch(pred,target,null_value=-1.)
  result[split]={'samples':len(d[split]['X']), 'spatial_counts':spatial_counts.tolist(),
    'temporal_counts':counts.reshape(ks,kt).sum(0).tolist(), 'used_pairs':int((counts>0).sum()),
    'pair_counts':counts.reshape(ks,kt).tolist(), 'max_pair_fraction':float(p.max()),
    'effective_pair_count_entropy':float(torch.exp(-(p*p.clamp_min(1e-20).log()).sum())),
    'reconstruction_max_error':largest_error,'MAE':metrics[0],'RMSE':metrics[1],
    'most_common_pair':[most//kt,most%kt],
    'most_common_pair_target_horizon_mean_quantiles':torch.quantile(means,torch.tensor([0.,.1,.5,.9,1.])).tolist(),
    'most_common_pair_prediction_horizon_mean':bank[most].mean().item()}
 result['comparison']={}
 for other in ['results/cq/PROPHET_20261019_134013','results/cq/PROPHET_20261019_144711']:
  p=Path(other);oc=json.loads((p/'config.json').read_text())
  result['comparison'][other]={'metrics':json.loads((p/'test_results.json').read_text()), 'config_differences':{k:[c.get(k),oc.get(k)] for k in c.keys()|oc.keys() if c.get(k)!=oc.get(k)}}
 h=json.loads((root/'training_history.json').read_text())
 result['training_curve']=[{k:e.get(k) for k in ['epoch','train_loss','val_mae','lr']} for e in h]
Path('docs/prototype_diagnosis/hard_cq.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:({a:b for a,b in v.items() if a not in ['pair_counts','spatial_counts','temporal_counts']} if k in ['train','test'] else v) for k,v in result.items() if k not in ['comparison','training_curve']},indent=2))
print('counts', {split: {name:result[split][name] for name in ['spatial_counts','temporal_counts']} for split in ['train','test']})
