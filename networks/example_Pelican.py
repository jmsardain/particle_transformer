import torch
from weaver.nn.model.pelican.models.pelican_classifier import PELICANClassifier
from weaver.utils.logger import _logger
import awkward as ak
'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''


class PELICANClassifierWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = PELICANClassifier(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        l_v = torch.stack((lorentz_vectors[:,3,:],lorentz_vectors[:,0,:],lorentz_vectors[:,1,:],lorentz_vectors[:,2,:]),2)
        s = l_v.shape

        if self.cfg["add_beams"] and not self.cfg["read_pid"]:
            p = 1
            beams = torch.tensor([[[p,0,0,p], [p,0,0,-p]]], dtype=l_v.dtype, device=l_v.device).expand(s[0], 2, 4)
            l_v = torch.cat([beams, l_v], dim=1)
            num_classes=2
            pdgid = torch.cat([2212 * torch.ones(s[0], 2, dtype=torch.long, device=l_v.device),
                                       torch.zeros(s[0], s[1], dtype=torch.long, device=l_v.device)], dim=1)

        particle_mask = torch.square(l_v[...,1]) + torch.square(l_v[...,2]) > 1.105171**2 #log(pt) > 0.1
        edge_mask = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)
        
        if self.cfg["add_beams"]:
            scalars = pdg_onehot(pdgid, num_classes=num_classes, mask=particle_mask.unsqueeze(-1))
            data = {"Pmu":l_v.cuda(), "particle_mask":particle_mask.cuda(),"edge_mask":edge_mask.cuda(),"scalars":scalars.cuda()} 
        else:
            data = {"Pmu":l_v.cuda(), "particle_mask":particle_mask.cuda(),"edge_mask":edge_mask.cuda()} 
        #from torchsummary import summary
        #summary(self.mod,self.mod.prepare_input(data))
        #raise SystemExit
 
        return self.mod(data)


def pdg_onehot(x, num_classes=14, mask=None):
    if num_classes==14:
        x = 0*(x==22) + 1*(x==211) + 2*(x==-211) + 3*(x==321) + 4*(x==-321) + 5*(x==130) + 6*(x==2112) + 7*(x==-2112) + 8*(x==2212) + 9*(x==-2212) + 10*(x==11) + 11*(x==-11) + 12*(x==13) + 13*(x==-13)
    elif num_classes==2:
        x = 0*(x!=2212) + 1*(x==2212)
    x = torch.nn.functional.one_hot(x, num_classes=num_classes)
    zero = torch.tensor(0, device=x.device, dtype=torch.long)
    if mask is not None:
        x = torch.where(mask, x, zero)
    
    return x

def get_model(data_config, **kwargs):
    device = torch.device('cuda')
    cfg = dict(num_channels_scalar = 35,
               	num_channels_m = [[60],]*5,
		num_channels_2to2 = [35,]*5,
		num_channels_out = [60],
                num_channels_m_out = [60, 35],
                device=device, dtype=None,
		activation='leakyrelu',
                add_beams=True,
                read_pid=False,
                batchnorm='b',
                average_nobj=51.83,#CHECK THIS
#num_classes=2,activate_agg_in=False, activate_lin_in=True,activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, read_pid=False, config='s', config_out='s', average_nobj=49, factorize=False, masked=True,
                 #activate_agg_out=True, activate_lin_out=False, mlp_out=True,scale=1, irc_safe=False, dropout = False, drop_rate=0.1, drop_rate_out=0.1, batchnorm=None,
                 #weaver misc
                trim=True,
                for_inference=False)
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = PELICANClassifierWrapper(**cfg)
    model.cfg = cfg
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': [cfg['activation']],
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss(reduction='none')
