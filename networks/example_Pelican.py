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
        particle_mask = l_v[...,0] > 1.105171 #log(E) > 0.1
        edge_mask = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)
        data = {"Pmu":l_v.cuda(), "particle_mask":particle_mask.cuda(),"edge_mask":edge_mask.cuda()}#,"scalars":None} 
        return self.mod(data)


def get_model(data_config, **kwargs):
    device = torch.device('cuda')
    cfg = dict(num_channels_scalar = 78,
               	num_channels_m = [[132],]*5,
		num_channels_2to2 = [78,]*5,
		num_channels_out = [132],
                num_channels_m_out = [132, 78],
                device=device, dtype=None,
		activation='leakyrelu',
                add_beams=False,
                batchnorm='b',
                average_nobj=49,#CHECK THIS
#num_classes=2,activate_agg_in=False, activate_lin_in=True,activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, read_pid=False, config='s', config_out='s', average_nobj=49, factorize=False, masked=True,
                 #activate_agg_out=True, activate_lin_out=False, mlp_out=True,scale=1, irc_safe=False, dropout = False, drop_rate=0.1, drop_rate_out=0.1, batchnorm=None,
                 #weaver misc
                trim=True,
                for_inference=False)
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = PELICANClassifierWrapper(**cfg)
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': [cfg['activation']],
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss(reduction='none')
