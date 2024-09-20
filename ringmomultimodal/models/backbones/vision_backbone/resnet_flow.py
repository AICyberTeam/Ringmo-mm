from mmdet.models.backbones.resnet import ResNet
from ..utils import StageWrapper
from ...builder import BACKBONE
@BACKBONE.register_module()
class ResNetFlow(ResNet):
    def __init__(self,stage_names=None, **kwargs):
        super(ResNetFlow, self).__init__(**kwargs)
        if stage_names is None:
            self.stage_names = [f'stage_{i}' for i in range(len(self.res_layers) + 1)]
        else:
            self.stage_names = stage_names
        self.components = self.generate_stages()
    def generate_stages(self, ):
        model_flow = []
        def start_stage(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            return x
        model_flow.append(StageWrapper(start_stage, self.stage_names[0]))

        for stage_idx, layer_name in enumerate(self.res_layers):
            stage_name = self.stage_names[stage_idx + 1]
            def stage_block(x, i=stage_idx, **kwargs):
                res_layer = getattr(self, self.res_layers[i])
                x = res_layer(x)
                return x, {'output': x} if i in self.out_indices else dict()
            stage = StageWrapper(stage_block, f'{stage_name}')
            model_flow.append(stage)
        return model_flow