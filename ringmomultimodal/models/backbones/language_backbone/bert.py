from ..utils import FlowBackBoneBase
from pytorch_pretrained_bert.modeling import BertModel
from ...builder import BACKBONE


@BACKBONE.register_module()
class Bert(FlowBackBoneBase):
    def __init__(self, init_cfg=None):
        bert_model = init_cfg.pop('bert_model')
        self.tuned = init_cfg.pop('tuned')
        super(Bert, self).__init__(init_cfg)
        if bert_model == 'bert-base-uncased':
            self.textdim = 768
        else:
            self.textdim = 1024
        self.model = BertModel.from_pretrained(bert_model)

    def forward(self, word_id, word_mask):
        all_encoder_layers, pooled_output = self.model(word_id, token_type_ids=None, attention_mask=word_mask)
        feature_language = [all_encoder_layers[-1],
                            all_encoder_layers[-2],
                            all_encoder_layers[-3],
                            all_encoder_layers[-4]]
        return [(feature_language, pooled_output)]

    def train(self, mode=True):
        super(Bert, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.tuned:
            return
        for param in self.model.parameters():
            param.requires_grad = False
