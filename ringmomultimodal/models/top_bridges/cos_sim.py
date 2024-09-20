import torch.nn as nn
import torch


class HVSA(nn.Module):
    def __init__(self, config={}, vocab_words=[]):
        super(HVSA, self).__init__()
        self.Eiters = 0

    def forward(self, img_feature, text_feature, text_lens):
        if self.training is True:
            self.Eiters += 1

        batch_img = img_feature.shape[0]
        batch_text = text_feature.shape[0]
        dual_sim = cosine_similarity(img_feature.unsqueeze(dim=1).expand(-1, batch_text, -1),
                                     text_feature.unsqueeze(dim=0).expand(batch_img, -1, -1))


def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)

    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
