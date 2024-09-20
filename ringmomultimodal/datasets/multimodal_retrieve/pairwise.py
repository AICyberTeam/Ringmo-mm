from torch.utils.data import Dataset
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import nltk
from .utils import make_vocab
from mmdet.datasets.pipelines import Compose


class PairwiseDataset(Dataset):
    def __init__(self, ann_file, pipeline, image_path, dictionary_path, test_mode=False):
        self.vocab = make_vocab(dictionary_path)
        self.img_path = image_path

        # Captions
        self.captions = []
        self.maxlength = 0
        self.matching = self.load_annotations(ann_file)
        self.pipeline = Compose(pipeline)

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            matching = [tuple(map(str.strip, line.split('|'))) for line in f.readlines()]
        return matching

    def __len__(self):
        return len(self.matching)

    def get_ann_info(self, idx):
        return self.matching[idx]

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        caption = self.captions[index]

        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]

        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)

        image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        return image, caption, tokens_UNK, index, img_id

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass
