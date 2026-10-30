import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from dataloader import FSL_VQA


class CocoQASplitTests(unittest.TestCase):
    def test_new_and_legacy_splits_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'COCO').mkdir()
            (root / 'vinvl').mkdir()
            np.save(root / 'predict_vinvl.npy', {'1': ['cat']})
            image_name = 'COCO_train2014_000000000001.jpg'
            Image.new('RGB', (224, 224)).save(root / 'COCO' / image_name)
            np.savez(root / 'vinvl' / (image_name + '.npz'), x=np.ones((2048, 3), dtype=np.float32))
            for folder, suffix, size in (('COCO-QA', '_fpait', 2), ('FSL COCO-QA', '', 1)):
                (root / folder).mkdir()
                row = {'question': 'cat', 'answer': 'cat', 'img_id': '1', 'img_path': image_name}
                for partition in ('train', 'test'):
                    torch.save({'all_words': ['cat'], 'answers_index': {'cat': 0}, 'data': [row] * size},
                               root / folder / (partition + suffix + '.pth'))
            vocab = {'PAD': 0, 'UNK': 1, 'cat': 2}
            with patch('dataloader.tokenize', return_value=(vocab, np.zeros((3, 300), dtype=np.float32))), \
                    patch('dataloader.tokenize_class', return_value=(vocab, np.zeros((3, 300), dtype=np.float32))):
                train = FSL_VQA(str(root), category='COCO')
                test = FSL_VQA(str(root), category='COCO', partition='test',
                               token_to_ix=train.token_to_ix)
                legacy = FSL_VQA(str(root), category='COCO',
                                 split_file='FSL COCO-QA/train.pth')
                self.assertEqual(len(train), 2)
                self.assertEqual(len(test), 2)
                self.assertEqual(len(legacy), 1)
                self.assertEqual(Path(train.split_path), root / 'COCO-QA' / 'train_fpait.pth')
                self.assertEqual(Path(test.split_path), root / 'COCO-QA' / 'test_fpait.pth')
                self.assertEqual(Path(legacy.split_path), root / 'FSL COCO-QA' / 'train.pth')
                self.assertEqual(Path(train.image_path[0]), root / 'COCO' / image_name)
                absolute = FSL_VQA(str(root), category='COCO',
                                   split_file=str(root / 'FSL COCO-QA' / 'train.pth'))
                self.assertEqual(len(absolute), 1)
                with self.assertRaises(FileNotFoundError):
                    FSL_VQA(str(root), category='COCO',
                            split_file='missing/train.pth')
                train[0]
                test[0]
                with self.assertRaisesRegex(FileNotFoundError, 'val_fpait.pth'):
                    FSL_VQA(str(root), category='COCO', partition='val')


if __name__ == '__main__':
    unittest.main()
