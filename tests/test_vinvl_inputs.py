import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from dataloader import FSL_VQA


class VinVLInputTests(unittest.TestCase):
    def test_separate_split_image_and_vinvl_roots(self):
            """Splits, images and VinVL features may live in three different roots."""
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                split_root = base / 'splits'
                image_root = base / 'images'
                vinvl_root = base / 'features'
                (split_root / 'COCO-QA').mkdir(parents=True)
                (image_root / 'COCO').mkdir(parents=True)
                (vinvl_root / 'vinvl').mkdir(parents=True)
                image_name = 'COCO_train2014_000000000042.jpg'
                Image.new('RGB', (224, 224)).save(image_root / 'COCO' / image_name)
                np.savez(vinvl_root / 'vinvl' / (image_name + '.npz'),
                         x=np.zeros((2048, 3), dtype=np.float32))
                np.save(vinvl_root / 'predict_vinvl.npy', np.array({'42': ['cat', 'dog']}, dtype=object))
                torch.save({'all_words': ['cat'], 'answers_index': {'cat': 0},
                            'data': [{'question': 'cat', 'img_path': image_name,
                                      'img_id': '42', 'answer': 'cat'}]},
                           split_root / 'COCO-QA' / 'train_fpait.pth')
                vocabulary = {'PAD': 0, 'UNK': 1, 'cat': 2, 'dog': 3}
                embeddings = np.zeros((4, 300), dtype=np.float32)
                with patch('dataloader.tokenize', return_value=(vocabulary, embeddings)), \
                        patch('dataloader.tokenize_class', return_value=(vocabulary, embeddings)):
                    dataset = FSL_VQA(str(split_root), 
                                      image_root=str(image_root), vinvl_root=str(vinvl_root))
                    sample = dataset[0]
                self.assertEqual(Path(dataset.split_path), split_root / 'COCO-QA' / 'train_fpait.pth')
                self.assertEqual(Path(dataset.image_path[0]), image_root / 'COCO' / image_name)
                self.assertEqual(Path(dataset.vinvl_path[0]), vinvl_root / 'vinvl' / (image_name + '.npz'))
                self.assertEqual(sample[3].shape, (50, 2048))
                self.assertEqual(int(sample[4][0]), vocabulary['cat'])
