import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from dataloader import FSL_VQA, DataLoader
from backbone import ResNet12


class BackboneInputSizeTests(unittest.TestCase):
    def test_train_and_test_episodes(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'COCO').mkdir()
            (root / 'vinvl').mkdir()
            np.save(root / 'predict_vinvl.npy', {'1': ['cat'], '2': ['dog']})
            (root / 'COCO-QA').mkdir()
            rows = []
            for label, name in enumerate(['cat', 'dog']):
                image = 'COCO_train2014_{:012d}.jpg'.format(label + 1)
                Image.new('RGB', (320, 240), color=(40, 90, 130)).save(root / 'COCO' / image)
                np.savez(root / 'vinvl' / (image + '.npz'), x=np.ones((2048, 3), dtype=np.float32))
                rows.extend([{'question': name, 'answer': name,
                              'img_id': str(label + 1), 'img_path': image}] * 2)
            for partition in ['train', 'test']:
                torch.save({'all_words': ['cat', 'dog'], 'answers_index': {'cat': 0, 'dog': 1},
                            'data': rows}, root / 'COCO-QA' / (partition + '_fpait.pth'))
            vocab = {'PAD': 0, 'UNK': 1, 'cat': 2, 'dog': 3}
            with patch('dataloader.tokenize', return_value=(vocab, np.zeros((4, 300), dtype=np.float32))), \
                    patch('dataloader.tokenize_class', return_value=(vocab, np.zeros((4, 300), dtype=np.float32))):
                for backbone, size in [('resnet12', 84), ('clip', 224)]:
                    for partition in ['train', 'test']:
                        with self.subTest(backbone=backbone, partition=partition):
                            dataset = FSL_VQA(root, category='COCO', partition=partition,
                                              token_to_ix=vocab, backbone=backbone)
                            loader = DataLoader(dataset, num_tasks=1, num_ways=2, num_shots=1,
                                                num_queries=1, epoch_size=1, num_workers=0)
                            batch = loader.get_task_batch()
                            self.assertEqual(tuple(batch[0].shape), (1, 2, 3, size, size))
                            self.assertEqual(tuple(batch[3].shape), (1, 2, 3, size, size))
                            self.assertEqual(dataset.data_size, [3, size, size])
                            if backbone == 'resnet12':
                                with torch.no_grad():
                                    outputs = ResNet12(128).eval()(batch[0][0])
                                for output in outputs:
                                    self.assertEqual(tuple(output.shape), (2, 128))
                                    self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
