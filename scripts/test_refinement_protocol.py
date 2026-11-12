"""Check train-only weights and deterministic label interventions, not claimed accuracy."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import segmentation as seg


class RefinementProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=seg.ROOT/'work')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        image=self.root/'image.jpg';mask=self.root/'mask.png'
        Image.new('RGB',(512,512),'#334455').save(image)
        value=np.zeros((512,512),dtype=np.uint8);value[200:205,200:205]=7
        Image.fromarray(value).save(mask)
        self.row={'image':str(image),'mask':str(mask)}

    def test_weights_read_only_supplied_training_masks(self):
        original=seg.Image.open
        calls=[]
        def audited_open(path,*args,**kwargs):
            calls.append(Path(path).resolve());return original(path,*args,**kwargs)
        with patch.object(seg.Image,'open',side_effect=audited_open):
            evidence=seg.training_evidence([self.row],'inverse-sqrt')
        self.assertEqual(calls,[Path(self.row['mask']).resolve()])
        weights=np.array(evidence['class_weights'])
        self.assertTrue(np.isfinite(weights).all());self.assertTrue((weights>0).all())
        self.assertGreater(weights[7],weights[0])
        self.assertEqual(evidence['train_pixel_counts']['human'],25)

    def test_weighted_loss_increases_rare_class_gradient_relative_to_common(self):
        evidence=seg.training_evidence([self.row],'inverse-sqrt')
        logits=torch.zeros(1,8,1,2,requires_grad=True)
        target=torch.tensor([[[0,7]]])
        F.cross_entropy(logits,target,weight=torch.tensor(evidence['class_weights'])).backward()
        ratio=logits.grad[0,:,0,1].abs().sum()/logits.grad[0,:,0,0].abs().sum()
        self.assertGreater(float(ratio),5.)
        self.assertTrue(bool(logits.grad.isfinite().all()))

    def test_corruption_realization_is_fixed_when_training_seed_changes(self):
        for variant in ['clean','boundary','wrong']:
            dataset=seg.Crops([self.row],variant)
            np.random.seed(43);torch.manual_seed(43);_,first=dataset[0]
            np.random.seed(44);torch.manual_seed(44);_,second=dataset[0]
            self.assertTrue(torch.equal(first,second))
            if variant=='clean':self.assertEqual(int((first==7).sum()),25)


if __name__=='__main__':unittest.main()
