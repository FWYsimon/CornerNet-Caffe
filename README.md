# CornerNet Caffe

This is just inference code of [Cornernet](https://arxiv.org/pdf/1808.01244v1.pdf) in caffe. And I implement corner pool and unsample layer by myself. Please tell me if there is something wrong.

The original pytorch implementation repository is [here](https://github.com/princeton-vl/CornerNet)

## How to use
### CC5.0
I use [cc5.0](https://github.com/dlunion/CC5.0) which is a good extend frame of caffe and support windows. You can watch the introduction by README

### PytorchToCaffe
I convert the original pytorch model to caffemodel by [PytorchToCaffe](https://github.com/xxradon/PytorchToCaffe). And I add some new feature in the pytorch_to_caffe.py. Just look in my pytorch_to_caffe code.
However, the caffemodel I converted has some problems, so the output of the inference code is wrong, I'm still looking for the answer.

**To do**
- [ ] Train code