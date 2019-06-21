# CornerNet Caffe

This is just inference code of [Cornernet](https://arxiv.org/pdf/1808.01244v1.pdf) in caffe. And I implement corner pool and unsample layer by myself. Please tell me if there is something wrong.

The original pytorch implementation repository is [here](https://github.com/princeton-vl/CornerNet)

## How to use
### CC5.0
I use [cc5.0](https://github.com/dlunion/CC5.0) which is a good extended frame of caffe and support windows. You can watch the introduction by README

### PytorchToCaffe
I convert the original pytorch model to caffemodel by [PytorchToCaffe](https://github.com/xxradon/PytorchToCaffe). And I add some new feature in the pytorch_to_caffe.py which can find in my pytorch_to_caffe code.
~~However, the caffemodel I converted has some problems, so the output of the inference code is wrong, I'm still looking for the answer.~~

I have tested the converted model and code. And it's correct. The pytorch model I converted is the offical CornerNet_500000.pkl. The caffemodel and prototxt can be downloaded by [BaiduNetDisk](https://pan.baidu.com/s/1dPwNigOf0dhhbCWZUGe-aQ). The extract code is 34p9.(The public link is in maintenance).

### Why use my own relu layer
The weights of one conv layer in pytorch is too big which is over the boundary of float32. So if inputs times this weights, the result will be NaN. Howerver when the output of this conv layer crosses the caffe relu layer, NaN is still NaN while crossing the pytorch relu layer NaN will be zero.

**To do**
- [ ] Train code
