https://blog.csdn.net/rosefun96/article/details/78807609

## Scalar

运行程序时，出错，AttributeError: 'SummaryMetadata' object has no attribute 'display_name'
只有graph图像。
后来，发现这是TensorFlow版本问题。

由于，之前装的GPU版本是tensorflow (1.3.0rc0),但是运行tensorboard的时候，没有出现scalar，然后试了升级TensorFlow版本，成功解决问题。
在anaconda prompt运行

pip install --ignore-installed --upgrade tensorflow-gpu
