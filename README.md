## CNN Pytorch实现 中文情感分类
## 论文

[1] Li S ,  Zhao Z ,  Hu R , et al. Analogical Reasoning on Chinese Morphological and Semantic Relations[C]，ACL 2018.
[2] Kim Y . Convolutional Neural Networks for Sentence Classification[J]. Eprint Arxiv, 2014.


## 参考
* https://github.com/yoonkim/CNN_sentence
* https://github.com/dennybritz/cnn-text-classification-tf
* https://github.com/Shawn1993/cnn-text-classification-pytorch

## 依赖项
* python3.8
* pytorch==1.8.0
* torchtext==0.9.0
* tensorboard==2.6.0
* jieba==0.42.1

## 词向量
https://github.com/Embedding/Chinese-Word-Vectors<br>
（这里用的是训练出来的word Word2vec）
## 用法
```bash
python3 main.py -h
```

## 训练
```bash
python3 main.py
```

## 准确率
-  CNN-multichannel 使用预训练的静态词向量微调词向量
    ```bash
python main.py -static=true -non-static=true -multichannel=true
    ```
## 训练 CNN 并将日志保存到 runs 文件夹,要查看日志，只需cmd下运行
tensorboard --logdir=runs 
