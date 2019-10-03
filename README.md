# Pointer Networks in PyTorch

A minimal PyTorch implementation of Pointer Networks.

Supported features:
- Mini-batch training with CUDA
- Lookup, CNNs, RNNs and/or self-attentive encoding in the embedding layer
- Vectorized computation of alignment scores in the attention layer
- Beam search decoding

## Usage

Training data should be formatted as below:
```
source_sequence \t target_sequence
source_sequence \t target_sequence
...
```

To prepare data:
```
python3 prepare.py training_data
```

To train:
```
python3 train.py model vocab training_data.csv (validation_data) num_epoch
```

To predict:
```
python3 predict.py model.epochN vocab test_data
```

To evaluate:
```
python3 evaluate.py model.epochN vocab test_data
```

## References

Jing Li, Aixin Sun, Shafiq Joty. 2018. [SEGBOT: A Generic Neural Text Segmentation Model with Pointer Network.](https://www.ijcai.org/proceedings/2018/0579.pdf) In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence.

Xuezhe Ma, Zecong Hu, Jingzhou Liu, Nanyun Peng, Graham Neubig, Eduard Hovy. 2018. [Stack-Pointer Networks for Dependency Parsing.](https://aclweb.org/anthology/P18-1130) In ACL.

Abigail See, Peter J. Liu, Christopher D. Manning. 2017. [Get To The Point: Summarization with Pointer-Generator Networks.](https://arxiv.org/abs/1704.04368) arXiv:1704.04368.

Oriol Vinyals, Meire Fortunato, Navdeep Jaitly. 2015. [Pointer Networks.](https://arxiv.org/abs/1506.03134) In NIPS.

Oriol Vinyals, Samy Bengio, Manjunath Kudlur. 2015. [Order Matters: Sequence to sequence for sets.](https://arxiv.org/abs/1511.06391) In ICLR.

Feifei Zhai, Saloni Potdar, Bing Xiang, Bowen Zhou. 2017. [Neural Models for Sequence Chunking.](https://arxiv.org/abs/1701.04027) In AAAI.
