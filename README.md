# IAN-pytorch

This is a pytorch implementation for [Interactive Attention Networks for Aspect-Level Sentiment Classification][1] (Dehong Ma et al, IJCAI 2017).

## requirements

```
pytorch 1.0.0
numpy
spacy
```

## Quick Start

- Download the 300-dimensional pre-trained word vectors from [Glove][2] and save it in the 'data' folder as 'data/glove.840B.300d.txt'

- Use the following command to run the code.

```
python main.py
```

## Results

| Dataset    | Accuracy |
| ---------- | -------- |
| Laptop     | 70.376   |
| Restaurant | 77.679   |

## Reference

During the procedure of implementation, the code in [lpq29743/IAN][3] is referenced.

[1]:https://arxiv.org/abs/1709.00893
[2]:https://nlp.stanford.edu/projects/glove/
[3]:https://github.com/lpq29743/IAN