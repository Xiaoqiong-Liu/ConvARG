## Welcome
To the best of my knowledge, this is the first pytorch implementation for ARG(antibiotic resistance genes) detection, it is inspired by HMD-ARG. And this revised version has higher accuracy than HMD-ARG or Deep-ARG reported in <a href="https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-021-01002-3">[1]</a>.

## Environment
* Python == 3.8
* Download the repository
```
git clone https://github.com/Xiaoqiong-Liu/ConvARG.git
```

* Create a new environment
```
conda create -n arg python=3.8 # I use python 3.8 in my experiment
conda activate arg
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio cudatoolkit=10.2 -c pytorch -c conda-forge

pip install bio-datasets
conda install numpy
conda install pandas
```

## Test
* To reproduce the reported test accuracy(0.97+), you could train a new model or simply use the pretrained model under ./repoistory.
```
python test.py

```

## Train
* Run below command to train
```
python train.py

```

## Result
| Method            | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| ConvARG (Ours)    | **0.972**| **1.0**   | **0.98**| **0.99** |
| HMD-ARG           | 0.948    | 0.939     | 0.971  | 0.948    |
| DeepARG           | 0.965    | 0.998     | 0.93   | 0.963    |
| CARD              | 0.71     | 0.999     | 0.421  | 0.592    |

## Reference
[1] Li, Yu et al. “HMD-ARG: hierarchical multi-task deep learning for annotating antibiotic resistance genes.” Microbiome 9 (2021): n. pag.

