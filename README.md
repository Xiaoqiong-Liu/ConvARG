## Welcome
### This repository is the first open code for ARG(anti resistance gene) detection, it is inspired by HMD-ARG for my best knowledge. And this revised version has higher validation accuracy than HMD-ARG.

## Envrionement
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

* Validation
```
python test.py

```

* Train
```
python train.py

```