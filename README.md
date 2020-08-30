# Recurent Color Constancy

Y Qian, K Chen, J Nikkanen, JK Kamarainen, J Matas 

ICCV 2017

# A Benchmark of Burst Color Constancy

Y Qian, J Käpylä, JK Kämäräinen, S Koskinen, J Matas

ECCV-W 2020



This implementation uses [Pytorch](http://pytorch.org/).

## Installation
Please install [Anaconda](https://www.anaconda.com/distribution/) firstly.

```shell
git clone https://github.com/yanlinqian/Temporal-Color-Constancy.git
cd Temporal-Color-Constancy
## Create python env with relevant packages
conda create --name Temporal-Color-Constancy python=3.6
source activate Temporal-Color-Constancy
pip install -U pip
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch  # cudatoolkit=10.0 for cuda10
```

Tested on pytorch >= 1.0 and python3.

## Download
### Dataset

-- will upload soon. this project relys on the bcc-benchmark. 

<!---
[*Shi's Re-processing of Gehler's Raw Dataset*:](http://www.cs.sfu.ca/~colour/data/shi_gehler/)

 - Download the 4 zip files from the website and unzip them 
 - Extract images in the `/cs/chroma/data/canon_dataset/586_dataset/png` directory into `./data/images/`, without creating subfolders.
 - Masking MCC chats: 
```shell
  bash ./data/run.sh
```
-->

<!---
### Pretrained models
* Pretrained models can be downloaded [here](https://1drv.ms/u/s!AkGWFI5PP7sYarUAuXBGR3leujQ?e=Klqeg0). To reproduce the results reported in the paper, the pretrained models(*.pth) should be placed in `./trained_models/`, and then test model directly
-->


## Run code
Open the visdom service
```shell
python -m visdom.server -p 8008

```
### Training
* Train the rcc-net:
```shell
python ./rcc_net/train_rccnet.sh
```
* Train the bcc-net
```shell
python ./rcc_net/train_bccnet.sh
```

### Testing



* To reproduce the results reported in the paper, move the pretrained models to `./trained_models/`, and then test model directly.
```shell
python ./test/test_rccnet.py --pth_path0 ./trained_models/rccnet/fold0.pth
python ./test/test_bccnet.py --pth_path0 ./trained_models/bccnet/fold0.pth
```