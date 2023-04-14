### GPT

This is the simplest natural language processing model (2.5B). 
The entire project includes distributed training (DDP) and inference.
The training method is to predict the next word.
Please make sure that you have installed a CUDA environment with pytorch >=2.0. The specific usage instructions are as follows:

#### Setup

```bash
pip install -r requirements.txt
cd ext; python setup.py install; cd ..
python check_ext.py
```

#### Prepare the basic dataset

Currently, we are using approximately 6GB of Chinese and English datasets:
https://huggingface.co/datasets/xytpai/chatdataset

#### Training

```bash
# Change to your dataset dir (containing *.txtl)
export DATA_DIR=/data/chatdataset

# If you want to re-generate vocab.json
# python tokenization.py --dir=${DATA_DIR}

# base training cmd
python train_sft.py --model=nano --batch_size=1 --data=${DATA_DIR} --end=5000000 --gradient_accumulation_steps=16

# if you want to use tensorboard
tensorboard --logdir=summary
```

#### Predicting Demo

```bash
python eval.py --model=nano --text="Hello"
```
