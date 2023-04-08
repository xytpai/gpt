### GPT

This is the simplest natural language processing model. 
The entire project includes distributed training (DDP) and inference.
The training method is to predict the next word.
Please make sure that you have installed a CUDA environment with pytorch >=2.0. The specific usage instructions are as follows:

#### Setup

```bash
pip install -r requirements.txt
```

#### Training

```bash
# Change to your dataset dir (containing .json)
export DATA_DIR=./minidata/

# If you want to re-generate vocab.json
# python tokenization.py --dir=${DATA_DIR}

# base training cmd
python train_level_zero.py --model=nano --batch_size=4 --data=${DATA_DIR} --end=10000000

# if you want to use tensorboard
tensorboard --logdir=summary
```

#### Predicting Demo

```bash
python eval.py --model=nano --text="Hello"
```
