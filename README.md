### GPT

This is the simplest natural language processing model.
Our model backbone is fully compatible with llama.
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
export DATA_DIR=/data/chatdataset/merges

# If you want to re-generate vocab.json:
# mv vocab.json old_vocab.json
# python tokenization.py --dir=${DATA_DIR}
# export CKPT_FILE=checkpoints/nano_0001913600.ckpt # Change to your ckpt
# python migrate_wordemb.py --param=word_embeddings_56000.weight --old_vocab=old_vocab.json --ckpt=${CKPT_FILE}
# rm -rf ${CKPT_FILE} ; mv ${CKPT_FILE}.new ${CKPT_FILE} ; rm -rf old_vocab.json
# rm -rf .datacache

# base training cmd
python train_sft.py --model=nano --batch_size=4 --data=${DATA_DIR} --end=5000000 --gradient_accumulation_steps=4

# if you want to use tensorboard
tensorboard --logdir=summary
```

#### Predicting Demo

```bash
python eval.py --model=nano --text="你好"
```
