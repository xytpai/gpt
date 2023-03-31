### GPT

This is the simplest natural language processing model. 
The entire project includes distributed training and inference.
The training method is to predict the next word.
Please make sure that you have installed a CUDA environment with pytorch >=1.12. The specific usage instructions are as follows:

#### Setup

```bash
pip install -r requirements.txt
```

#### Get dataset (Wiki)

```bash
rm -rf /data/wiki/ # Change to your own path
python ./data/wiki_downloader.py --language=en --save_path=/data/wiki/
python ./data/WikiExtractor.py /data/wiki/enwiki-20230101-pages-articles.xml -o /data/wiki/text
```

#### Training

```bash
python train.py       \
    --model=nano      \
    --batch_size=8    \
    --data=./minidata \
    --end=40000
```

#### Predicting

```bash
python eval.py --model=nano --text=Hello
```
