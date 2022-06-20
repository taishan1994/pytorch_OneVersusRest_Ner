# pytorch_OneVersusRest_Ner
延申：

- 一种级联Bert用于命名实体识别，解决标签过多问题：https://github.com/taishan1994/pytorch_Cascade_Bert_Ner
- 一种多头选择Bert用于命名实体识别：https://github.com/taishan1994/pytorch_Multi_Head_Selection_Ner
- 中文命名实体识别最新进展：https://github.com/taishan1994/awesome-chinese-ner
- 信息抽取三剑客：实体抽取、关系抽取、事件抽取：https://github.com/taishan1994/chinese_information_extraction
- 一种基于机器阅读理解的命名实体识别：https://github.com/taishan1994/BERT_MRC_NER_chinese
- W2NER：命名实体识别最新sota：https://github.com/taishan1994/W2NER_predict

****

基于pytorch的one vs rest中文命名实体识别。

这里的one vs rest是指进行实体识别时，每次只识别出其中的一类，因此这里针对于每一类都有一个条件随机场计算损失，最终的损失是将每一类的损失都进行相加。这样做可以解决一个实体可能属于不同的类型问题。同时需要注意的是在解码的时候没有使用维特比解码，而是直接用np.argmax进行选择。这有点类似于基于机器阅读理解的实体识别，只不过这里没有提供问题，而且对于所有的类别只需要编码一次。

# 说明

这里是以程序中的cner数据为例，其余两个数据集需要自己按照模板进行修改尝试，数据地址参考：[基于pytorch的bert_bilstm_crf中文命名实体识别 (github.com)](https://github.com/taishan1994/pytorch_bert_bilstm_crf_ner)。如何修改：

- 1、在raw_data下是原始数据，新建一个process.py处理数据得到mid_data下的数据。
- 2、运行preprocess_mc.py，得到final_data下的数据。具体相关的数据格式可以参考cner。
- 3、运行指令进行训练、验证和测试。

数据及训练好的模型下载：链接：https://pan.baidu.com/s/1sJsat-bksQH8PCBVZQOHKA?pwd=6udv  提取码：6udv

# 依赖

```
pytorch==1.6.0
tensorboasX
seqeval
pytorch-crf==0.7.2
transformers==4.4.0
```

# 运行

```python
!python main.py \
--bert_dir="model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/cner/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=8 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=150 \
--lr=3e-5 \
--crf_lr=3e-2 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=10 \
--eval_batch_size=16 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--dropout_prob=0.3 \
--dropout=0.3 \

```

### 结果

基于bert_crf：

```python
precision:0.9113 recall:0.8919 micro_f1:0.9015
          precision    recall  f1-score   support

    RACE       0.88      1.00      0.94        15
     LOC       0.00      0.00      0.00         2
    CONT       1.00      1.00      1.00        33
     EDU       0.87      0.97      0.92       109
     PRO       0.86      0.95      0.90        19
    NAME       0.99      1.00      1.00       110
     ORG       0.90      0.89      0.90       543
   TITLE       0.91      0.86      0.88       770

micro-f1       0.91      0.89      0.90      1601

虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。
{'RACE': [('汉族', 17)], 'CONT': [('中国国籍', 20)], 'EDU': [('MBA', 45)], 'NAME': [('虞兔良', 0)], 'TITLE': [('中共党员', 40), ('经济师', 49)]}
```

# 补充

还可以进一步的进行修改，比如参考：https://github.com/taishan1994/pytorch_Cascade_Bert_Ner 设计一个多标签分类的任务来判别句子中是否存在某类实体。又或者是参考机器阅读理解，将标签的信息融合到具体的某类实体中。而且，我们可以根据需要去针对于某类实体进行调整，以达到相对好的效果，比如增加一些数据、使用其它的损失等等。
