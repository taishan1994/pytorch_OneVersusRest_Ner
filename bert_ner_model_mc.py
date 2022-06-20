import torch
import torch.nn as nn
from bert_base_model import BaseModel
from torchcrf import CRF
import config


class BertNerModel(BaseModel):
    def __init__(self,
                 args,
                 **kwargs):
        super(BertNerModel, self).__init__(bert_dir=args.bert_dir, dropout_prob=args.dropout_prob)
        self.args = args
        self.num_layers = args.num_layers
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device

        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 256)
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(args.dropout))
        #
        out_dims = mid_linear_dims

        # self.dropout = nn.Dropout(dropout_prob)
        # 有多少个类别，这里就定义多少个分类器
        self.classifier_list = nn.ModuleList([
          nn.Linear(out_dims, 3) for _ in range(args.num_tags) 
        ])

        # self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.criterion = nn.CrossEntropyLoss()


        init_blocks = [self.mid_linear] + [classifier for classifier in self.classifier_list]
        # init_blocks = [self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

        # 0,1,2
        self.crf_list = nn.ModuleList([CRF(3, batch_first=True) for _ in range(args.num_tags)])


    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                labels):
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]  # [batchsize, max_len, 768]
        batch_size = seq_out.size(0)
        seq_len = seq_out.size(1)


        seq_out = self.mid_linear(seq_out)  # [batchsize, max_len, 256]
        # seq_out = self.dropout(seq_out)
        seq_out_list = [classifier(seq_out) for classifier in self.classifier_list]
        # print(len(seq_out_list))
        
        if labels is None:
            return seq_out_list
        sep_labels = []
        for i in range(self.args.num_tags):
          sep_labels.append(labels[:, i, :])
        loss_list = [-crf(seq_out, label.squeeze().reshape(batch_size, seq_len), mask=attention_masks, reduction='mean') for seq_out, label, crf in zip(seq_out_list, sep_labels, self.crf_list)]
        loss = sum(loss_list) / len(loss_list)
        return loss, seq_out_list


if __name__ == '__main__':
    args = config.Args().get_parser()
    args.num_tags = 8
    model = BertNerModel(args)
    for name,weight in model.named_parameters():
        print(name)