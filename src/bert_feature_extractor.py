
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer

import os
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from lm_finetuning.simple_lm_finetuning import _truncate_seq_pair
import numpy as np
import logging

"""
Feature Extractor for BERT
"""

cn_output_dir = "../finetune_llm/fine_tuning_ab/"
# cn_output_dir = "../bert_model_embeddings/nodes-lm-conceptnet/"

fb_output_dir = "../finetune_llm/fine_tuning_margin_16/"

model_gpu = 4

class InputExample(object):
    """A single training/test example for simple sequence classification with BERT."""

    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, max_seq_length, tokenizer, label_list=None):
    """Loads a data file into a list of `InputBatch`s."""

    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.label:
            label_id = label_map[example.label]
        else:
            label_id = None

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return (" ".join([m.group(0) for m in matches])).lower()


def convert_edges_to_examples(edges, labels, network):
    examples = []

    for i, edge in enumerate(edges):
        edge = edge.cpu().numpy()
        text_a = network.graph.nodes[edge[0]].name + " " + camel_case_split(network.graph.relations[edge[1]].name) + " " + network.graph.nodes[edge[2]].name
        label = labels[i].cpu().item()
        examples.append(
            InputExample(text_a=text_a, text_b=None, label=label))

    return examples


def convert_nodes_to_examples(dataset, node_list):       # 将node->text
    examples = []
    ent2text = change_entity_to_text(dataset)
    for node in node_list:
        if (dataset == "fb15k237") | (dataset =="fb15k"):

            text_a = ent2text[node.name]
        else:
            text_a = node.name
        examples.append(
            InputExample(text_a=text_a))

    return examples

def change_entity_to_text(dataset):
    if (dataset == "fb15k237") | (dataset == "fb15k"):
        ent2text = {}
        with open('../data/conventionalKGs/FB15k-237/entity2text.txt', 'r', encoding='utf-8') as file:
            for line in file:
                relation = line.strip().split('\t')
                ent2text[relation[0]] = relation[1]
        return ent2text
    else:
        return 0


class BertLayer(nn.Module):
    def __init__(self, dataset):
        super(BertLayer, self).__init__()
        pretrained_model = "bert-large-uncased"       # 用于加载Tokenizer
        # print("aaaaaaaaaaaaaa")
        self.dataset = dataset
        logger = logging.getLogger()
        if self.dataset == "conceptnet":
            # output_dir = "../bert_model_embeddings/nodes-lm-conceptnet/"
            output_dir = cn_output_dir
        elif self.dataset == "atomic":
            # output_dir = "../bert_model_embeddings/nodes-lm-atomic/"
            output_dir = "../pretrained/log_prompt_atomic/"
        elif self.dataset == "fb15k237" or self.dataset == "FB15K-237":
            output_dir = fb_output_dir
        elif self.dataset == "fb15k":
            output_dir = "../pretrained/log_path_fb15k/"

        logging.info(output_dir)
        print(output_dir)

        self.filename = os.path.join(output_dir, self.dataset + "0_bert_embeddings_path.pt")
        # self.filename = os.path.join(output_dir, self.dataset + "_bert_embeddings.pt")
        print(self.filename)
 
        if os.path.isfile(self.filename):
            self.exists = True
            return

        self.exists = False
        self.max_seq_length = 32
        self.eval_batch_size = 128
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False)     # 加载Tokenizer
        output_model_file = os.path.join(output_dir, "lm_pytorch_model.bin")
        logger.info("Loading model from %s" % output_dir)
        self.pretrained_model = torch.load(output_model_file, map_location='cpu')     # 加载预训练模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.bert_model = BertLayer.load_state_dict(torch.load(output_model_file))

        self.pretrained_model.to(self.device)

        # Make BERT parameters non-trainable
        # bert_params = list(self.bert_model.parameters())
        # for param in bert_params:
        #     param.requires_grad = False

    def forward(self, node_list):

        # if self.exists:
        #     print("Loading BERT embeddings from disk..")
        #     return torch.load(self.filename, map_location=lambda storage, loc: storage.cuda(4))        # 加载已有模型
        #     # return torch.load(self.filename, map_location='cuda:1')

        if self.exists:
            print("Loading BERT embeddings from disk..")
            sequence_outputs = []
            if self.dataset == "conceptnet":
                # output_dir = "../bert_model_embeddings/nodes-lm-conceptnet/"
                output_dir = cn_output_dir
            elif self.dataset == "atomic":
                # output_dir = "../bert_model_embeddings/nodes-lm-atomic/"
                output_dir = "../pretrained/log_prompt_atomic/"
            elif self.dataset == "fb15k237" or self.dataset == "FB15K-237":
                output_dir = fb_output_dir
            elif self.dataset == "fb15k":
                output_dir = "../pretrained/log_path_fb15k/"
            for i in range(0, 1):
                # output_dir = "../pretrained/plm_roberta_large_margin_0.1/"
                self.filename = os.path.join(output_dir, self.dataset + str(i) + "_bert_embeddings_path.pt")
                sequence_output = torch.load(self.filename, map_location=lambda storage, loc: storage.cuda(model_gpu))
                sequence_outputs.append(sequence_output)
            return torch.cat(sequence_outputs, dim=0)

        print("Computing BERT embeddings..")
        self.pretrained_model.eval()

        eval_examples = convert_nodes_to_examples(self.dataset, node_list)
        eval_features = convert_examples_to_features(
            eval_examples, max_seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        sequence_outputs = []

        idx = 0
        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                sequence_output, _ = self.pretrained_model.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
            sequence_outputs.append(sequence_output[:, 0])

            if len(sequence_outputs) == 800:
                self.save_to_disk(torch.cat(sequence_outputs, dim=0), idx)
                sequence_outputs = []
                idx += 1
            
        self.save_to_disk(torch.cat(sequence_outputs, dim=0), idx)

        return torch.cat(sequence_outputs, dim=0)

    def forward_as_init(self, num_nodes, network=None):

        # if self.exists:
        #     print("Loading BERT embeddings from disk..")
        #     return torch.load(self.filename, map_location=lambda storage, loc: storage.cuda(4))
        # 如果有embedding这里直接返回
        if self.exists:
            print("Loading BERT embeddings from disk..")
            sequence_outputs = []
            if self.dataset == "conceptnet":
                # output_dir = "../bert_model_embeddings/nodes-lm-conceptnet/"
                output_dir = cn_output_dir
            elif self.dataset == "atomic":
                # output_dir = "../bert_model_embeddings/nodes-lm-atomic/"
                output_dir = "../pretrained/log_prompt_atomic/"
            elif self.dataset == "fb15k237" or self.dataset == "FB15K-237":
                output_dir = fb_output_dir
            elif self.dataset == "fb15k":
                output_dir = "../pretrained/log_path_fb15k/"
            for i in range(0, 1):
                # output_dir = "../pretrained/plm_roberta_large_margin_0.1/"
                self.filename = os.path.join(output_dir, self.dataset + str(i) + "_bert_embeddings_path.pt")
                sequence_output = torch.load(self.filename, map_location=lambda storage, loc: storage.cuda(model_gpu))
                sequence_outputs.append(sequence_output)
            return torch.cat(sequence_outputs, dim=0)

        node_ids = np.arange(num_nodes)
        node_list = [network.graph.nodes[idx] for idx in node_ids]      # 子图中所有节点

        print("Computing BERT embeddings..")
        self.pretrained_model.eval()

        eval_examples = convert_nodes_to_examples(node_list)
        eval_features = convert_examples_to_features(
            eval_examples, max_seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        sequence_outputs = []

        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                sequence_output, _ = self.pretrained_model.bert(input_ids, segment_ids, input_mask,
                                                          output_all_encoded_layers=False)
            sequence_outputs.append(sequence_output[:, 0])
            
        return torch.cat(sequence_outputs, dim=0)

    def save_to_disk(self, tensor, idx):
        torch.save(tensor, self.dataset + str(idx) + "_bert_embeddings.pt")
