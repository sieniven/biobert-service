from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib2to3.pytree import convert

import os
import yaml
import pickle
import logging
import collections

import biobert.modeling
import biobert.tf_metrics
import biobert.optimization
import biobert.tokenization

import tensorflow as tf
from tensorflow.python.ops import math_ops

with open('./config.yaml', "r") as f:
    config = yaml.safe_load(f)

# initialize tf flags
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", "NER", "The name of the task to train.")
flags.DEFINE_string("output_dir", config["output_dir"], "")
flags.DEFINE_string("bert_config_file", config["bert_config_file"], "")
flags.DEFINE_string("vocab_file", config["vocab_file"], "")
flags.DEFINE_string("init_checkpoint", config["init_checkpoint"], "")
flags.DEFINE_string("master", None, "")

flags.DEFINE_bool("use_tpu", config["use_tpu"], "")
flags.DEFINE_bool("do_predict", config["do_predict"], "")
flags.DEFINE_bool("do_lower_case", config["do_lower_case"], "")

flags.DEFINE_integer("num_tpu_cores", 8, "")
flags.DEFINE_integer("predict_batch_size", config["predict_batch_size"], "")
flags.DEFINE_integer("max_seq_length", config["max_seq_length"], "")
flags.DEFINE_integer("iterations_per_loop", config["iterations_per_loop"], "")
flags.DEFINE_integer("save_checkpoints_steps", config["save_checkpoints_steps"], "")

"""
A single training/test example for simple sequence classification.
1. guid: Unique id for the example.
2. text: string. The untokenized text of the first sequence.
"""
class InputData(object):
    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label

"""
Representation for a single set of features of an input data entry
"""
class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

"""
BioBert class to predict named entities present in BioBertModel class. 
"""
class Biobert(object):
    def __init__(self, input = None):
        self.input_title = []
        self.input_abstract = []
        self.output_title = []
        self.output_abstract = []
        self.logger = logging.getLogger(__name__)

        self.max_seq_length = FLAGS.max_seq_length
        self.use_tpu = FLAGS.use_tpu
        self.use_one_hot_embeddings = FLAGS.use_tpu
        self.init_checkpoint = FLAGS.init_checkpoint

        self.label_list = self.get_labels()
        self.num_labels = len(self.label_list)
        self.label_map = {}
        self.id_to_label_map = {}
        for idx, label in enumerate(self.label_list):
            self.label_map[label] = idx
            self.id_to_label_map[idx] = label
      
        with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(self.label_map, w)

        self.tokenizer = biobert.tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        self.bert_config = biobert.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

        if (input):
            self.load_data(input)
        
        # run prediction for input data
        self.predict()
    """
    method to get list of labels
    """
    def get_labels(self):
        return ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]

    """
    method to create inputs for BioBert model
    """
    def load_data(self, input_data_list):
        for index, data in enumerate(input_data_list):
            text_title = biobert.tokenization.convert_to_unicode(data.article_data["title"])
            text_abstract = biobert.tokenization.convert_to_unicode(data.article_data["abstract"])
            label = biobert.tokenization.convert_to_unicode("O")
            self.input_title.append(InputData(index, text_title, label))
            self.input_abstract.append(InputData(index, text_abstract, label))
    
    """
    method to write tokens to output file
    """
    def write_tokens(self, tokens, mode):
        path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
        if os.path.exists(path):
            os.remove(path)
        
        with open(path, 'a', encoding="utf-8") as wf:
            for token in tokens:
                if token!="[PAD]":
                    wf.write(token+'\n')
            wf.close()

    def convert_data_entry(self, input_data, mode):
        text_list = input_data.text.split()
        label_list = input_data.label.split()
        tokens = []
        labels = []

        for word in text_list:
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_list[0]
            for m,  tok in enumerate(token):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")

        # drop if token is longer than max_seq_length
        if len(tokens) >= self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 2)]
            labels = labels[0:(self.max_seq_length - 2)]
        
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[CLS]"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(self.label_map[labels[i]])
        
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[SEP]"])

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("[PAD]")

        assert(len(input_ids)) == self.max_seq_length
        assert(len(input_mask)) == self.max_seq_length
        assert(len(segment_ids)) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length

        # log out examples before model starts running
        if input_data.guid < 4:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid %s" % input_data.guid)
            tf.logging.info("tokens: %s" % " ".join([biobert.tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        self.write_tokens(ntokens, mode)
        features = InputFeatures(input_ids, input_mask, segment_ids, label_ids)
        return features

    """
    method to convert input data to features
    """
    def file_based_convert_data_entries_to_features(self, data_entries, output_file, mode):
        writer = tf.python_io.TFRecordWriter(output_file)
        for data in data_entries:
            if (data.guid % 5000 == 0):
                tf.logging.info(f"Writing example {data.guid} of {len(data_entries)}")
            feature = self.convert_data_entry(data, mode)
            
            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

    def file_based_input_fn_builder(self, input_file, drop_remainder):
        name_to_features = {
        "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64)
        }

        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            return example

        def input_fn(params):
            batch_size = params["batch_size"]
            d = tf.data.TFRecordDataset(input_file)
            d = d.apply(tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size, drop_remainder=drop_remainder))
            return d
        return input_fn

    def create_model(self, input_ids, input_mask, segment_ids, labels):
        model = biobert.modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        output_layer = model.get_sequence_output()
        hidden_size = output_layer.shape[-1].value
        
        output_weight = tf.get_variable(
            "output_weights", [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        
        output_bias = tf.get_variable(
            "output_bias", [self.num_labels], initializer=tf.zeros_initializer())
        
        with tf.variable_scope("loss"):
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, self.max_seq_length, self.num_labels])
            # mask = tf.cast(input_mask,tf.float32)
            # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
            # return (loss, logits, predict)

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_sum(per_example_loss)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predict = {"predict": tf.argmax(probabilities,axis=-1), "log_probs": log_probs}
            return (loss, per_example_loss, logits, predict)

    def model_fn_builder(self):
        def model_fn(features, labels, mode, params):
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            (total_loss, per_example_loss, logits, predictsDict) = self.create_model(input_ids, input_mask, segment_ids, label_ids)
            predictsDict["input_mask"] = input_mask
            tvars = tf.trainable_variables()
            scaffold_fn = None
            (assignment_map, initialized_variable_names) = biobert.modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
            if self.use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
            
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
            output_spec = None
            if mode == tf.estimator.ModeKeys.PREDICT:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode = mode, predictions = predictsDict, scaffold_fn = scaffold_fn)
            return output_spec
        return model_fn

    def predict(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.gfile.MakeDirs(FLAGS.output_dir)

        if self.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(f"Cannot use sequence length {self.max_seq_length} because the BERT model "
                "was only trained up to sequence length {self.bert_config.max_position_embeddings}")

        tpu_cluster_resolver = None
        if self.use_tpu and FLAGS.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver, master=FLAGS.master,
            model_dir=FLAGS.output_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores, per_host_input_for_training=is_per_host))

        model_fn = self.model_fn_builder()
        estimator = tf.contrib.tpu.TPUEstimator(use_tpu=self.use_tpu, model_fn=model_fn,
            config=run_config, predict_batch_size=FLAGS.predict_batch_size)

        # predict input titles
        modes = ["title", "abstract"]

        for mode in modes:
            predict_file = os.path.join(FLAGS.output_dir, "predict_"+mode+".tf_record")

            if (mode == "title"):
                self.file_based_convert_data_entries_to_features(self.input_title, predict_file, mode)
            else:
                self.file_based_convert_data_entries_to_features(self.input_abstract, predict_file, mode)

            tf.logging.info("***** Running prediction for titles *****")
            tf.logging.info("  Number of entries = %d", len(self.input_title))
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
            tf.logging.info("  Example of predict_examples = %s", self.input_title[0].text)

            if self.use_tpu:
                raise ValueError("Prediction in TPU not supported")

            predict_drop_remainder = True if self.use_tpu else False
            predict_input_fn = self.file_based_input_fn_builder(predict_file, drop_remainder=predict_drop_remainder)
            result = estimator.predict(input_fn=predict_input_fn)
            
            output_predict_file = os.path.join(FLAGS.output_dir, "label_"+mode+".txt")
            with open(output_predict_file,'w') as writer:
                for resultIdx, prediction in enumerate(result):
                    # Fix for "padding occurrence amid sentence" error
                    # (which occasionally cause mismatch between the number of predicted tokens and labels.)
                    assert len(prediction["predict"]) == len(prediction["input_mask"]), "len(prediction['predict']) != len(prediction['input_mask']) Please report us!"
                    predLabelSent = []
                    for predLabel, inputMask in zip(prediction["predict"], prediction["input_mask"]):
                        # predLabel : Numerical Value
                        if inputMask != 0:
                            if predLabel == self.label_map['[PAD]']:
                                predLabelSent.append('O')
                            else:
                                predLabelSent.append(self.id_to_label_map[predLabel])

                    if (mode == "title"):
                        self.output_title.append(predLabelSent)
                    else:
                        self.output_abstract.append(predLabelSent)

                    output_line = "\n".join(predLabelSent) + "\n"
                    writer.write(output_line)