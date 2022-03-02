import os
import json
import pickle
import tf_metrics
import collections
import tensorflow as tf
from tensorflow.python.ops import math_ops
from datetime import datetime, timedelta

import modeling
import optimization
import tokenization

with open('../gtt_docker/ingest/ingestDataset.json') as f:
    data = json.loads("[" + f.read().replace("}{", "},{") +  "]")

print(data[0]["article_data"]["abstract"])