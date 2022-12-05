#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import json
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from tensorflow.keras import metrics
from tabulate import tabulate
import matplotlib.pyplot as plt
from scaaml.aes import ap_preds_to_key_preds
from scaaml.plot import plot_trace, plot_confusion_matrix
from scaaml.utils import tf_cap_memory, from_categorical
from scaaml.model import get_models_by_attack_point, get_models_list, load_model_from_disk
from scaaml.intro.generator import list_shards, load_attack_shard
from scaaml.utils import hex_display, bytelist_to_hex

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# # Environment setup

# In[3]:


target = 'stm32f415_tinyaes'
tf_cap_memory()
target_config = json.loads(open("config/" + target + '.json').read())
BATCH_SIZE = target_config['batch_size']
TRACE_LEN = target_config['max_trace_len']


# ## Available models

# In[4]:


available_models = get_models_by_attack_point(target_config)


# ## Dataset paths

# In[7]:


DATASET_GLOB = "datasets/%s/test/*" % target_config['algorithm']
shard_paths  = list_shards(DATASET_GLOB, 256)


# # Single byte recovery
# 
# Let's start by recovering one of the 16 bytes to get a sense of how well the attack work.
# 
# **Note**: looking at model accuracy is not enough as we need to invert the attack point AND combine the traces scores. See [presentation](https://elie.net/scaaml) for more information

# In[8]:


# let's select an attack point that have all the needed models -- Key is not a good target: it doesn't work
ATTACK_POINT = 'sub_bytes_out'

# let's also pick the key byte we want to use SCAAML to recover and load the related model
ATTACK_BYTE = 7

# load model
model = load_model_from_disk(available_models[ATTACK_POINT][ATTACK_BYTE])


# Using our model to predicting bytes attack point value, recovering byte key and combining prediction for the 256 test keys we have in our dataset

# In[9]:


NUM_TRACES = 10  # maximum number of traces to use to recover a given key byte. 10 is already overkill
correct_prediction_rank = defaultdict(list)
y_pred = []
y_true = []
model_metrics = {"acc": metrics.Accuracy()}
for shard in tqdm(shard_paths, desc='Recovering bytes', unit='shards'):
    keys, pts, x, y = load_attack_shard(shard, ATTACK_BYTE, ATTACK_POINT, TRACE_LEN, num_traces=NUM_TRACES)

    # prediction
    predictions = model.predict(x)
    
    # computing byte prediction from intermediate predictions
    key_preds = ap_preds_to_key_preds(predictions, pts, ATTACK_POINT)
    
    c_preds = from_categorical(predictions)
    c_y = from_categorical(y)
    # metric tracking
    for metric in model_metrics.values():
        metric.update_state(c_y, c_preds)
    # for the confusion matrix
    y_pred.extend(c_preds)
    y_true.extend(c_y)
    
    # accumulating probabilities and checking correct guess position.
    # if all goes well it will be at position 0 (highest probability)
    # see below on how to use for the real attack
    
    
    key = keys[0] # all the same in the same shard - not used in real attack
    vals = np.zeros((256))
    for trace_count, kp in enumerate(key_preds):
        vals = vals  + np.log10(kp + 1e-22) 
        guess_ranks = (np.argsort(vals, )[-256:][::-1])
        byte_rank = list(guess_ranks).index(key)
        correct_prediction_rank[trace_count].append(byte_rank)


# ## checking model accuracy & confusion matrix
# the model accuracy is very good given that the baseline is 1/256 > 0.04. However accuracy is not enough to assess of 
# effective the attack is. Instead you need to look at how many traces it takes to recovers 100% of the sub_bytes as visible below.
# 
# Looking at the confusion matrix shows that the model perform equaly well on all the predicted value and generalized well: we see a crisp diagonal with no obvious weak points

# In[10]:


print("Accuracy: %.2f" % model_metrics['acc'].result())


# In[11]:


plot_confusion_matrix(y_true, y_pred, normalize=True, title="%s byte %s prediction confusion matrix" % (ATTACK_POINT, ATTACK_BYTE))


# ## byte recovery efficiency
# 

# In[12]:


NUM_TRACES_TO_PLOT = 10
avg_preds = np.array([correct_prediction_rank[i].count(0) for i in range(NUM_TRACES_TO_PLOT)])
y = avg_preds / len(correct_prediction_rank[0]) * 100 
x = [i + 1 for i in range(NUM_TRACES_TO_PLOT)]
plt.plot(x, y)
plt.xlabel("Num traces")
plt.ylabel("Recovery success rate in %")
plt.title("%s ap:%s byte:%s recovery performance" % (target_config['algorithm'], ATTACK_POINT, ATTACK_BYTE))
plt.show()


# ## metric computations
# 
# Let's look at some of the various performances metrics that are used to evaluate attack efficency.
# - In the worst case for the implementation the attacker can recover ~40% of the key with a single trace.
# - The best case is not really better: 4 traces is all you need.

# In[13]:


min_traces = 0
max_traces = 0
cumulative_aa = 0
for idx, val in enumerate(y):
    cumulative_aa += val
    if not min_traces and val > 0:
        min_traces = idx + 1
    if not max_traces and val == 100.0:
        max_traces = idx + 1
        break 

cumulative_aa = round( cumulative_aa / (idx + 1), 2) # divide by the number of steps

rows = [
    ["min traces", min_traces, round(y[min_traces -1 ], 1)],
    ["max traces", max_traces, round(y[max_traces - 1], 1)],
    ["cumulative score", cumulative_aa, '-'] 
]
print(tabulate(rows, headers=['metric', 'num traces', '% of keys']))


# # recover the full keys

# In[14]:


ATTACK_POINT = 'sub_bytes_out' # let's pick an attack point- Key is not a good target: it doesn't work for TinyAEs
TARGET_SHARD = 42 # a shard == a different key. Pick the one you would like
NUM_TRACES = 5  # how many traces to use - as seen in single byte, 5 traces is enough


# In[15]:


# perfoming 16x the byte recovery algorithm showecased above - one for each key byte
real_key = [] # what we are supposed to find
recovered_key = [] # what we predicted
pb = tqdm(total=16, desc="guessing key", unit='guesses')
for ATTACK_BYTE in range(16):
    # data
    keys, pts, x, y = load_attack_shard(shard_paths[TARGET_SHARD], ATTACK_BYTE, ATTACK_POINT, TRACE_LEN, num_traces=NUM_TRACES, full_key=True)
    real_key.append(keys[0])
    
    # load model
    model = load_model_from_disk(available_models[ATTACK_POINT][ATTACK_BYTE])
    
    # prediction
    predictions = model.predict(x)
    
    # computing byte prediction from intermediate predictions
    key_preds = ap_preds_to_key_preds(predictions, pts, ATTACK_POINT)
    
    # accumulating probabity
    vals = np.zeros((256))
    for trace_count, kp in enumerate(key_preds):
        vals = vals  + np.log10(kp + 1e-22)
    
    # order predictions by probability
    guess_ranks = (np.argsort(vals, )[-256:][::-1])
    
    # take strongest guess as our key guess
    recovered_key.append(guess_ranks[0])
    
    # update display
    pb.set_postfix({'Recovered key': bytelist_to_hex(recovered_key), "Real key": bytelist_to_hex(real_key)})
    pb.update()
    
    
pb.close()


# In[16]:


# check that everything worked out: the recovered key match the real keys
hex_display(real_key, 'real key')
hex_display(recovered_key, 'recovered key')


# # Congratulation
# you were able to use deep learning to recover an AES key with only a few traces.
# For more information check out [papers/tutorials](https://elie.net/scaaml/)
