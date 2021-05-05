import tarfile
import os
import json
import requests
import sys
import shutil
import re
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib
import time
from datetime import datetime
import csv
import argparse
import random

# if in Google Colaboratory
try:
    from google.colab import drive
except:
    pass

from models import model, encoder
from data.load_dataset import process_training_data
from models.accumulate import AccumulatingOptimizer

assert tf.__version__ < '2.0.0', "gpt-2-simple currently does not support " \
    "TensorFlow 2.0. You'll need to use a virtualenv/cloud computer which " \
    "has Tensorflow 1.X on it."


def start_tf_sess(threads=-1, server=None):
    """
    Returns a tf.Session w/ config
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    if threads > 0:
        config.intra_op_parallelism_threads = threads
        config.inter_op_parallelism_threads = threads

    if server is not None:
        return tf.compat.v1.Session(target=server.target, config=config)
    
    return tf.compat.v1.Session(config=config)


def reset_session(sess, threads=-1, server=None):
    """Resets the current TensorFlow session, to clear memory
    or load another model.
    """

    tf.compat.v1.reset_default_graph()
    sess.close()
    sess = start_tf_sess(threads, server)
    return sess

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def finetune(sess,
             dataset,
             steps=-1,
             model_name='124M',
             model_dir='models',
             batch_size=1,
             learning_rate=0.0001,
             accumulate_gradients=5,
             input_maxlen=100,
             history_len=4,
             restore_from='latest',
             run_name='run1',
             checkpoint_dir='checkpoint',
             multi_gpu=False,
             save_every=1000,
             print_every=1,
             max_checkpoints=1,
             optimizer='adam',
             overwrite=False):
    """Finetunes the model on the given dataset.
    """

    # assert model_name not in ['774M', '1558M'] or multi_gpu, "Currently, a modern single GPU cannot finetune the 774M GPT-2 model or larger."

    #SAMPLE_DIR = 'samples'

    checkpoint_path = os.path.join(checkpoint_dir, run_name)

    def maketree(path):
        try:
            os.makedirs(path)
        except:
            pass

    maketree(checkpoint_path)
    files = [f for f in os.listdir(checkpoint_path)]
    for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
        try:
            shutil.copyfile(os.path.join(model_dir, model_name, file),
                            os.path.join(checkpoint_path, file))
        except FileNotFoundError as fnf_error:
            print("You need to download the GPT-2 model first via download_gpt2()")
            raise(fnf_error)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    gpus = []

    if multi_gpu:
        gpus = get_available_gpus()

    output = model.model(hparams=hparams, X=context, gpus=gpus)
    loss = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    all_vars = [v for v in tf.compat.v1.trainable_variables() if 'model' in v.name]
    train_vars = all_vars

    if optimizer == 'adam':
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

    opt = AccumulatingOptimizer(
          opt=opt,
          var_list=train_vars)
    opt_reset = opt.reset()
    opt_compute = opt.compute_gradients(loss)
    opt_apply = opt.apply_gradients()
    summary_loss = tf.compat.v1.summary.scalar('loss', opt_apply)

    summary_log = tf.compat.v1.summary.FileWriter(checkpoint_path)

    saver = tf.compat.v1.train.Saver(
        var_list=all_vars,
        max_to_keep=max_checkpoints)
    
    sess.run(tf.compat.v1.global_variables_initializer())

    if restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join(model_dir, model_name))
    elif restore_from == 'fresh':
        ckpt = tf.train.latest_checkpoint(
            os.path.join(model_dir, model_name))
    else:
        ckpt = tf.train.latest_checkpoint(restore_from)
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    print('Loading dataset...')
    data = process_training_data(dataset, enc, maxlen=input_maxlen, history_len=history_len, reactonly=False)
    print('dataset has', len(data), 'training samples')
    print('Training...')

    counter = 1
    counter_path = os.path.join(checkpoint_path, 'counter')
    if os.path.exists(counter_path) and restore_from == 'latest':
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1
    counter_base = counter

    def save():
        maketree(checkpoint_path)
        print(
            'Saving',
            os.path.join(checkpoint_path,
                         'model-{}').format(counter-1))
        saver.save(
            sess,
            os.path.join(checkpoint_path, 'model'),
            global_step=counter-1)
        with open(counter_path, 'w') as fp:
            fp.write(str(counter-1) + '\n')

    def sample_batch():
        return random.sample(data, batch_size)

    if overwrite and restore_from == 'latest':
        for file in files:
            if file.startswith('model') or file.startswith('events'):
                os.remove(os.path.join(checkpoint_path, file))
        save()

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    if steps:
        steps = int(steps)
    
    try:
        while True:
            if steps > 0 and counter == (counter_base + steps):
                save()
                return
            if (counter - 1) % save_every == 0 and counter > 1:
                save()

            sess.run(opt_reset)
            for _ in range(accumulate_gradients):
                sess.run(
                   opt_compute, feed_dict={context: sample_batch()})
            (v_loss, v_summary) = sess.run((opt_apply, summary_loss))

            summary_log.add_summary(v_summary, counter)

            if counter % print_every == 0:
                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

            counter += 1
    except KeyboardInterrupt:
        print('interrupted')
        save()

