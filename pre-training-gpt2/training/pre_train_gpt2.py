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

from models import model, encodings, sample
from data.load_dataset import process_training_data, Batch_Sampler
from data.sample_query import sample_query
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


def finetune(sess,
             train_dataset,
             val_dataset,
             steps=-1,
             model_name='124M',
             model_dir='models',
             batch_size=1,
             learning_rate=0.0001,
             accumulate_gradients=5,
             context_maxlen=100,
             response_maxlen=100,
             history_len=4,
             patience=20,
             restore_from='latest',
             run_name='run1',
             checkpoint_dir='checkpoint',
             multi_gpu=False,
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

    enc = encodings.Encoding("main", n_vocab=50257, eot_token=50256, base_path=checkpoint_path).get_encoder()
    hparams = model.HParams()
    hparams.override_from_json_file(os.path.join(checkpoint_path, 'hparams.json'))

    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])

    lm_model = model.Model(hparams=hparams)

    output = lm_model(X=context, padding_token=enc.padding_token)
    
    train_vars = lm_model.get_params()

    if optimizer == 'adam':
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

    if accumulate_gradients > 1:
        opt = AccumulatingOptimizer(
            opt=opt,
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(output['lm_losses'])
        opt_apply = opt.apply_gradients()
        summary_loss = tf.compat.v1.summary.scalar('loss', opt_apply)
    
    else:
        opt_grads = tf.gradients(ys=output['lm_losses'], xs=train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.compat.v1.summary.scalar('loss', output['lm_losses'])

    summary_log = tf.compat.v1.summary.FileWriter(checkpoint_path)

    saver = tf.compat.v1.train.Saver(
        var_list=train_vars,
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
    train_data = process_training_data(train_dataset, enc, context_maxlen=context_maxlen, response_maxlen=response_maxlen, history_len=history_len, reactonly=False)
    print('Training dataset has', len(train_data), 'training samples')
    train_batch_sampler = Batch_Sampler(train_data, batch_size, shuffle=True)

    val_data = process_training_data(val_dataset, enc, context_maxlen=context_maxlen, response_maxlen=response_maxlen, history_len=history_len, reactonly=False)
    print('Validation dataset has', len(val_data), 'validation samples')
    val_batch_sampler = Batch_Sampler(val_data, batch_size, shuffle=False)
    

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

    def validate():

        val_loss = 0
        n_examples = 0
        for i in range(len(val_data) // batch_size):
            (loss_v) = sess.run(output, feed_dict={context: val_batch_sampler.sample()})
            val_loss += loss_v['lm_losses']
            n_examples += 1

        return val_loss / n_examples


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
        best_loss = float("+inf")
        while True:
            if steps > 0 and counter == (counter_base + steps):
                return

            # Training
            if accumulate_gradients > 1:
                sess.run(opt_reset)
                for _ in range(accumulate_gradients):
                    sess.run(opt_compute, feed_dict={context: train_batch_sampler.sample()})
                (v_loss, v_summary) = sess.run((opt_apply, summary_loss))
            
            else:
                (_, loss, v_summary) = sess.run(
                    (opt_apply, output, summary_loss),
                    feed_dict={context: sample_batch()})
                v_loss = loss['lm_losses']
            
            summary_log.add_summary(v_summary, counter)

            # Validating
            avg_val_loss = validate()
            
            if counter % print_every == 0:
                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f} avg_val_loss={avg_val_loss:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1], avg_val_loss=avg_val_loss))
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_loss_step = counter
                save()

            if counter - best_loss_step >= patience:
                break

            counter += 1
    except KeyboardInterrupt:
        print('interrupted')


def step_core(model_hparams, tokens, past=None, past_tokens=None, do_dropout=False, name=None):
    lm_model = model.Model(hparams=model_hparams, scope='model')
    lm_model.built = tf.AUTO_REUSE
    lm_output = lm_model(X=tokens, past=past, past_tokens=past_tokens,
                    do_dropout=do_dropout, padding_token=50259)

# need to slice logits since we don't want to generate special tokens
    logits = lm_output['lm_logits'][:,:,:model_hparams.n_vocab]
    presents = lm_output['present']
    return {
        'logits': logits,
        'presents': presents,
    }

def format_response(response, response_maxlen):
    itemindex = np.where(response == 198)
    if itemindex[0].size == 0:
        return response.tolist()
    new_response = response[:itemindex[0][0] + 1].tolist()
    if len(new_response) == response_maxlen:
        return new_response
    new_response = new_response + (response_maxlen - len(new_response)) * [50259]
    assert len(new_response) == response_maxlen
    return new_response
    

def sample_responses(dataset,
           run_name,
           checkpoint_dir,
           write_path,
           seed=None,
           length=100,
           history_len=1,
           batch_size=1,
           temperature=1,
           top_k=0,
           top_p=1,
           n_samples=1):
    
    checkpoint_path = os.path.join(checkpoint_dir, run_name)
    enc = encodings.Encoding("main", n_vocab=50257, eot_token=50256, base_path=checkpoint_path).get_encoder()
    hparams = model.HParams()
    hparams.override_from_json_file(os.path.join(checkpoint_path, 'hparams.json'))
    data = sample_query(dataset, enc, context_maxlen=100, history_len=history_len)
    json_data = []
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
                            step=step_core,
                            model_hparams=hparams, length=length,
                            context=context,
                            batch_size=batch_size,
                            temperature=temperature, top_k=top_k, top_p=top_p
                        )
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        saver.restore(sess, ckpt)
        generated = 0
        while generated != n_samples:
            
            def batch():
                l = (len(data) // batch_size) * batch_size
                for ndx in range(0, l, batch_size):
                    yield data[ndx: ndx + batch_size]
            
            counter = 0
            for x in batch():
                context_tokens = [np.append(y, enc.encode(" </s> ")) for y in x]
                (out) = sess.run(output, feed_dict={context: context_tokens})
                result = out['tokens'][:, len(context_tokens[0].tolist()):]
                for i in range(len(result)):
                    if generated == 0:
                        temp = {}
                        temp['query'] = x[i].tolist()
                        temp['sample0'] = format_response(result[i], response_maxlen=length)
                        json_data.append(temp)
                    
                    else:
                        json_data[counter][f"sample{generated}"] = format_response(result[i], response_maxlen=length)
                        counter += 1
                        
            generated += 1
        
        with open(write_path, 'w') as fp:
            json.dump(json_data, fp)
    
    return