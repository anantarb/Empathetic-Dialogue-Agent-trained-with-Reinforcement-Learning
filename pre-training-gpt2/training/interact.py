import numpy as np
import tensorflow as tf
from models import model, encodings, sample
import os

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
                
def run(run_name='run1',
    checkpoint_dir='checkpoint',
    seed=None,
    length=100,
    history_len=1,
    temperature=0.7,
    top_k=0,
    top_p=1):
        
    checkpoint_path = os.path.join(checkpoint_dir, run_name)
    enc = encodings.Encoding("main", n_vocab=50257, eot_token=50256, base_path=checkpoint_path).get_encoder()
    hparams = model.HParams()
    hparams.override_from_json_file(os.path.join(checkpoint_path, 'hparams.json'))
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
                            step=step_core,
                            model_hparams=hparams, length=length,
                            context=context,
                            batch_size=1,
                            temperature=temperature, top_k=top_k, top_p=top_p
                        )
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        saver.restore(sess, ckpt)
        history = []
        print("Type exit anytime if you want to leave.")
        while True:
            raw_text = input("You: ")
            while not raw_text:
                print('Model will not reply to empty prompt')
                raw_text = input("You: ")
            if raw_text == 'exit':
                break
            history.append(raw_text)
            prev_str = " <SOC> ".join(history[-history_len:])
            padding_token = enc.padding_token
            context_tokens = enc.encode(prev_str)
            context_tokens = context_tokens[:100]
            if len(context_tokens) < 100:
                context_tokens = context_tokens + [padding_token] * (100 - len(context_tokens))
            else:
                context_tokens =  context_tokens[len(context_tokens) - 100:]
            context_tokens = context_tokens + enc.encode(" </s> ")
            (out) = sess.run(output, feed_dict={context: [context_tokens]})
            result = out['tokens'][:, len(context_tokens):]
            text = enc.decode(result[0])
            chunks = text.split('\n')
            history.append(chunks[0])
            print("BOT: " + chunks[0])
    
    return 