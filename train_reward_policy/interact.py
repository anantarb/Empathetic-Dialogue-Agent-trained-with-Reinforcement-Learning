import os

from mpi4py import MPI
import tensorflow as tf

from lm_human_preferences.utils import hyperparams
from lm_human_preferences.utils import core as utils
from lm_human_preferences.policy import Policy
from lm_human_preferences.language import trained_models
from lm_human_preferences import lm_tasks
from lm_human_preferences import train_policy


def run(run_name='run1',
    checkpoint_dir='checkpoint',
    seed=None,
    length=100,
    history_len=1,
    temperature=0.7):
    
    hparams = train_policy.HParams()
    hparams.override_from_json_file(os.path.join(checkpoint_dir, run_name, 'train_policy_hparams.json'))
    task = hparams.task
    comm = MPI.COMM_WORLD
    with tf.Graph().as_default():
        m = trained_models.TrainedModel(name='sample', savedir=os.path.join(checkpoint_dir, run_name, 'policy'), scope='policy')
        encoder = m.encoding.get_encoder()
        hyperparams.dump(m.hparams(), name='model_hparams')
        utils.set_mpi_seed(seed)
        
        policy = Policy(
            m, scope='policy',
            is_root=True, # just init on every rank, simplifies code
            embed_queries=lm_tasks.query_formatter(task, encoder),
            temperature=temperature,
        )
        
        init_ops = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        )
        
        with utils.mpi_session() as sess:
            history = []
            init_ops.run()
            def get_input():
                raw_text = input("You: ")
                while not raw_text:
                    print('Model will not reply to empty prompt')
                    raw_text = input("You: ")
                return raw_text
            #tf.get_default_graph().finalize()
            print("Type exit anytime if you want to leave.")
            while True:
                query = get_input()
                if query == 'exit':
                    break
                history.append(query)
                prev_str = " <SOC> ".join(history[-history_len:])
                queries = encoder.encode(prev_str)
                rollouts = policy.respond([queries], length=length)
                response = rollouts['responses'].tolist()
                res = encoder.decode(response[0])
                history.append(res.split('\n')[0])
                print("BOT: " + res.split('\n')[0])
                
                
                