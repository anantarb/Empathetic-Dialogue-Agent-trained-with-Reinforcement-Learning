#!/usr/bin/env python3

import os
from functools import partial

from mpi4py import MPI
import tensorflow as tf

from lm_human_preferences.utils import launch, hyperparams
from lm_human_preferences.utils import core as utils
from lm_human_preferences.policy import Policy
from lm_human_preferences.language import trained_models
from lm_human_preferences import lm_tasks
from lm_human_preferences import train_policy
import json

def sample_policy(write_path=None, save_dir=None, savescope='policy', temperature=0.5, seed=None, batch_size=32, nsamples=0):
    hparams = train_policy.HParams()
    hparams.override_from_json_file(os.path.join(save_dir, 'train_policy_hparams.json'))
    print('hparams', hparams)
    task = hparams.task

    comm = MPI.COMM_WORLD
    nsamples_per_rank = utils.exact_div(nsamples, comm.Get_size())
    with tf.Graph().as_default():
        m = trained_models.TrainedModel(name='sample', savedir=os.path.join(save_dir, savescope), scope='policy')
        encoder = m.encoding.get_encoder()
        hyperparams.dump(m.hparams(), name='model_hparams')

        utils.set_mpi_seed(seed)

        policy = Policy(
            m, scope='policy',
            is_root=True, # just init on every rank, simplifies code
            embed_queries=lm_tasks.query_formatter(task, encoder),
            temperature=temperature,
        )

        query_sampler = lm_tasks.make_query_sampler(
            hparams=task, encoder=encoder, comm=comm,
            batch_size=batch_size, mode='test'
        )

        init_ops = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        )

        with utils.mpi_session() as sess:
            init_ops.run()
            @utils.graph_function()
            def sample_queries():
                return query_sampler()['tokens']

            tf.get_default_graph().finalize()

            generated = 0
            json_data = []
            while nsamples_per_rank == 0 or generated < nsamples_per_rank:
                queries = sample_queries()
                rollouts = policy.respond(queries, length=task.response_length)
                assert len(queries.tolist()) == batch_size
                assert len(rollouts['responses'].tolist()) == batch_size
                for q, r in zip(queries.tolist(), rollouts['responses'].tolist()):
                    temp = {}
                    temp['query'] = encoder.decode(q)
                    res = encoder.decode(r)
                    temp['sample0'] = res.split('\n')[0]
                    json_data.append(temp)
                generated += batch_size
                if generated % 1024 == 0:
                    print(generated)
                if generated >= 8416:
                    break
            
            with open(write_path, 'w') as fp:
                json.dump(json_data, fp)

def launch_sample(mode='local', mpi=8, **kwargs):
    launch.launch('sample', partial(sample_policy, **kwargs), mode=mode, mpi=mpi)

if __name__ == '__main__':
    launch.main(dict(
        sample=launch_sample,
    ))

"""
./sample.py sample --save_dir gs://jeffwu-rcall/results/safety/lmhf-sent-69c5170-1909161359/ --mpi 8
"""
