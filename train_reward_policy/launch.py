#!/usr/bin/env python3

from lm_human_preferences.utils import launch
from lm_human_preferences.utils.combos import bind, combos, each, label, options_shortdesc, bind_nested
from lm_human_preferences import train_policy, train_reward

dialog_task = combos(
    bind('query_suffix', ' </s> '),
    bind('query_dataset', 'dialog'),
    bind('query_length', 100),
    bind('response_length', 100),
    bind('start_text', None),
    bind('truncate_token', 198),  # '\n'
    bind('policy.temperature', 0.7),
    bind('policy.initial_model', '355M'),
    
)

def get_train_reward_experiments():
    _shared = combos(
        bind('labels.type', 'best_of_2'),
        bind('normalize_after', True),
        bind('normalize_before', True),
        bind('normalize_samples', 256),
    )
    
    _dialog_task = combos(
        bind_nested('task', dialog_task),
        _shared,
        bind('labels.source', 'datasets/combined_labels_2.json'),
        bind('labels.num_train', 9280),

        bind('batch_size', 4),
        bind('lr', 1.5e-5),
        bind('rollout_batch_size', 4)
    )
    
    return locals()


def get_experiments():
    train_reward_experiments = get_train_reward_experiments()
    
    _dialog_task = combos(
        bind_nested('task', dialog_task),

        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['_dialog_task']),
        #bind('rewards.trained_model', 'temp/testmodel/reward_model'),
        bind('ppo.total_episodes', 10000),
        bind('ppo.lr', 2e-6),
        bind('rewards.kl_coef', 0.03), # 0.01 too low
        #bind('rewards.adaptive_kl', 'on'),
        #bind('rewards.adaptive_kl.target', 5.0),
        bind('ppo.batch_size', 5),
        bind('ppo.whiten_rewards', False),
        bind('run.log_interval', 25),
        bind('run.save_interval', 25)
    )
    return locals()


def launch_train_policy(exp, name, dry_run=False, mpi=8, mode='local', save_dir='/tmp/save/train_policy', **extra_hparams):
    experiment_dict = get_experiments()
    try:
        trials = experiment_dict[exp]
    except KeyError:
        raise ValueError(f"Couldn't find experiment '{exp}'")

    launch.launch_trials(
        name, fn=train_policy.train, trials=trials, mpi=mpi, mode=mode, save_dir=save_dir,
        hparam_class=train_policy.HParams, extra_hparams=extra_hparams, dry_run=dry_run)


def launch_train_reward(exp, name, dry_run=False, mpi=8, mode='local', save_dir='/tmp/save/train_reward', **extra_hparams):
    experiment_dict = get_train_reward_experiments()
    try:
        trials = experiment_dict[exp]
    except KeyError:
        raise ValueError(f"Couldn't find experiment '{exp}'")

    launch.launch_trials(
        name, fn=train_reward.train, trials=trials, mpi=mpi, mode=mode, save_dir=save_dir,
        hparam_class=train_reward.HParams, extra_hparams=extra_hparams, dry_run=dry_run)


if __name__ == '__main__':
    launch.main(dict(
        train_policy=launch_train_policy,
        train_reward=launch_train_reward
    ))
