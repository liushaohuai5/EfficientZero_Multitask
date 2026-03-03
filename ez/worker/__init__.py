import os
import time
# import SMOS
import ray
import torch
import numpy as np
import random

from omegaconf import OmegaConf

import torch.distributed as dist
import torch.multiprocessing as mp

from ez.worker.watchdog_worker import start_watchdog_server, start_watchdog_worker, get_watchdog_server
from ez.data.global_storage import GlobalStorage
from ez.data.replay_buffer import ReplayBuffer, start_replay_buffer_server, get_replay_buffer
from ez.worker.data_worker import start_data_worker
from ez.worker.batch_worker import start_batch_worker, start_batch_worker_cpu, start_batch_worker_gpu
from ez.worker.eval_worker import start_eval_worker
from ez.utils.format import RayQueue, PreQueue


def start_workers(agent, manager, config):
    # ==================================================================================================================
    # start server
    # ==================================================================================================================

    # global storage server
    storage_server = GlobalStorage.remote(agent.build_model(), agent.build_model(), agent.build_model())
    print('[main process] Global storage server has been started from main process.')

    # batch queue
    batch_storage = RayQueue(15, 20)
    print('[main process] Batch storage has been initialized.')

    # replay buffer server
    # replay_buffer_seed = random.randint(0, 1000)
    replay_buffer_server = ReplayBuffer.remote(batch_size=config.train.batch_size,
                                               buffer_size=config.data.buffer_size,
                                               top_transitions=config.data.top_transitions,
                                               use_priority=config.priority.use_priority,
                                               env=config.env.env,
                                               total_transitions=config.data.total_transitions,
                                               multi_task=config.env.multi_task,
                                               task_num=config.env.task_num,
                                               seed=config.env.base_seed,
                                               ind_exp_rp=config.data.ind_exp_rp,
                                               continuous_action=config.env.continuous_action,)
    print('[main process] Replay buffer server has been started from main process.')

    # watchdog server
    watchdog_server = start_watchdog_server(manager)
    print('[main process] Watchdog server has been started from main process.')

    # ==================================================================================================================
    # start worker
    # ==================================================================================================================

    # data workers
    # data_worker_seeds = random.sample(range(1000), config.actors.data_worker)
    data_workers = [
        start_data_worker(rank, agent, replay_buffer_server, storage_server, config, seed=config.env.base_seed)
        for rank in range(0, config.actors.data_worker)]
    print(f'[main process] {config.actors.data_worker} Data workers have all been launched.')

    # batch worker
    # batch_worker_seeds = random.sample(range(1000), config.actors.batch_worker)
    batch_workers = [start_batch_worker(rank, agent, replay_buffer_server, storage_server, batch_storage, config, seed=config.env.base_seed)
                     for rank in range(0, config.actors.batch_worker)]
    print(f'[main process] {config.actors.batch_worker} Batch workers have all been launched.')

    # eval worker
    start_eval_worker(agent, replay_buffer_server, storage_server, config)

    if int(torch.__version__[0]) == 2:
        print(f'[main process] torch version is {torch.__version__}, enabled torch_compile.')

    # trainer (in current process)
    worker_lst = [data_workers, batch_workers, eval_worker]
    server_lst = [storage_server, replay_buffer_server, watchdog_server, batch_storage]

    return worker_lst, server_lst


def join_workers(worker_lst, server_lst):
    data_workers, batch_workers, eval_worker, watchdog_worker = worker_lst
    storage_server, replay_buffer_server, watchdog_server, smos_server = server_lst

    # wait for all workers to finish
    watchdog_worker.terminate()
    for data_worker in data_workers:
        data_worker.join()
    for batch_worker in batch_workers:
        batch_worker.join()
    eval_worker.join()
    print(f'[main process] All workers have stopped.')

    # stop servers
    storage_server.terminate()
    replay_buffer_server.terminate()
    watchdog_server.terminate()
    smos_server.stop()
    print(f'[main process] All servers have stopped.')

