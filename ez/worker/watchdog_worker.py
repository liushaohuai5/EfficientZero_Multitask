import os
import time
import ray

@ray.remote
class WatchdogServer(object):
    def __init__(self):
        self.reanalyze_batch_count = 0
        self.training_step_count = 0

    def increase_reanalyze_batch_count(self):
        self.reanalyze_batch_count += 1

    def get_reanalyze_batch_count(self):
        return self.reanalyze_batch_count

    def increase_training_step_count(self):
        self.training_step_count += 1

    def get_training_step_count(self):
        return self.training_step_count


# ======================================================================================================================
# watchdog server
# ======================================================================================================================
def start_watchdog_server(manager):
    """
    Start a watchdog server. Call this method remotely.
    """
    watchdog_server = WatchdogServer.remote()
    print('[Watchdog Server] Watchdog server initialized.')
    return watchdog_server


def get_watchdog_server(manager):
    return get_remote_object(manager.watchdog_server_connection, manager.watchdog_server_register_name)


@ray.remote
def start_watchdog_worker(watchdog_server):
    """
    Start a watchdog that monitors training statistics. Call this method remotely.
    """
    # watchdog_server = get_watchdog_server(manager)

    # get SMOS client
    # smos_client = SMOS.Client(connection=manager.smos_connection)

    # start watching statistics
    last_batch_count = 0
    last_training_step_count = 0
    while True:

        # watchdog
        time.sleep(10)
        batch_count = ray.get(watchdog_server.get_reanalyze_batch_count.remote())
        training_step_count = ray.get(watchdog_server.get_training_step_count.remote())
        # if manager.log_smos:
        # print('*********************** Watchdog ***********************')
        # print(f'Reanalyze speed: {batch_count - last_batch_count} batches/10sec')
        # print(f'Training speed: {training_step_count - last_training_step_count} steps/10sec')
        # print('********************************************************')
        last_batch_count = batch_count
        last_training_step_count = training_step_count

        # zombie killer
        # time.sleep(5)
        # zombie_list = []
        # kill_zombie_count = 0
        # remaining_zombie_list = []
        # while True:
        #     status, handle, idx = smos_client.pop_from_object(name=manager.share_memory_config.zombie_queue_name)
        #     if status == SMOS.SMOS_FAIL:
        #         break
        #     else:
        #         zombie_list.append(idx)
        #         smos_client.free_handle(object_handle=handle)
        # for zombie_idx in zombie_list:
        #     status = smos_client.delete_entry(name=manager.share_memory_config.replay_buffer_name, entry_idx=zombie_idx)
        #     if status == SMOS.SMOS_PERMISSION_DENIED:
        #         smos_client.push_to_object(name=manager.share_memory_config.zombie_queue_name, data=zombie_idx)
        #         remaining_zombie_list.append(zombie_idx)
        #     else:
        #         kill_zombie_count += 1
        # _, replay_buffer_size = smos_client.get_entry_count(name=manager.share_memory_config.replay_buffer_name)
        # if manager.log_smos:
        #     print('********************* Zombie Killer ********************')
        #     print(f'Replay buffer size: {replay_buffer_size}')
        #     print(f'Killed: {kill_zombie_count} zombies, remaining: {remaining_zombie_list}')
        #     print('********************************************************')