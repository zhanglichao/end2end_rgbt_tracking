import os
import sys
import multiprocessing

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from ltr.run_training import run_training


def main():
    train_module = 'seq_tracking'
    train_name = 'debug_lichao_comb23_2_lr4_lr2' # 'debug_martin'

    run_training(train_module, train_name)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
