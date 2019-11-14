from pytracking import run_tracker


def main():
    tracker_name = 'etcom'
    tracker_param = 'DiMP50_ICCV_submission'
    runid = None
    dataset = 'nfs'
    sequence = 77
    debug = 1

    run_tracker(tracker_name, tracker_param, runid, dataset, sequence, debug)


if __name__ == '__main__':
    main()
