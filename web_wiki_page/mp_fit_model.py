#!/usr/bin/env python3
#-*- coding: utf8 -*-
import argparse
import sys
import pickle
from multiprocessing import Process
from multiprocessing import cpu_count, current_process
from multiprocessing import Queue, JoinableQueue
from multiprocessing.queues import Empty
import pandas as pd
from tasks import fit_model


def worker(in_queue, out_queue):
    pid = current_process().pid
    while True:
        try:
            task = in_queue.get_nowait()
            if task is None:
                print("[{}] Sentinal value detected, worker exit".format(pid),
                      flush=True)
                in_queue.task_done()
                break
            lags_to_try, ts = task
            page, access = ts[0], ts[1:]
            print("[{}] Processing {}".format(pid, page), flush=True)
            results = {}
            for num_lags in lags_to_try:
                model = fit_model(num_lags, access)
                result = {}
                result["loss"] = model.fun
                result["success"] = model.success
                result["x"] = model.x
                results[num_lags] = result
            out_queue.put((page, results))
            in_queue.task_done()
        except Empty:
            # waiting for task
            pass
        except Exception as err:
            # TODO: join workers with errors
            # In this case, there may be undone tasks in in_queue.
            # That is, this program will hang there and wait for in_queue which will never join.
            print("Unknown error for worker:\n{}".format(err), flush=True)
            print("[{}] worker exit with error".format(pid), flush=True)
            break


def consumer(out_queue, out_fname):
    models = {}
    while True:
        try:
            record = out_queue.get_nowait()
            if record is None:
                print("Sentinal value detected, consumer exit", flush=True)
                break
            page, result = record
            models[page] = result
        except Empty:
            # waiting for record
            pass
        except Exception as err:
            print("Unknown error for consumer:\n{}".format(err), flush=True)
            print("Consumer exit", flush=True)
            return None

    # Saving results
    with open(out_fname, "wb") as wf:
        pickle.dump(models, wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-workers",
                        dest="num_workers",
                        help="number of processes",
                        default=cpu_count()*2,
                        type=int)
    parser.add_argument("csv_file", help="input csv file")
    parser.add_argument("-o", "--output-fname", help="output file name (pickle file)",
                        dest="out_fname", default=None)
    args = vars(parser.parse_args())
    num_workers = args["num_workers"]
    csv_file = args["csv_file"]
    out_fname = args["out_fname"]
    if out_fname is None:
        out_fname = csv_file.split(".")[0] + ".pickle"

    print("reading data: {}".format(csv_file), flush=True)
    train_data = pd.read_csv(csv_file, engine='c')
    train_data.fillna(0, inplace=True)

    in_queue = JoinableQueue()
    out_queue = Queue()
    for _ in range(num_workers):
        p = Process(target=worker, args=(in_queue, out_queue,))
        p.daemon = True
        p.start()

    p_consume = Process(target=consumer, args=(out_queue, out_fname,))
    p_consume.start()

    lags_to_try = [i for i in range(1, 9, 2)]
    num_series = train_data.shape[0]
    print("Setting input queue...")
    print("data enqued: 0.00%", flush=True)
    for index in range(num_series):
        series = train_data.iloc[index, :]
        in_queue.put((lags_to_try, series))
        if (index+1) % 10000 == 0:
            print("data enqued: {:.2f}%".format(100.0*(index+1)/num_series),
                  flush=True)
    print("data enqued: 100.00%", flush=True)
    for _ in range(num_workers):
        in_queue.put(None)
    print("Input queue setup done", flush=True)
    in_queue.close()
    in_queue.join()
    print("Input queue joined", flush=True)
    out_queue.put(None)
    p_consume.join()
    print("consumer joined", flush=True)

    print("All tasks done")
    sys.exit(0)
