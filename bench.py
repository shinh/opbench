import argparse
import glob
import importlib
import json
import os
import re

import task as task_lib
import utils


def collect_all_driver_names():
    driver_pys = sorted(glob.glob('drivers/*.py'))
    driver_pys += sorted(glob.glob('*/drivers/*.py'))
    return [d[:-3].replace('drivers/', '') for d in driver_pys]


def load_driver(driver_name):
    toks = driver_name.split('/')
    if len(toks) == 1:
        module_name = 'drivers.%s' % toks[0]
    else:
        assert len(toks) == 2, driver_name
        module_name = '%s.drivers.%s' % (toks[0], toks[1])
    module = importlib.import_module(module_name)
    return module.get_driver()


def report(driver, task, result):
    count = len(result)
    elapsed = sum(result) / count
    gflops = task.model.flops / elapsed / 1000 / 1000 / 1000
    print(task.name, '%.1f GFLOPS/sec' % gflops, 'cnt=%d' % count)

    dirname = os.path.join('results', driver.name())
    utils.makedirs(dirname)
    info = [
        ('category', task.category),
        ('name', task.name),
        ('gflops', gflops),
        ('elapsed', elapsed),
        ('count', count),
    ]
    info.extend(task.model.info())
    with open(os.path.join(dirname, task.name + '.json'), 'w') as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run benchmark for ops')
    parser.add_argument('driver_name', choices=collect_all_driver_names())
    parser.add_argument('--filter', '-f', type=str)
    parser.add_argument('--time_per_task', '-t', default=1.0)
    args = parser.parse_args()

    driver = load_driver(args.driver_name)

    filter = None
    if args.filter:
        filter = re.compile(args.filter)

    tasks = task_lib.collect_all_tasks()
    for task in tasks:
        if filter and not filter.search(task.name):
            continue
        result = driver.bench(task, time_budget_sec=args.time_per_task)
        report(driver, task, result)
        task.finish()


if __name__ == '__main__':
    main()
