import argparse
import glob
import importlib

import task as task_lib


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


def main():
    parser = argparse.ArgumentParser(description='Run benchmark for ops')
    parser.add_argument('driver_name', choices=collect_all_driver_names())
    parser.add_argument('--time_per_task', '-t', default=1.0)
    args = parser.parse_args()

    driver = load_driver(args.driver_name)

    tasks = task_lib.collect_all_tasks()
    for task in tasks:
        result = driver.bench(task, time_budget_sec=args.time_per_task)
        elapsed = sum(result) / len(result)
        flops = task.model.flops / elapsed / 1000 / 1000 / 1000
        print(task.name, '%.1f GFLOPS/sec' % flops)


if __name__ == '__main__':
    main()
