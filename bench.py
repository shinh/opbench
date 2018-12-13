import glob
import importlib
import os


def import_file(filename):
    module_name = filename[:-3].replace('/', '.')
    return importlib.import_module(module_name)


def collect_all_tasks():
    task_pys = sorted(glob.glob('tasks/*.py'))
    task_pys += sorted(glob.glob('*/tasks/*.py'))
    tasks = []
    for task_py in task_pys:
        module = import_file(task_py)
        tasks.extend(module.get_tasks())
    return tasks


def main():
    tasks = collect_all_tasks()
    for task in tasks:
        print(task.name)
        task.to_gpu()
        inputs = task.inputs()
        if not isinstance(inputs, tuple):
            inputs = [inputs]
        inputs = [task.xp.array(input) for input in inputs]
        outputs = task(*inputs)


main()
