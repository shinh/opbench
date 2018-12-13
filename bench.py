import task as task_lib


def main():
    tasks = task_lib.collect_all_tasks()
    for task in tasks:
        print(task.name)
        task.run()


if __name__ == '__main__':
    main()
