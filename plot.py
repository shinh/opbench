#!/usr/bin/python3

import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Run benchmark for ops')
    parser.add_argument('drivers', type=str)
    parser.add_argument('--dir', type=str, default='results')
    parser.add_argument('-x', type=str, default='index')
    parser.add_argument('-y', type=str, default='gflops')
    args = parser.parse_args()

    drivers = args.drivers.split(',')

    by_category = {}
    for driver in drivers:
        for js in sorted(glob.glob('%s/%s/*.json' % (args.dir, driver))):
            with open(js) as f:
                info = dict(json.load(f))

            m = re.search(r'(\d+).json$', js)
            assert m
            index = int(m.group(1))
            info['driver'] = driver
            info['index'] = index
            category = info['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(info)

    for category, infos in sorted(by_category.items()):
        for driver in drivers:
            x = []
            y = []
            for info in infos:
                if info['driver'] != driver:
                    continue
                x.append(info[args.x])
                y.append(info[args.y])
            plt.scatter(x, y, label=driver)
        plt.legend(loc='upper left')
        plt.xlabel(args.x)
        plt.ylabel(args.y)
        plt.title(category)
        plt.savefig('%s/%s.png' % (args.dir, category))
        plt.show()


if __name__ == '__main__':
    main()
