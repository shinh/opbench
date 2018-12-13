import glob
import json
import os


def main():
    results = {}
    for js in sorted(glob.glob('results/*/*.json')):
        with open(js) as f:
            info = json.load(f)

        dirname = os.path.dirname(js)
        key, category = info[0]
        assert key == 'category'
        if (dirname, category) not in results:
            results[(dirname, category)] = []
        results[(dirname, category)].append(info[1:])

    for (dirname, category), infos in results.items():
        keys = [k for k, _ in infos[0]]
        with open('%s/%s.tsv' % (dirname, category), 'w') as f:
            f.write('\t'.join(keys) + '\n')
            for info in infos:
                values = [str(v) for _, v in info]
                f.write('\t'.join(values) + '\n')


if __name__ == '__main__':
    main()
