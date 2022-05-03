import argparse
import pandas as pd


def check_kge(kge_file, target1, target2, tol):

    df = pd.read_csv(kge_file)
    kge_max = float(df['effmax_R'][-1:])
    check = abs(kge_max - target2) < tol and kge_max > target1

    if check:
        print('Yay! KGE target reached!')
    else:
        raise Exception("Target not reached! abs({} - {}) = {} is > {} or {} < {}".format(kge_max, target2, abs(kge_max-target2), tol, kge_max, target1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('kge_file', help='KGE front history file')
    parser.add_argument('tol', help='KGE history file')
    parser.add_argument('target1', default='0.99', help='lower target value')
    parser.add_argument('target2', default='1.00', help='higher target value')
    args = parser.parse_args()

    tol = float(args.tol)
    target1 = float(args.target1)
    target2 = float(args.target2)
    print(f'KGE file: {args.kge_file}')
    print(f'Tolerance is {tol}')
    print(f'Target 1 is {target1}')
    print(f'Target 2 is {target2}')
    # check KGE
    check_kge(args.kge_file, target1=target1, target2=target2, tol=2*tol)
