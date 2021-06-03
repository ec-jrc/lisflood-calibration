import argparse
import pandas as pd


def check_kge(kge_file, target1, target2, tol):

    df = pd.read_csv(kge_file)
    kge_max = float(df['effmax_R'][-1:])
    check = abs(kge_max - target1) < tol and kge_max > target2

    if check:
        print('Yay! KGE target reached!')
    else:
        raise Exception("Target not reached! abs({} - {}) = {} is > {} or {} < {}".format(kge_max, target1, abs(kge_max-target1), tol, kge_max, target2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('kge_file', help='KGE front history file')
    parser.add_argument('tol', help='KGE history file')
    args = parser.parse_args()

    tol = float(args.tol)
    print('KGE file: {}'.format(args.kge_file))
    print('Tolerance is {}'.format(tol))

    # check KGE
    check_kge(args.kge_file, target1=1., target2=0.99, tol=2*tol)
