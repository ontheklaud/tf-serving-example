import numpy as np


def main():

    factor = 1034
    random_values = 5000

    # open file header
    f = open('sample.csv', mode='w')

    # generate random vars per line
    for _ in range(random_values):
        rand_line = np.random.randn(factor).astype(np.float32)

        fit_str = ''
        for i in range(factor):
            fit_str += '{:f}{:s}'.format(rand_line[i], ',' if i < factor -1 else '')

        # write on file
        f.write('{:s}\n'.format(fit_str))
        f.flush()

    f.close()

    # fin
    return


if __name__ == '__main__':
    main()
