def read_csv(path):
    data = open(path).readlines()
    #names = data[0].replace('\n', '').split('|')
    save_arr = []
    for line in data:
        save_arr.append(line)
    return save_arr

def make_sd(paths):
    train_f = open('../../GSL_iso_files/sd/train_greek_iso.csv','w')
    test_f = open('../../GSL_iso_files/sd/test_greek_iso.csv','w')
    data = []
    for path in paths:
        res = read_csv(path)
        data.extend(res)
    for line in data:
        if 'signer3' in line:
            test_f.write(line)
        else:
            train_f.write(line)
    train_f.close()
    test_f.close()

if __name__ == '__main__':

    train_gt_path = '../../GSL_iso_files/si/train_greek_iso.csv'
    test_gt_path = '../../GSL_iso_files/si/test_greek_iso.csv'

    make_sd([train_gt_path, test_gt_path])