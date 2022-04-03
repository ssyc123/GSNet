import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Polt a Loss')
    parser.add_argument('--path', help='log file path', default="/home/syc/mmdection/CG-Net-master/syc/faster_rcnn_RoITrans_r101_res_gbfpn_k=11_epoch=50/20211208_191851.log")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.path, "r") as f:
        f.readline()
        f.readline()
        text_list = []
        x = []
        for i in range(6750):
            x.append(i+1)
            text = eval(f.readline().split("loss:")[1].split("\n")[0].split(" ")[1])
            # print(text)
            text_list.append(text)
        print(text_list)
        print("minindex:", text_list.index(min(text_list))/96)
        # plt.plot(text_list)
        # plt.show()
        plt.plot(text_list)
        plt.show()