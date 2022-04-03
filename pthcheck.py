import torch  # 命令行是逐行立即执行的
content = torch.load('/home/syc/mmdection/CG-Net-master/syc/faster_rcnn_RoITrans_r101_res_gbfpn_k=11/epoch_12.pth')
print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
print(content['state_dict'].keys())
