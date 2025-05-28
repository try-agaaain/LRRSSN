import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import scipy.io as scio
import mat73
from matplotlib import pyplot as plt
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim

def cal_ssim(Recover, Clean, data_range, win_size=7):
    """
    Recover和Clean是多时相多光谱图像
    Recover.shape: M, N, B, T
    Clean.shape: M, N, B, T

    return: 返回每一个时相的ssim
    """
    _, _, B, T = Recover.shape
    ssim = []
    for t in range(T):
        ssim_t = 0
        for b in range(B):
            ssim_t += compare_ssim(Recover[:, :, b, t], Clean[:, :, b, t], data_range=data_range, win_size=win_size)
        ssim.append(ssim_t/B)
    return ssim

def cal_psnr(Recover, Clean, data_range):
    """
    Recover和Clean是多时相多光谱图像
    Recover.shape: M, N, B, T
    Clean.shape: M, N, B, T

    return: 返回每一个时相的psnr
    """
    _, _, B, T = Recover.shape
    psnr = []
    for t in range(T):
        psnr_t = 0
        for b in range(B):
            psnr_t += compare_psnr(Recover[:, :, b, t], Clean[:, :, b, t], data_range=data_range)
        psnr.append(psnr_t/B)
    return psnr
    

def show_result(image_Clean, image_X, image_Y, image_C, image_M, bands, scale=2,
                save_result_path=None, update_num=0, row_nums=6):
    """
    bands：从每个时相的多光谱图像中选择bands波段，以RGB的形式展示
    save_result_path：matplot绘制图像后保存的文件路径
    update_num：当前的迭代次数

    图像的维度顺序：M, N, B, T
    子图的排列：5*T，每列5个图像，从上到下依次为 干净图像Clean，恢复图像X，观测图像Y，Y-X，云影图像C

    return：没有返回值
    """
    M, N, B, T = image_Clean.shape
    # for t in range(T):
    #     image_Clean[:, :, :, t] = normal(image_Clean[:, :, :, t]) * scale
    #     image_X[:, :, :, t] = normal(image_X[:, :, :, t]) * scale
    #     image_Y[:, :, :, t] = normal(image_Y[:, :, :, t]) * scale
    #     image_C[:, :, :, t] = normal(image_C[:, :, :, t]) * scale
    if isinstance(scale, int) or isinstance(scale, float):
        scale = [scale for t in range(T)]
    plt.figure(figsize=(40, 60))
    for t in range(T):
        plt.subplot(row_nums, T, 1 + t)
        plt.title("Clean %dth iter" % update_num)
        plt.imshow(image_Clean[:, :, bands, t] * scale[t])
        plt.axis('off')

        plt.subplot(row_nums, T, 1 + T + t)
        plt.title("X %dth iter" % update_num)
        plt.imshow(image_X[:, :, bands, t] * scale[t])
        plt.axis('off')

        plt.subplot(row_nums, T, 1 + 2*T + t)
        plt.title("Y %dth iter" % update_num)
        plt.imshow(image_Y[:, :, bands, t] * scale[t])
        plt.axis('off')

        plt.subplot(row_nums, T, 1 + 3*T + t)
        plt.title("Y-X")
        plt.imshow((image_Y[:, :, bands, t] - image_X[:, :, bands, t]) * scale[t])
        plt.axis('off')

        plt.subplot(row_nums, T, 1 + 4*T + t)
        plt.title("Cloud")
        plt.imshow(image_C[:, :, 0, t] * scale[t], cmap='gray')
        plt.axis('off')

        plt.subplot(row_nums, T, 1 + 5*T + t)
        plt.title("Mask")
        plt.imshow(image_M[:, :, 0, t], cmap='gray')
        plt.axis('off')
    if not save_result_path:
        plt.show()
    else:
        plt.savefig(save_result_path)
    plt.clf()
    plt.close()

def normal(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def get_args(args_list, strategy):
    """
    args_list：超参数列表
    strategy：获取参数的策略，grid是网格搜索，candidate是遍历候选参数
    """
    assert strategy == "grid" or strategy == "candidate", "没有选择合适的参数策略"

    if strategy == "grid":
        
        arg_num = len(args_list)
        for arg1 in args_list[0]:
            for arg2 in args_list[1]:
                for arg3 in args_list[2]:
                    for arg4 in args_list[3]:
                        if arg_num == 4:
                            yield arg1, arg2, arg3,arg4
                            break
                        for arg5 in args_list[4]:
                            if arg_num == 5:
                                yield arg1, arg2, arg3,arg4, arg5
                                break
                            for arg6 in args_list[5]:
                                if arg_num == 6:
                                    yield arg1, arg2, arg3,arg4, arg5, arg6
                                    break
                                for arg7 in args_list[6]:
                                    if arg_num == 7:
                                        yield arg1, arg2, arg3,arg4, arg5, arg6, arg7
                                        break
                                    for arg8 in args_list[7]:
                                        if arg_num == 8:
                                            yield arg1, arg2, arg3,arg4, arg5, arg6, arg7, arg8
                                            break
                                        else:
                                            assert True, "参数过多"

    elif strategy == "candidate":
        for args in args_list:
            yield args

def grid_search(args_list):
    """
    args_list：超参数列表
    """
    arg_num = len(args_list)
    assert arg_num==3 or arg_num==4 or arg_num==5, "超参数不为3和4，需要修改grid_search"
    if arg_num == 3:
        for arg1 in args_list[0]:
            for arg2 in args_list[1]:
                for arg3 in args_list[2]:
                    yield arg1, arg2, arg3
    if arg_num == 4:
        for arg1 in args_list[0]:
            for arg2 in args_list[1]:
                for arg3 in args_list[2]:
                    for arg4 in args_list[3]:
                        yield arg1, arg2, arg3, arg4
    if arg_num == 5:
        for arg1 in args_list[0]:
            for arg2 in args_list[1]:
                for arg3 in args_list[2]:
                    for arg4 in args_list[3]:
                        for arg5 in args_list[4]:
                            yield arg1, arg2, arg3,arg4, arg5

def show_loss_history(loss_history, loss1_history, loss2_history, save_path):
        plt.figure(figsize=(30, 60))
        plt.subplot(6, 1, 1)
        plt.title("total loss")
        plt.plot([i for i in range(len(loss_history[50:]))], loss_history[50:])

        plt.subplot(6, 1, 2)
        plt.title("loss1")
        plt.plot([i for i in range(len(loss1_history[50:]))], loss1_history[50:])

        plt.subplot(6, 1, 3)
        plt.title("loss2")
        plt.plot([i for i in range(len(loss2_history[50:]))], loss2_history[50:])

        plt.subplot(6, 1, 4)
        plt.title("total loss")
        plt.plot([i for i in range(len(loss_history))], loss_history)

        plt.subplot(6, 1, 5)
        plt.title("loss1")
        plt.plot([i for i in range(len(loss1_history))], loss1_history)

        plt.subplot(6, 1, 6)
        plt.title("loss2")
        plt.plot([i for i in range(len(loss2_history))], loss2_history)

        plt.savefig(save_path)
        plt.clf()
        plt.close()

def tensor_dilate_or_erode(bin_img, ksize=5, mode="erode"):
    """
    图像的膨胀和腐蚀操作，作用于掩码
    bin_img：掩码图像，维度为M,N,B,T
    ksize：膨胀和腐蚀操作kernel大小
    mode：指定操作方式，dilate为膨胀，erode为腐蚀
    返回膨胀或腐蚀后的图像
    """
    bin_img = bin_img.permute(3, 2, 0, 1)
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='reflect')

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    if mode == "erode":
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded.permute(2, 3, 1, 0)
    else:
        dilated, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
        return dilated.permute(2, 3, 1, 0)

def tensor_erode(bin_img, ksize=5):
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, H, W, C = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img.permute(0, 3, 1, 2), [pad, pad, pad, pad], mode='reflect')

    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    # 取每个 patch 中最小的值，i.e., 1
    eroded, _ = patches.reshape(B, H, W, C, -1).min(dim=-1)
    return eroded.permute(0, 3, 1, 2)
