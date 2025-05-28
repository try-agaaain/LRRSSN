from cloud_removal import CloudRemoval
from tools import get_args
import torch
from load_data import load_dataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda")

need_save_loss = False  # 是否保存损失值
need_save_X = True  # 是否保存恢复结果X
need_save_best_recover = False  # 是否保存最佳恢复结果
need_update_mask = True  # 是否更新掩码

just_gdd = False  # 是否只使用GDD网络，去掉数学模型（消融）
just_math_model = False  # 是否只使用基础数学模型（消融）
save_all_image = True  # 是否在日志中保存中间结果的图像

LR = 0.002  # 学习率
strategy = "candidate"  # 参数获取策略，grid为网格搜索（调参阶段），candidate为候选参数遍历（测试阶段

acc_mode = "inacc" if need_update_mask else "acc"   # 是否采用准确的掩码
method_mode = "just_model" if just_math_model else "just_gdd" if just_gdd else "model_and_gdd"

args_list = [
    # rate：对应公式（11）中的 λ1 / ρ，调节低秩约束的强度
    # lambda2：正则化参数，控制低秩正则项的强度，有助于稀疏或低秩特征的学习
    # rho：HQS算法中的惩罚参数，影响收敛速度与效果
    # alpha：ADMM中的惩罚参数更新系数，调节 rho 的增长速度
    # cloud_threshold：云掩码阈值，用于区分云和非云区域，作用于生成云掩码的敏感度
    # clean_weight：引导图像中干净区域的权重比例
    # epoch_num：训练的最大epoch数
    # iter_num：每次自监督学习的迭代次数
    [20, 0.5, 0.001, 0.02, 0.6, 0.05, 600, 2],
]

# 上述参数为较好的参数配置，可直接使用，需要调整的

if __name__ == "__main__":
    scene_selected = "city_time_num[6]"

    order = 100 # 记录实验的顺序
    repeat = 5  # 每组参数实验的重复次数
    root_log = f"./logs/{scene_selected}/{acc_mode}/{method_mode}-args"
#    
    for rate, lambda2, rho, alpha, cloud_threshold, clean_weight, epoch_num, iter_num in get_args(args_list, strategy):
        for t in range(repeat):
            
            # 加载数据集
            data = load_dataset(scene_selected)
            CR = CloudRemoval(dtype, device, scene_selected, strategy,
                    save_all_image, need_save_loss, need_save_X, 
                    need_save_best_recover, need_update_mask, just_math_model, just_gdd,
                    root_log, iter_num, LR)
            CR.cloud_remove(data,
                rate, lambda2, rho, alpha, cloud_threshold, 
                order, clean_weight, epoch_num)
            order += 1
