from __future__ import print_function
import os
import random
import threading
import time
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import torch.optim
from load_data import load_dataset
from tools import show_result, cal_psnr, show_loss_history, get_args, normal, cal_ssim
from gdd.GDD import gdd as net

################################ 参数设置 ################################
class CloudRemoval():
    def __init__(self, dtype, device,
                scene_selected, strategy,
                save_all_image, need_save_loss, 
                need_save_X, need_save_best_recover, 
                need_update_mask, just_math_model, just_gdd,
                root_log, iter_num,
                LR, time_num=None) -> None:
        self.dtype, self.device = dtype, device
        self.scene_selected, self.strategy = scene_selected, strategy
        self.save_all_image, self.need_save_loss = save_all_image, need_save_loss
        self.need_save_X, self.need_save_best_recover = need_save_X, need_save_best_recover
        self.need_update_mask, self.just_math_model, self.just_gdd = need_update_mask, just_math_model, just_gdd
        self.root_log, self.iter_num = root_log, iter_num
        self.LR = LR
        self.time_num = time_num
        # 设置锁
        self.spin_lock = threading.Lock()
        self.get_args = get_args

    @staticmethod
    def set_seed(seed=4096):
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    @staticmethod
    def thres_21(L, tau, M, N, B):
        S = torch.sqrt(torch.sum(torch.mul(L, L), 2))
        S[S == 0] = 1
        T = 1 - tau / S
        T[T < 0] = 0
        R = T.reshape(M, N, 1).repeat((1, 1, B))
        res = torch.mul(R, L)
        return res

    @staticmethod
    def deal_real_data_for_gdd(Noisy):
        """
        数据的维度：M, N, B, T
        将图像尺寸裁剪为32的倍数，把云层最少的图像作为引导图像
        返回Clean，Noisy，Mask和Guide，他们的维度均为M, N, B, T
        """
        Noisy = Noisy[:, :, [2, 3, 4, 8], :]
        M, N, B, T = Noisy.shape
        M, N = M - M % 32, N - N % 32   # 确保图像尺寸为32的倍数
        Noisy = Noisy[:M, :N, :, :]
        Guide = Noisy[:M, :N, :, [0]].repeat(1, 1, 1, T)
        return Noisy, Guide
    def cloud_remove(self, data, rate, lambda2, rho, alpha, cloud_threshold, order, clean_weight, epoch_num):
        rho_orig = rho
        lambda1 = rate * rho
        ################################ 读取数据 ################################
        Clean, Noisy, Mask, clean_time, bands, data_range, scale = data
        Clean = torch.from_numpy(Clean).type(self.dtype).to(self.device)
        Mask = torch.from_numpy(Mask).type(self.dtype).to(self.device)  # 云=0
        Noisy = torch.from_numpy(Noisy).type(self.dtype).to(self.device)
        M, N, B, T = Clean.shape
        ################################ 创建日志文件 ################################
        if not os.path.exists(self.root_log):
            os.makedirs(self.root_log)
        log_dir = f"{self.root_log}/order[{order:03d}]-" + \
                  f"iter[{self.iter_num:3d}]-epoch[{epoch_num:3d}]-" + \
                  f"rate[{rate:.3f}]-lambda2[{lambda2:.3f}]-" + \
                  f"rho[{rho:.4f}, {alpha:.3f}]-threshold[{cloud_threshold:.5f}]"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            print("\u001b[1;34m", "Pass: ", log_dir, "\u001b[0m")
            return
        loss_trend_path = "%s/loss_trend" % log_dir
        if self.need_save_loss and not os.path.exists(loss_trend_path):
            os.makedirs(loss_trend_path)
        image_save_path = "%s/images" % log_dir
        if self.save_all_image and not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        
        print("\u001b[1;32m", "Start: ", log_dir, "\u001b[0m")
        ################################ 初始化 ################################
        Y = Noisy.clone().type(self.dtype)
        X = Noisy.clone().type(self.dtype)
        M_update = torch.zeros(Noisy.shape).type(self.dtype).to(self.device) \
                    if self.need_update_mask else Mask.clone()
        W = Noisy.clone().type(self.dtype)
        C = Y - torch.mul(Y, M_update)
        # net_input = torch.permute(Guide, [2, 3, 0, 1])    # B, T, M, N
        # net_input = net_input.reshape(1, B * T, M, N).type(self.dtype).to(self.device)
        # 初始化网络
        channels = 32
        # CloudRemoval.set_seed()
        model = net( num_input_channels = B * T, 
                num_output_channels = B * T,
                num_channels_down = channels,
                num_channels_up = channels,
                num_channels_skip = channels,
                filter_size_up = 3, filter_size_down = 3, filter_skip_size=1,
                upsample_mode='bilinear', # downsample_mode='avg',
                need1x1_up=False,
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU',
                guide_input_channels_num=T*B).type(self.dtype)
        # model.load_state_dict(torch.load("save.pt"))
        # model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR, betas=(0.9, 0.999))
        iters = 0
        max_psnr = -10000
        best_iters = iters
        avg_psnr = avg_psnr_record = 0
        ################################ 外迭代，迭代的更新各个子问题 ################################
        stop_increase = 0
        Last_R = torch.zeros(Noisy.shape).type(self.dtype).to(self.device)
        while iters < self.iter_num:
            if self.just_math_model and iters > 25:
                if avg_psnr < 15:
                    break
                if (iters - best_iters) >= 5 and avg_psnr <= avg_psnr_record:
                    break
            iters += 1
            epoch, min_loss = 0, 1000000

            ################################ 更新子问题X ################################
            loss_history, loss1_history, loss2_history = [], [], []
            X_save = None
            CloudRemoval.set_seed()
            noise = torch.rand((1, B*T, int(M / 32), int(N / 32))).type(self.dtype).to(self.device)
            if clean_time != []:
                cloud_time = [i for i in range(T) if i not in clean_time]
                X_g = X.clone()
                X_g[M_update == 1] = Y[M_update == 1]
                X_avg = torch.zeros([M, N, B]).type(self.dtype).to(self.device)
                factor = 0
                for t in range(T):
                    if t in clean_time:
                        X_avg += clean_weight * X_g[:, :, :, t]
                        factor += clean_weight
                    else:
                        X_avg += X_g[:, :, :, t]
                        factor += 1
                X_avg = X_avg / factor
                net_input = X_avg.reshape([M, N, B, 1]).repeat(1, 1, 1, T)
                net_input = net_input.permute([2, 3, 0, 1]).reshape((1, B*T, M, N)).type(self.dtype).to(self.device)
            if not self.just_math_model:
                # 内迭代：用网络更新子问题X
                while epoch < epoch_num:
                    optimizer.zero_grad()
                    out = model(net_input, noise)
                    X = torch.permute(out.reshape(B, T, M, N), [2, 3, 0, 1]).type(self.dtype)
                    # 计算损失并更新网络
                    loss1 = 1 / 2 * torch.norm(Y - torch.mul(X, M_update) - C, 'fro') ** 2
                    loss2 = rho / 2 * torch.norm(X - W, 'fro') ** 2
                    loss = loss1 if self.just_gdd else loss1 + loss2
                    loss.backward()
                    optimizer.step()
                    # 保存loss最低时的恢复图像作为本次网络的更新结果
                    if loss.item() < min_loss:
                        X_save = X.clone()
                        min_loss = loss.item()
                    # scheduler.step()
                    # 记录损失变化情况
                    loss_history.append(loss.item())
                    loss1_history.append(loss1.item())
                    loss2_history.append(loss2.item())
                    if epoch % 40 == 0:
                        print("\tThe %03dth iters, the %03dth epoch, loss: %.5f, loss1: %.5f, loss2: %.5f" \
                            % (iters, epoch, loss.item(), loss1.item(), loss2.item()))
                    Y, C, W, X = Y.detach(), C.detach(), W.detach(), X.detach()
                    epoch += 1
                X = X_save
                Y, C = Y.detach().type(self.dtype), C.detach().type(self.dtype), 
                W, X = W.detach().type(self.dtype), X.detach().type(self.dtype)
                loss = loss.item()
            else:
                epoch = 0
                X = (torch.mul(M_update, (Y - C)) + rho * W ) / (M_update + rho)
                loss = 1 / 2 * torch.norm(Y - torch.mul(X, M_update) - C, 'fro') ** 2 + \
                    rho / 2 * torch.norm(X - W, 'fro') ** 2
            ################################ 更新子问题W ################################
            if not self.just_gdd:
                U, s, VH = torch.linalg.svd(X.reshape(M * N, B * T), full_matrices=False)  # B*T, M*N
                s = s - lambda1 / rho
                s[s < 0] = 0
                S = torch.diag(s)
                W = torch.mm(torch.mm(U, S), VH).reshape(M, N, B, T).type(self.dtype)
            ################################ 更新子问题C ################################
            L = Y - torch.mul(X, M_update)
            for t in range(T):
                C[:, :, :, t] = CloudRemoval.thres_21(L[:, :, :, t], lambda2, M, N, B)
            ################################ 更新掩码M ################################
            if self.need_update_mask:
                B_mean = torch.mean(torch.abs(C), dim=2).reshape(M, N, 1, T).repeat((1, 1, B, 1))
                M_update[B_mean <= cloud_threshold] = 1
                M_update[:, :, :, clean_time] = 1

            rho = rho_orig * np.power((1 + alpha), iters+1)

            mse1 = torch.sum((X - Clean) ** 2)
            mse2 = torch.sum((torch.mul(X, M_update) - torch.mul(Clean, M_update))**2)
            psnr = cal_psnr(X.detach().cpu().numpy(), Clean.detach().cpu().numpy(), data_range)
            X_r = X.clone()
            X_r[M_update == 1] = Y[M_update == 1]
            psnr_r = cal_psnr(X_r.detach().cpu().numpy(), Clean.detach().cpu().numpy(), data_range)
            avg_psnr_record = avg_psnr

            all_time = [t for t in range(T)]
            cloud_time = [t  for t in all_time if t not in clean_time]
            psnr_t = [psnr[t] for t in cloud_time]
            avg_psnr = sum(psnr_t)/len(psnr_t)
            psnr_str = f"{psnr_t} "

            psnr_t_r = [psnr_r[t] for t in cloud_time]
            avg_psnr_r = sum(psnr_t_r)/len(psnr_t_r)
            # psnr_str = psnr_str + f"{psnr_t_r}"
            
            RelCha_1 = torch.norm(Last_R[:, :, :, 0] - X_r[:, :, :, 0]) / torch.norm(Last_R[:, :, :, 0])
            RelCha_2 = torch.norm(Last_R[:, :, :, 1] - X_r[:, :, :, 1]) / torch.norm(Last_R[:, :, :, 1])
            RelCha_3 = torch.norm(Last_R[:, :, :, 2] - X_r[:, :, :, 2]) / torch.norm(Last_R[:, :, :, 2])
            Last_R = X_r.clone()
            ssim = cal_ssim(X.detach().cpu().numpy(), Clean.detach().cpu().numpy(), data_range)
            ssim_r = cal_ssim(X_r.detach().cpu().numpy(), Clean.detach().cpu().numpy(), data_range)

            time_info = time.localtime(time.time())
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time_info)
            psnr_info = "iters=%d, avg_psnr=%.3f, %.3f, psnr=%s" % (iters, avg_psnr, avg_psnr_r, psnr_str)
            log_info = "%s-ssim[%.3f, %.3f]-mse[%.5E, %.5E]-loss[%.5E]-RelCha[%.3f, %.3f, %.3f]-time[%s]" % \
                        (psnr_info, np.mean(ssim), np.mean(ssim_r), mse1, mse2, loss, RelCha_1, RelCha_2, RelCha_3, cur_time)
            # 以文本日志的形式保存结果
            with open("%s/recover_log.txt" %(log_dir), "a+") as log_file:
                log_file.write(log_info + "\n")

            # 保存中间图像
            image_Clean = Clean.cpu().numpy() * scale
            image_Y = Y.cpu().numpy() * scale
            image_X = X_r.cpu().detach().numpy() * scale
            image_C = C.cpu().numpy() * scale
            image_M = M_update.cpu().numpy()
            if self.save_all_image:
                # 将恢复结果detach，在show_result函数中绘制并保存恢复图像
                show_result(image_Clean, image_X, image_Y, image_C, image_M, bands,
                    save_result_path="%s/%dth recover_image.png" % (image_save_path, iters), 
                    update_num=iters)
            if self.need_save_loss:
                # 保存本次外迭代的损失变化情况
                show_loss_history(loss_history, loss1_history, loss2_history, 
                                save_path="%s/%dth loss_image.png" % (loss_trend_path, iters))
            stop_increase += 1
            if max_psnr < avg_psnr_r:
                stop_increase = 0
                best_iters = iters
                max_psnr = avg_psnr_r
                X_best = X.clone()
                best_image_record = [image_Clean, image_X, image_Y, image_C, image_M]
                
                best_info = f"scene={self.scene_selected}, order={order:03d}, epoch_num={epoch_num:03d}, " + \
                            f"rate-lambda2-rho-alpha-threshold=" + \
                            f"{rate:.3f}, {lambda2:.3f}, {rho_orig:.5f}, {alpha:.3f}, {cloud_threshold:.5f}, " + \
                            f"epoch={epoch:03d}, {psnr_info}, avg_ssim={np.mean(ssim)}, " + \
                            f"avg_ssim_r={np.mean(ssim_r)}, ssim={ssim}, mse={mse1:.5E}, {mse2:.5E}, loss={loss:.5E}, " + \
                            f"cur_time{cur_time}"
            # 超过4次迭代psnr没有增长则终止本次实验
            # if stop_increase >= 4:
            #      break
        # 将无云区域投影回去
        X[M_update == 1] = Y[M_update == 1]

        # # 以mat格式保存本次实验的恢复结果
        if self.need_save_X:
            scio.savemat(f"{log_dir}/{self.scene_selected}_{order:03d}.mat", 
                        {f"recover_data": X.cpu().detach().numpy()})
            # scio.savemat("%s/recover_data_final.mat" % log_dir, 
            #             {"recover_data": X.cpu().detach().numpy()})
        # 读取文件内容
        while True:
            acquired = self.spin_lock.acquire(blocking=True)
            if acquired:
                with open('%s/best_record.txt' % self.root_log, 'a+') as record_file:
                    record_file.write(best_info + "\n")
                self.spin_lock.release()
                break
        if self.need_save_best_recover:
            image_Clean, image_X, image_Y, image_C, image_M = best_image_record
            show_result(image_Clean, image_X, image_Y, image_C, image_M, bands,
                        save_result_path="%s/best_recover_image.png" % log_dir, 
                        update_num=iters)
        print("\u001b[1;34m", "Done: ", log_dir, "\u001b[0m")
