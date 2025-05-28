from matplotlib import pyplot as plt
from tools import normal
import scipy.io as scio

def load_dataset(data_name, time_num=None):
    data = scio.loadmat(f"./dataset/{data_name}.mat")
    
    bands, data_range = [2, 1, 0], 1
    if time_num is None:
        _, _, _, time_num = data["Noisy"].shape
    Noisy = data["Noisy"][:, :, :, :time_num]
    Mask = data["Mask"][:, :, :, :time_num]

    M, N, B, T = Noisy.shape
    for t in range(T):
        Noisy[:, :, :, t] = normal(Noisy[:, :, :, t])
    scene_list = {"city_time_num[12]": [8,9,10,11], "city_time_num[3]": [2], 
                  "city_time_num[6]": [4, 5], "city_time_num[9]": [6, 7, 8], 
                  "framland_band_num[10]": [3, 4, 5], "framland_band_num[4]": [3, 4, 5], 
                  "framland_band_num[6]": [3, 4, 5], "framland_band_num[8]": [3, 4, 5],
                  "mountain_cover_rank[1]": [3, 4, 5], "mountain_cover_rank[2]": [3, 4, 5], 
                  "mountain_cover_rank[3]": [3, 4, 5], "ROIs1970_112_120": [3]}
    if data_name in scene_list.keys():
        clean_time = scene_list[data_name]
    else:
        clean_time = [t for t in range(time_num) if t not in [0, 1, 2]]
    scale = 1
    if data_name not in ["ROIs1970_112_120"]:
        Clean = data["Clean"][:, :, :, :time_num]
        return Clean, Noisy, Mask, clean_time, bands, data_range, scale
    else:
        return Noisy, Mask, clean_time, bands, data_range, scale