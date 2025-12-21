import numpy as np

data_path = "data/USB/data.npz"

data=np.load(data_path,allow_pickle=True)
print(data.files)
print("data_size:", data["fail_point_clouds"].shape[0])