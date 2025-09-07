import numpy as np

PATH = r"C:/Users/84332/Desktop/AnhKiet hoc tap/CDCSAI/CDCSAI/features/3637013_c675de7705.jpg.npy"
feat = np.load(PATH)
print("Feature shape:", feat.shape)  # Expect: (2048,)
