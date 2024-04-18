import os
paths = os.listdir(path="datasets/corrosion_dataset/mask")

for path in paths:
  substring = path[:5]
  print(substring)
  os.rename(f"datasets/corrosion_dataset/mask/{path}", f"datasets/corrosion_dataset/mask/{substring}.png")
