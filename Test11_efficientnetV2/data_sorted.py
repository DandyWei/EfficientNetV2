from pathlib import Path
import os, shutil
dataset_dir = Path("D:/Datasets/Effi")
"""
str len 27
"""

# s_str = set([d for d in dataset_dir.glob("14/*.npy") if len(str(d.stem).split("_"))==6])
# for p in s_str:
#     break
# ppp = Path("D:/Datasets/test") /Path(str(19)) /p.name
#
# len(s_str)
# for d in dataset_dir.glob("14/*.npy"):
#     print(d)
#     break
#
# str(d.stem).split("_")

def get_test_data():
    """
    10 % data for test
    """
    for i in range(20):
        num = str(i).zfill(2)
        s_str = set([d for d in dataset_dir.glob(f"{num}/*.npy") if len(str(d.stem).split("_"))==6])
        test_num = round(len(s_str) * 0.1)
        dst_dir = Path("D:/Datasets/test") / Path(num)
        os.mkdir(dst_dir) if not os.path.isdir(dst_dir) else None
        for src in list(s_str)[:test_num]:
            dst = dst_dir / src.name
            print(f"from {src} to {dst}")
            shutil.move(src, dst)
#擴超慢 刪他媽快
def clean_argment():
    for d in dataset_dir.glob("*/*.npy"):
        if len(str(d.stem).split("_"))!=6:
            os.remove(d)
            print(d.name)


if __name__ == "__main__":
    get_test_data()
