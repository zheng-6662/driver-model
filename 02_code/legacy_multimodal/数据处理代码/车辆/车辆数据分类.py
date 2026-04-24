

import os
import shutil
from glob import glob

# 多模态被试集合根目录
ROOT = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合"

# 被试列表（你可以继续加）
subjects = ["byx","cwh","gf","gzj","hzh","jy",
            "lx","lxy","rjy","txj","tyy","xst",
            "yyl","yzy","zdq","zt","zx","zxy"]

# 精准匹配的文件名模式
VEHICLE_PATTERN = "*_vehicle_aligned_cleaned.csv"
EVENT_PATTERN   = "*_vehicle_aligned_cleaned_events_v312.csv"

print("\n========== 开始整理车辆文件（车辆 → vehicle，事件 → event） ==========\n")

for sub in subjects:
    sub_dir = os.path.join(ROOT, sub)

    if not os.path.exists(sub_dir):
        print(f"⚠ 被试 {sub} 文件夹不存在，跳过")
        continue

    # 找车辆文件
    vehicle_files = glob(os.path.join(sub_dir, VEHICLE_PATTERN))
    event_files   = glob(os.path.join(sub_dir, EVENT_PATTERN))

    # === 车辆文件 → vehicle 文件夹 ===
    if vehicle_files:
        dst_vehicle = os.path.join(sub_dir, "vehicle")
        os.makedirs(dst_vehicle, exist_ok=True)

        for f in vehicle_files:
            shutil.move(f, dst_vehicle)
            print(f"✓ 车辆文件已移动：{os.path.basename(f)} → {sub}/vehicle")

    # === 事件文件 → event 文件夹 ===
    if event_files:
        dst_event = os.path.join(sub_dir, "event")
        os.makedirs(dst_event, exist_ok=True)

        for f in event_files:
            shutil.move(f, dst_event)
            print(f"✓ 事件文件已移动：{os.path.basename(f)} → {sub}/event")

    if not vehicle_files and not event_files:
        print(f"⚠ 被试 {sub} 未找到车辆对齐文件或事件文件")

print("\n🎉 车辆对齐文件和事件文件均已按类型整理完毕！")
