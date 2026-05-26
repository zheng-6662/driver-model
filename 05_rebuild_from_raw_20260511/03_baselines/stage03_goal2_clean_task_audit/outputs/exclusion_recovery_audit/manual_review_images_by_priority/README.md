# Goal2 被排除样本人工审核图片整理

- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority`
- 总样本：`1407`
- 已复制图片：`487`
- 缺少图片路径：`920`

## 分类数量

| recovery_priority         |   total_rows |   copied_images |   missing_images | folder                                                                                                                                                                                                  |
|:--------------------------|-------------:|----------------:|-----------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| B_较可能可恢复            |          265 |              69 |              196 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\01_B_较可能可恢复_看图确认  |
| C2_高度姿态重点复核       |          323 |             253 |               70 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\03_C2_高度姿态明显_谨慎复核 |
| A_优先人工恢复复核        |          792 |             163 |              629 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\00_A_优先看_旧结论可能误伤  |
| U_原因不清_需要复核       |            8 |               1 |                7 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\05_U_原因不清_需要复核      |
| D_暂不恢复_疑似路边或路外 |           16 |               1 |               15 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\04_D_暂不恢复_疑似路边路外  |
| C1_弯道高度变化重点复核   |            3 |               0 |                3 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\manual_review_images_by_priority\02_C1_弯道高度变化_重点复核 |

## 审核方式

1. 优先打开 `index.html`，再进入 A/B 分类。
2. 如果看图后认为样本可以恢复，建议在对应 `index.csv` 里填写 `manual_keep=保留`。
3. 如果认为明显下马路/路边/上斜坡，填写 `manual_keep=排除`。
4. 如果判断不清，填写 `manual_keep=不确定`，后续结合道路源文件再判断。