# Goal2 clean vehicle-only 任务审计报告

## 这次做了什么

本轮修正 goal1 中的四类问题：严格排除斜坡/路边/下马路/明显高度异常样本；删除后验峰值输入；补全中文 anchor_quality 的质量标记；建立锚点审计和共同评价集。当前仍然只做 vehicle-only，不加入连续驾驶风格、生理、脑电或教师蒸馏。

## 样本排除结果

- goal1 原 excluded_slope_or_offroad：`162`
- goal2 严格 excluded_slope_or_offroad：`1407`
- goal2 strict clean train：`129`
- common eval：`{'common_eval_all_clean': 10, 'common_eval_noncurve_clean': 8, 'common_eval_curve_clean': 2}`

```csv
goal1_role,goal1_count,goal2_newly_strict_excluded
excluded_slope_or_offroad,162,162
main_train,746,672
curve_task,238,183
aux_train,285,193

```

## 锚点审计

- 全部样本 anchor_late 比例：`0.5793`

```csv
split,anchor_late_rate
test,0.6304347826086957
train,0.5731814198071867
val,0.5587392550143266

```

```csv
response_type,anchor_late_rate
brake_dominant,0.5274725274725275
conservative_pass,1.0
no_clear_response,0.786833855799373
strong_steer,0.5038826574633305
unknown,0.75
vehicle_dominant_no_clear_action,1.0
weak_steer,0.863013698630137

```

## 实验结果

```csv
experiment,name_cn,split,train,val,test,window_incomplete_used,steering_rmse,wrong_side_rate,severe_under_amplitude_rate,large_response_recall,response_type_macro_f1,curve_type_macro_f1,keypoint_brake_onset_time_mae,keypoint_speed_drop_max_mae,keypoint_roll_peak_value_mae,keypoint_roll_rate_peak_value_mae,figure_count
G2_E0_clean_steering_only,clean 固定窗口方向盘-only 对照,train,73,22,12,0,0.429052323102951,0.0526315789473684,0.4210526315789473,0.8947368421052632,,,,,,,70
G2_E0_clean_steering_only,clean 固定窗口方向盘-only 对照,val,73,22,12,0,0.4768257737159729,0.75,0.75,1.0,,,,,,,70
G2_E0_clean_steering_only,clean 固定窗口方向盘-only 对照,test,73,22,12,0,0.2974657714366913,,,,,,,,,,70
G2_E0_clean_steering_only,clean 固定窗口方向盘-only 对照,common_eval_all_clean,73,22,12,0,0.3125894367694855,,,,,,,,,,70
G2_E0_clean_steering_only,clean 固定窗口方向盘-only 对照,common_eval_noncurve_clean,73,22,12,0,0.314454585313797,,,,,,,,,,70
G2_E1_clean_multitask,clean 固定窗口多输出,train,73,22,12,0,0.4714973568916321,0.1052631578947368,0.4736842105263157,1.0,0.8975638740344624,,0.2958246171474457,5.779348373413086,0.0194636676460504,0.1273598074913025,140
G2_E1_clean_multitask,clean 固定窗口多输出,val,73,22,12,0,0.4878602921962738,0.75,0.75,1.0,0.4435286935286935,,0.2401642799377441,12.71570873260498,0.0405776351690292,0.2014152854681015,140
G2_E1_clean_multitask,clean 固定窗口多输出,test,73,22,12,0,0.306748628616333,,,,0.2358974358974359,,0.0700172409415245,8.430173873901367,0.0275475401431322,0.2000079751014709,140
G2_E1_clean_multitask,clean 固定窗口多输出,common_eval_all_clean,73,22,12,0,0.317192941904068,,,,0.1809523809523809,,0.6281313300132751,21.81005859375,0.0230003986507654,0.1585425436496734,140
G2_E1_clean_multitask,clean 固定窗口多输出,common_eval_noncurve_clean,73,22,12,0,0.3249682188034057,,,,0.1428571428571428,,0.6137567758560181,21.81005859375,0.028278611600399,0.1519568860530853,140
G2_E2_clean_masked_multihorizon,clean 掩码多时域多输出,train,128,33,32,9,0.4775391817092895,0.125,0.5,0.8125,0.8494367338404035,,0.3203192055225372,10.345805168151855,0.0197763107717037,0.1109934598207473,245
G2_E2_clean_masked_multihorizon,clean 掩码多时域多输出,val,128,33,32,9,0.4362054169178009,0.25,0.375,0.75,0.425050505050505,,0.3507179319858551,13.38133716583252,0.0352324768900871,0.1802796721458435,245
G2_E2_clean_masked_multihorizon,clean 掩码多时域多输出,test,128,33,32,9,0.2385756224393844,,,,0.6509009009009009,,0.4328015744686126,18.00038719177246,0.0150313898921012,0.1068217754364013,245
G2_E2_clean_masked_multihorizon,clean 掩码多时域多输出,common_eval_all_clean,128,33,32,9,0.2808715999126434,,,,0.225,,0.6715350151062012,21.309545516967773,0.0166254434734582,0.101095899939537,245
G2_E2_clean_masked_multihorizon,clean 掩码多时域多输出,common_eval_noncurve_clean,128,33,32,9,0.2800171375274658,,,,0.3571428571428571,,0.6981993913650513,21.309545516967773,0.019421262666583,0.1158851459622383,245
G2_E3_noncurve_response_aux,非弯道 response_type 辅助任务,train,168,43,45,9,0.2790415585041046,0.0714285714285714,0.3095238095238095,0.9523809523809524,0.7501119570085087,,0.6256277561187744,9.678695678710938,0.0158750042319297,0.0966676324605941,256
G2_E3_noncurve_response_aux,非弯道 response_type 辅助任务,val,168,43,45,9,0.3915565013885498,0.1333333333333333,0.5333333333333333,0.8,0.4722222222222221,,0.7652098536491394,9.794322967529297,0.0320223197340965,0.150846853852272,256
G2_E3_noncurve_response_aux,非弯道 response_type 辅助任务,test,168,43,45,9,0.1868976205587387,0.0,1.0,0.0,0.1637303265210241,,0.7583991885185242,15.789175987243652,0.0151124764233827,0.0875723659992218,256
G2_E3_noncurve_response_aux,非弯道 response_type 辅助任务,common_eval_all_clean,168,43,45,9,0.2515735328197479,,,,0.1333333333333333,,1.0599815845489502,25.358030319213867,0.0262300018221139,0.0912976786494255,256
G2_E3_noncurve_response_aux,非弯道 response_type 辅助任务,common_eval_noncurve_clean,168,43,45,9,0.2426314204931259,,,,0.1944444444444444,,1.200716495513916,25.358030319213867,0.0292102638632059,0.1000321507453918,256
G2_E4_curve_clean_specialized,clean 弯道专门任务,train,45,13,7,0,0.5284529328346252,0.0833333333333333,0.1666666666666666,1.0,,1.0,0.1151635572314262,3.631789207458496,0.0167859848588705,0.0748621895909309,93
G2_E4_curve_clean_specialized,clean 弯道专门任务,val,45,13,7,0,0.3326899111270904,0.0,1.0,1.0,,0.4347826086956521,0.1884002238512039,11.27505111694336,0.027842117473483,0.1062460467219352,93
G2_E4_curve_clean_specialized,clean 弯道专门任务,test,45,13,7,0,0.3321079611778259,,,,,0.4615384615384615,0.3180020153522491,12.626631736755373,0.0172501541674137,0.0909961387515068,93
G2_E4_curve_clean_specialized,clean 弯道专门任务,common_eval_curve_clean,45,13,7,0,0.4170728027820587,,,,,1.0,0.5952162146568298,,0.0178404338657856,0.0503414422273635,93
G2_E5A_train_candidates_only,E5A 只用训练候选,train,73,22,13,1,0.4069343209266662,0.0526315789473684,0.3157894736842105,0.8421052631578947,0.93160591421461,,0.2611039876937866,4.892523765563965,0.0191527679562568,0.1298446059226989,149
G2_E5A_train_candidates_only,E5A 只用训练候选,val,73,22,13,1,0.4736529588699341,0.5,0.75,0.75,0.5746934225195095,,0.3312419950962066,10.335997581481934,0.0363073721528053,0.2162459045648574,149
G2_E5A_train_candidates_only,E5A 只用训练候选,test,73,22,13,1,0.2803073525428772,,,,0.4481481481481482,,0.6132816672325134,6.439138889312744,0.0208220388740301,0.2040323913097381,149
G2_E5A_train_candidates_only,E5A 只用训练候选,common_eval_all_clean,73,22,13,1,0.3127700090408325,,,,0.08,,0.711390495300293,22.49225425720215,0.0206758882850408,0.1393029987812042,149
G2_E5A_train_candidates_only,E5A 只用训练候选,common_eval_noncurve_clean,73,22,13,1,0.3240018486976623,,,,0.0625,,0.6694855690002441,22.49225425720215,0.0230928137898445,0.1668668538331985,149
G2_E5B_train_plus_all_clean_review,E5B 训练候选 + 全部 clean 待复核,train,130,33,35,9,0.4741095006465912,0.1818181818181818,0.3939393939393939,0.8787878787878788,0.8054087438997225,,0.310873419046402,11.183025360107422,0.0193885583430528,0.1203312650322914,247
G2_E5B_train_plus_all_clean_review,E5B 训练候选 + 全部 clean 待复核,val,130,33,35,9,0.445869117975235,0.5,0.375,0.75,0.3433855799373041,,0.2591294944286346,11.496590614318848,0.0328028425574302,0.1575167030096054,247
G2_E5B_train_plus_all_clean_review,E5B 训练候选 + 全部 clean 待复核,test,130,33,35,9,0.2471386939287185,,,,0.4278388278388278,,0.3937012255191803,17.635330200195312,0.0146734779700636,0.1414289474487304,247
G2_E5B_train_plus_all_clean_review,E5B 训练候选 + 全部 clean 待复核,common_eval_all_clean,130,33,35,9,0.3019939064979553,,,,0.3,,0.6287043690681458,19.68238639831543,0.0178497545421123,0.1286058723926544,247
G2_E5B_train_plus_all_clean_review,E5B 训练候选 + 全部 clean 待复核,common_eval_noncurve_clean,130,33,35,9,0.2973915636539459,,,,0.3541666666666666,,0.6469151377677917,19.68238639831543,0.0195890516042709,0.1493992209434509,247
G2_E5C_train_plus_stratified_clean_review,E5C 训练候选 + 分层 clean 待复核,train,128,33,32,9,0.4137186110019684,0.09375,0.375,0.875,0.8748935717326521,,0.2722551822662353,10.436755180358888,0.0183437187224626,0.1139474213123321,245
G2_E5C_train_plus_stratified_clean_review,E5C 训练候选 + 分层 clean 待复核,val,128,33,32,9,0.4251865148544311,0.375,0.375,0.875,0.4527969348659004,,0.3462284505367279,11.44285488128662,0.0309467278420925,0.1946917921304702,245
G2_E5C_train_plus_stratified_clean_review,E5C 训练候选 + 分层 clean 待复核,test,128,33,32,9,0.2323224395513534,,,,0.4781209781209781,,0.4615058004856109,19.02925491333008,0.0226852670311927,0.1460724920034408,245
G2_E5C_train_plus_stratified_clean_review,E5C 训练候选 + 分层 clean 待复核,common_eval_all_clean,128,33,32,9,0.3100982904434204,,,,0.2833333333333333,,0.6425780057907104,20.638019561767575,0.0171110518276691,0.1269902437925338,245
G2_E5C_train_plus_stratified_clean_review,E5C 训练候选 + 分层 clean 待复核,common_eval_noncurve_clean,128,33,32,9,0.2756121456623077,,,,0.4404761904761904,,0.704871416091919,20.638019561767575,0.0195126384496688,0.1552466452121734,245

```

## 对 Goal2 问题的回答

1. goal1 中误留的明显高度异常样本见 `manifests/strict_exclusion_summary_goal2.csv`，已从所有主训练实验排除。
2. goal2 严格排除后，clean 样本以 `manifest_strict_clean_train.csv`、`manifest_strict_clean_noncurve.csv`、`manifest_strict_clean_curve.csv` 为准。
3. 后验峰值输入已删除，当前输入特征见 `outputs/feature_list_goal2.csv`，删除清单见 `leakage_removed_feature_list.txt`。
4. 当前窗口不完整样本很少，主要瓶颈更可能是 anchor_late 和样本语义，而不是窗口缺失。
5. 锚点偏晚比例见 `manifest_anchor_audit_goal2.csv` 和上面的 split/response_type 统计；如果 test 比 train/val 高，测试指标需要谨慎解释。
6. masked multi-horizon 是否有价值，应看 G2_E2 在 common_eval_all_clean 上的指标，而不是只看自身 test。
7. 非弯道 response_type 当前是辅助头，不是真正分支头；报告名称使用 response_aux。
8. 弯道 clean specialized 当前训练候选数为 `66`，如果 abnormal_roll 数量偏少，E4 只能作为诊断。
9. 待复核样本价值看 G2_E5A/B/C 在 common_eval_noncurve_clean 和 common_eval_all_clean 上的同口径比较。
10. 预测图按 steering/speed/brake/roll/yaw 分别输出 worst case，不能再被 speed 总 RMSE 支配。
11. 是否进入风格/生理阶段，要等 clean vehicle-only 的锚点和预测图稳定后再判断。

## 产物位置

- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\manifests`
- 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs`
- 最终报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\final_goal2_clean_task_audit_report.md`