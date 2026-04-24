# Codex Handoff: maintained v5.8 绗簩闃舵鏈€灏?loss 寰皟

## 浠诲姟鐩爣

璇峰湪 **涓嶆敼鏁版嵁鍒掑垎銆佷簨浠堕敋鐐广€乫uture horizon銆佹爣绛惧畾涔夈€乼eacher-state 妯″紡** 鐨勫墠鎻愪笅锛屽 maintained 涓昏缁冭剼鏈仛涓€杞?**闂鍐嶈瘖鏂?+ 鏈€灏?loss 寰皟寤鸿 + 鍗曡疆杩愯楠岃瘉**銆傚綋鍓嶉棶棰樺凡缁忎笉搴斿啀鍙弿杩颁负 tail 闂锛岃€屽簲鏀跺彛涓猴細

- early-response onset too flat
- whole-trajectory temporal dynamics over-smoothed
- tail amplitude collapse
- late-peak miss
- strong-reversal tail shrinkage

涔熷氨鏄锛岃繖涓€鐗堟ā鍨嬩笉鏄彧鍦ㄥ悗娈靛彂骞筹紝鑰屾槸 **鍓嶆鍚姩涓嶈冻銆佹暣浣撴椂闂寸粨鏋勫亸淇濆畧銆佸悗娈电户缁敹缂?*銆傜洰鏍囦笉鏄硾娉涒€滄彁鍒嗏€濓紝鑰屾槸锛?

1. 鍏堝垽鏂綋鍓嶆渶灏忔敼鍔ㄦ槸鍚︿粛搴斿彧鍔?`W_REVSEQ`
2. 鏄庣‘鍓嶆杩囧钩鏄惁闇€瑕佽繘鍏ュ浐瀹氳瘎浼伴棴鐜?
3. 鍦ㄤ繚鎸?protocol-safe 鍓嶆彁涓嬶紝浼樺厛鏀瑰杽鏃堕棿缁撴瀯锛岃€屼笉鏄彧鐩€讳綋 `rmse_steer`

濡傛灉浣犺涓哄彧鍋?`W_REVSEQ` 寰皟涓嶈冻浠ヨ鐩栧綋鍓嶉棶棰橈紝璇峰厛缁欏嚭鎬濊€冪粨璁猴紝鍐嶅喅瀹氭槸鍚﹂渶瑕?very small 鐨?onset/head 绾︽潫鎴栬瘎浼拌ˉ鍏呫€?

---

## 褰撳墠鍩虹嚎璇佹嵁

鍙傝€冭繖杞?smoke 杩愯鐩綍锛?

- `F:/data_set_process/data_process/03_results/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726`

鍏抽敭璇婃柇锛堝凡鐢辨柊澧炶瘎浼伴棴鐜ǔ瀹氬鍑猴級锛?

- `tail_flatness_rate = 0.961`
- `tail_amp_ratio_pred_over_gt = 0.156`
- `late_peak_recall = 0.145`
- `strong_pos.tail_amp_ratio_pred_over_gt = 0.065`

杩欒鏄庡綋鍓嶄富闂涓嶆槸鈥滃畬鍏ㄤ笉浼氬姩鈥濓紝鑰屾槸锛氬熬娈典粛涓ラ噸鏀剁缉銆佸悗鍗婃宄板€间繚涓嶄綇銆乻trong reversal hard case 灏炬鍑犱箮琚帇鎵併€?

鍙﹀锛屽熀浜庢渶鏂颁汉宸ョ湅鍥惧弽棣堬紙鍚屼竴杞?`2026-04-15 11:57` smoke 鍥撅級锛岄渶瑕佽ˉ鍏呬竴涓鍓嶆寚鏍囧皻鏈崟鐙鐩栫殑闂锛?

- 鍓嶆涔熸槑鏄捐繃骞筹紝棰勬祴鍚姩鍋忓急锛屾棭鏈熷搷搴斿箙鍊间笉瓒?

鍥犳 Codex 闇€瑕佹妸杩欒疆闂瑙嗕负 **head + tail 涓ょ閮借鍘嬬缉锛屼笖鏁翠綋鏃堕棿鍔ㄦ€佸亸淇濆畧**锛岃€屼笉鏄彧鎶婂畠褰撲綔鍗曠函 tail flattening銆?

璇风壒鍒€濊€冿細

- 鐜版湁 tail / peak / reversal 鎸囨爣鏄惁瓒充互瑙ｉ噴鍓嶆闂
- 鏄惁搴旀柊澧?head/onset 璇勪及鎸囨爣锛堝 `head_amp_ratio_pred_over_gt`銆乣head_flatness_rate`銆乣response_onset_delay`銆乣early_slope_ratio_pred_over_gt`锛?
- 鍦ㄨ缁冩渶灏忔敼鍔ㄤ笂锛屽簲鍏堝彧鍋?`W_REVSEQ` 褰掑洜锛岃繕鏄渶瑕佷竴涓?very small 鐨?early-response 绾︽潫閰嶅

---

## Codex 鏈疆鎬濊€冮噸鐐?

鍦ㄧ湡姝ｄ慨鏀瑰墠锛岃鍏堟槑纭洖绛斾互涓嬮棶棰橈細

1. 褰撳墠鏈€灏忚缁冩敼鍔ㄦ槸鍚︿粛搴斾紭鍏堝彧鍔?`W_REVSEQ`锛岃繕鏄繖浼氳繃搴﹁仛鐒﹀悗娈甸棶棰橈紵
2. 鈥滃墠娈典篃杩囧钩鈥濇槸鍚﹁鏄庡浐瀹氳瘎浼伴棴鐜噷缂哄皯 head/onset 鎸囨爣锛?
3. 濡傛灉瑕佺户缁潥鎸佹渶灏忓共棰勶紝鏈€鍚堢悊鐨勯『搴忔槸锛?
   - 鍏堣ˉ璇勪及锛屼笉鍔ㄨ缁?
   - 鍏堝彧鍔?`W_REVSEQ`
   - 鍏堝姩 `W_REVSEQ`锛屽啀 very small 鍦拌ˉ涓€涓?onset/head 绾︽潫
4. 浣犳帹鑽愮殑鏈€灏忔柟妗堜负浠€涔堟渶鍒╀簬褰掑洜锛岃€屼笉鏄彧鐪嬭捣鏉ユ洿鍏ㄩ潰锛?

璇锋妸鈥滃彲瑙ｉ噴銆佸彲褰掑洜銆佸敖閲忓皯鏀?active source鈥濅綔涓虹涓€鍘熷垯銆?

## 鎺ㄨ崘鏀瑰姩锛堟寜浼樺厛绾э級

### 璁粌渚ч粯璁ゆ渶灏忔柟妗堬細鍏堝彧鍔?`W_REVSEQ`

鍦ㄤ互涓嬫枃浠朵腑淇敼锛?

- `F:/data_set_process/data_process/02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`

褰撳墠鑴氭湰閲岋細

- `W_REVSEQ = 0.0`
- `W_PEAKTIME = 0.05`

### 鍗曡疆鎺ㄨ崘鏂规

鍙仛涓€妗ｆ渶灏忔敼鍔細

- `W_REVSEQ: 0.00 -> 0.05`

### 鏈疆涓嶈鍔?

- 涓嶈鏀规ā鍨嬬粨鏋?
- 涓嶈鏀硅緭鍏ョ壒寰?
- 涓嶈鏀?protocol / split / anchor / future length
- 涓嶈鏀?teacher-state mode
- 涓嶈鍚屾椂寮曞叆澶氬澶ф潈閲嶈皟鏁?

### 鍙€変絾涓嶄紭鍏?

濡傛灉浣犲垽鏂彧寮€ `W_REVSEQ` 杩囦簬淇濆畧锛屽彲浠?very small 鍦伴檮甯︼細

- `W_PEAKTIME: 0.05 -> 0.08`

浣嗛粯璁や粛寤鸿 **鍏堝彧鍔?`W_REVSEQ`**锛岃繖鏍锋洿鍒╀簬褰掑洜銆?

---

## 鍏抽敭鏂囦欢

### 闇€瑕佽鍙?/ 淇敼

- `F:/data_set_process/data_process/02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`

### 闇€瑕佽鍙栫殑鍩虹嚎缁撴灉

- `F:/data_set_process/data_process/03_results/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/test_metrics.json`
- `F:/data_set_process/data_process/03_results/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/test_metrics_tail.json`
- `F:/data_set_process/data_process/03_results/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/test_metrics_peak.json`
- `F:/data_set_process/data_process/03_results/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260415_115726/figures/test_metrics_reversal_structure.json`

### 鏃ュ織鍗忚鍙傝€?

- `F:/data_set_process/data_process/04_project_logs/reports/progress/ai_recording_protocol.md`
- `F:/data_set_process/data_process/04_project_logs/reports/progress/daily/2026-04-15.md`
- `F:/data_set_process/data_process/04_project_logs/reports/progress/experiment_registry.md`

**娉ㄦ剰锛氫笉瑕佺紪杈?tmp 杩愯鐩綍閲岀殑鑴氭湰鍓湰銆傚彧鏀?active source銆?*

---

## 寤鸿杩愯鍛戒护

浣跨敤鏈満瀹為檯鍙敤瑙ｉ噴鍣ㄨ矾寰勶紝涓嶈渚濊禆 shell 渚?`conda run`锛?

```bash
CUDA_VISIBLE_DEVICES=0 DRIVER_MODEL_RESULT_ROOT="F:/data_set_process/data_process/03_results/tmp/protocol_safe_runs" DRIVER_MODEL_SMOKE=1 DRIVER_MODEL_SMOKE_MAX_SAMPLES=512 DRIVER_MODEL_SMOKE_EPOCHS=2 DRIVER_MODEL_SMOKE_BATCH_SIZE=64 "D:/ProgramData/anaconda3/envs/predict_2/python.exe" "F:/data_set_process/data_process/02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py"
```

濡?smoke 鎸囨爣鏂瑰悜姝ｇ‘锛屽啀鍐冲畾鏄惁琛ユ寮?run銆?

---

## 楠屾敹鏍囧噯

浼樺厛鐪嬫柊澧?JSON 瀛楁锛岃€屼笉鏄彧鐪嬫€讳綋 RMSE銆?

鍚屾椂璇锋敞鎰忥細濡傛灉鏈疆浣犲垽鏂€滃墠娈佃繃骞斥€濆凡缁忚冻澶熸槑纭紝鑰岀幇鏈?JSON 鏃犳硶閲忓寲瀹冿紝閭ｄ箞涓€涓悎鏍艰緭鍑轰笉涓€瀹氬彧鑳芥槸璁粌缁撴灉锛屼篃鍙互鏄細

- 鏄庣‘璇存槑鐜版湁璇勪及闂幆瀵?head/onset 闂瑕嗙洊涓嶈冻
- 缁欏嚭鏈€灏忔柊澧炶瘎浼版寚鏍囨柟妗?
- 璇存槑涓轰粈涔堝綋鍓嶅簲鍏堣ˉ璇勪及鍐嶅仛璁粌褰掑洜锛屾垨涓轰粈涔堢浉鍙?

涔熷氨鏄锛屾湰杞獙鏀朵笉鍙帴鍙椻€滆窇鍑轰竴涓柊 smoke鈥濓紝涔熸帴鍙椻€滃厛鎶婇棶棰樺畾涔夊拰璇勪及缂哄彛鏀舵竻妤氣€濈殑楂樿川閲忕粨璁恒€?

### head / onset锛堝浣犲喅瀹氳ˉ璇勪及锛?
寤鸿浼樺厛鐪嬶細

- `head_amp_ratio_pred_over_gt`
- `head_flatness_rate`
- `response_onset_delay`
- `early_slope_ratio_pred_over_gt`

涓嶈姹傚洓涓兘瀹炵幇锛屼絾鑷冲皯瑕佺粰鍑烘槸鍚﹀€煎緱琛ャ€佷负浠€涔堛€佷互鍙婃渶灏忓疄鐜拌矾寰勩€?

### tail
鐪嬶細

- `test_metrics_tail.json`
  - `tail_amp_ratio_pred_over_gt` 瑕?**楂樹簬** `0.156`
  - `tail_flatness_rate` 瑕?**浣庝簬** `0.961`

### peak
鐪嬶細

- `test_metrics_peak.json`
  - `late_peak_recall` 瑕?**楂樹簬** `0.145`

### reversal structure
鐪嬶細

- `test_metrics_reversal_structure.json`
  - `strong_pos.tail_amp_ratio_pred_over_gt` 瑕?**楂樹簬** `0.065`
  - `strong_pos.tail_flatness_rate` 鏈€濂戒笉瑕佹伓鍖?

### overall regression
鐪嬶細

- `test_metrics.json`
  - `rmse_steer` 涓嶅簲鏄庢樉鎭跺寲
  - 濡傛灉缁撴瀯鎸囨爣鏄庢樉鏀瑰杽銆丷MSE 浠呰交寰尝鍔紝鍙帴鍙?

### tail
鐪嬶細

- `test_metrics_tail.json`
  - `tail_amp_ratio_pred_over_gt` 瑕?**楂樹簬** `0.156`
  - `tail_flatness_rate` 瑕?**浣庝簬** `0.961`

### peak
鐪嬶細

- `test_metrics_peak.json`
  - `late_peak_recall` 瑕?**楂樹簬** `0.145`

### reversal structure
鐪嬶細

- `test_metrics_reversal_structure.json`
  - `strong_pos.tail_amp_ratio_pred_over_gt` 瑕?**楂樹簬** `0.065`
  - `strong_pos.tail_flatness_rate` 鏈€濂戒笉瑕佹伓鍖?

### overall regression
鐪嬶細

- `test_metrics.json`
  - `rmse_steer` 涓嶅簲鏄庢樉鎭跺寲
  - 濡傛灉缁撴瀯鎸囨爣鏄庢樉鏀瑰杽銆丷MSE 浠呰交寰尝鍔紝鍙帴鍙?

---

## 鏃ュ織璁板綍瑕佹眰锛堝繀椤婚伒瀹堟柊鍗忚锛?

### 涓嶈榛樿鍐欏洖

- `F:/data_set_process/data_process/04_project_logs/reports/project_progress_master.md`

### 鏈疆蹇呴』鍏堝啓

- `F:/data_set_process/data_process/04_project_logs/reports/progress/daily/2026-04-15.md`

### 濡傛灉褰㈡垚鏂?run 缁撹锛屽啀琛?

- `F:/data_set_process/data_process/04_project_logs/reports/progress/experiment_registry.md`

### 璁板綍鏈€浣庤姹?

- 鎵ц涓讳綋
- Why
- 涓撲笟缁撹
- 鐧借瘽瑙ｉ噴
- 鍋氫簡浠€涔?
- 浜х墿 / 閾炬帴
- 涓嬩竴姝?

涓嶈閲嶅鎵嬫妱 `run_summary.json` 鎴?JSON 閲屼竴闀夸覆鑷姩鍙緱鎸囨爣锛涗汉宸ヨ褰曢噸鐐瑰啓鍒ゆ柇涓庣櫧璇濊В閲娿€?

---

## 鍗曡疆鏈€鎺ㄨ崘鏂规鎬荤粨

濡傛灉鍙仛涓€杞紝璇蜂紭鍏堬細

- **鍙敼 `W_REVSEQ: 0.00 -> 0.05`**
- 璺戜竴杞?smoke
- 鐢ㄦ柊澧炵殑 tail / peak / reversal 缁撴瀯鎸囨爣鍋氬鐓ч獙鏀?

杩欐槸涓€杞渶绗﹀悎鈥滄渶灏忓共棰勩€佺洿鎸囧綋鍓?hard case銆佷究浜庡綊鍥犫€濈殑鏂规銆?

