# 该脚本遍历指定目录下的所有子文件夹，查找并分析 test.log 文件。
# 它分别查找“ap”和“rec_auc_all”字段的最大值，并记录到相应的日志文件。
# 若日志文件包含“Epoch 1000”，则标记训练已完成。
# 结果分别保存在 best_AP_xx.log 和 best_AUC_xx.log 文件中。

import os
import re

# 编译正则模式，用于提取数值，避免在循环中重复编译
pattern_ap = re.compile(r'\bap\b\s*[:=]\s*([0-9]*\.?[0-9]+)')
pattern_auc = re.compile(r'\brec_auc_all\b\s*[:=]\s*([0-9]*\.?[0-9]+)')
metric_label = {'ap': 'AP', 'rec_auc_all': 'AUC'}  # 文件名中的大写标签

base_path = "../output"  # 指定要遍历的根目录
for root, dirs, files in os.walk(base_path):
    for filename in files:
        if filename.endswith("test.log"):
            filepath = os.path.join(root, filename)
            # 初始化存储两个指标的最大值及相关信息
            metrics = {
                'ap': {'max': None, 'val_str': None, 'line': None},
                'rec_auc_all': {'max': None, 'val_str': None, 'line': None}
            }
            complete = False  # 标记是否包含 "Epoch 1000"
            with open(filepath, 'r') as f:
                for line in f:
                    if "Epoch 1000" in line:
                        complete = True
                    # 若该行不含任何目标字段，则跳过，提高效率
                    if 'ap' not in line and 'rec_auc_all' not in line:
                        continue
                    # 提取 ap 值并更新最大值
                    m_ap = pattern_ap.search(line)
                    if m_ap:
                        val = float(m_ap.group(1))
                        if metrics['ap']['max'] is None or val > metrics['ap']['max']:
                            metrics['ap']['max'] = val
                            metrics['ap']['val_str'] = m_ap.group(1)
                            metrics['ap']['line'] = line.strip()
                    # 提取 rec_auc_all 值并更新最大值
                    m_auc = pattern_auc.search(line)
                    if m_auc:
                        val = float(m_auc.group(1))
                        if metrics['rec_auc_all']['max'] is None or val > metrics['rec_auc_all']['max']:
                            metrics['rec_auc_all']['max'] = val
                            metrics['rec_auc_all']['val_str'] = m_auc.group(1)
                            metrics['rec_auc_all']['line'] = line.strip()
            # 根据是否训练完整设置状态标签
            status_tag = "best" if complete else "not_complete_best"
            # 分别创建保存最佳 AP 和 AUC 的日志文件
            for key, data in metrics.items():
                if data['val_str'] is not None:  # 确认该指标在日志中出现过
                    out_name = f"best_{metric_label[key]}_{data['val_str']}_{status_tag}.log"
                    out_path = os.path.join(root, out_name)
                    with open(out_path, 'w') as out_file:
                        out_file.write(data['line'] + "\n")