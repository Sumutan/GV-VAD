"""
有视频异常描述文件anomaly descriptions.txt的组织格式如下：
序号  事件编号    生成的异常事件描述   对应场景下正常事件描述
1   1   In a bustling city street, a man in a hoodie is seen stealing a purse from an unsuspecting woman, quickly fleeing the scene.    On a city street, people are walking by, chatting, and going about their daily routines without any criminal activity.
2   1   At a busy subway station, a group of teenagers are involved in a physical altercation, pushing and shoving each other, causing a commotion.   At a subway station, commuters are calmly waiting for their trains, reading, or listening to music without any disturbances.
3   1   In a quiet residential neighborhood, a person is seen scaling a fence and entering a property, clearly trespassing.   In a residential area, neighbors are gardening, walking their dogs, and children are playing in their yards without any signs of intrusion.
......

将上面的异常描述文件转化为同名的csv文件，也可以将csv 文件转化为 csv 文件。
"""

import csv
import os
import re


class AnomalyFileConverter:
    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def split_line_into_parts(line):
        """
        将输入的行分割为四个部分。
        参数:
            line (str): 需要分割的字符串。
        返回:
            list: 分割后的四个部分（如果分割成功）；否则返回空列表。
        """
        if not line.strip():  # 检查是否是空行
            return []

        # 使用正则表达式分割两个或更多空格
        # parts = re.split(r'\s{2,}', line.strip())
        # 修改分割规则：前两个部分用单个或多个空格分割，后两个部分用两个或更多空格分割
        pattern = r'^(\d+)\s+(\d+)\s{2,}(.*?)\s{2,}(.*)$'
        match = re.match(pattern, line.strip())

        # 检查是否分割为四部分

        if match:
            # 如果匹配成功，返回分组结果
            return [match.group(1), match.group(2), match.group(3), match.group(4)]
        else:
            return []  # 返回空列表表示分割失败

    def txt_to_csv(self):
        """
        将异常描述的 txt 文件转换为 CSV 文件
        """
        csv_file_path = os.path.splitext(self.file_path)[0] + '.csv'

        with open(self.file_path, 'r') as txt_file, open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            reader = txt_file.readlines()
            writer = csv.writer(csv_file)

            # 写入表头
            writer.writerow(["序号", "事件编号", "生成的异常事件描述", "对应场景下正常事件描述"])

            # 解析并写入内容
            for line in reader:
                if line.strip():  # 跳过空行
                    parts = self.split_line_into_parts(line)
                    if len(parts) == 4:
                        writer.writerow(parts)
                    else:
                        print(f"警告: 跳过格式错误的行 -> {line.strip()}")

        print(f"TXT 文件已成功转换为 CSV 文件: {csv_file_path}")

    def csv_to_txt(self):
        """
        将异常描述的 CSV 文件转换为 txt 文件
        """
        txt_file_path = os.path.splitext(self.file_path)[0] + '_converted.txt'

        with open(self.file_path, 'r', encoding='utf-8') as csv_file, open(txt_file_path, 'w',
                                                                           encoding='utf-8') as txt_file:
            reader = csv.reader(csv_file)
            rows = list(reader)

            # 写入表头
            txt_file.write('\t'.join(rows[0]) + '\n')

            # 写入内容
            for row in rows[1:]:
                txt_file.write('\t'.join(row) + '\n')

        print(f"CSV 文件已成功转换为 TXT 文件: {txt_file_path}")


# 示例用法
if __name__ == "__main__":
    # 初始化转换器
    file_path = "kimi_fix_CN.txt"  # 替换为实际文件路径
    converter = AnomalyFileConverter(file_path)

    # 将 txt 转换为 csv
    converter.txt_to_csv()

    # 将 csv 转换为 txt
    # csv_file_path = "kimi_fix_CN.csv"  # 替换为实际文件路径
    # converter = AnomalyFileConverter(csv_file_path)
    # converter.csv_to_txt()