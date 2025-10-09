"""
VideoDescriptionsList.txt中每一行存放着异常视频描述。
API_access.py和https://bigmodel.cn/dev/api/videomodel/cogvideox中记录了文生视频 API 的使用示例。
通过 API 调用和VideoDescriptionsList.txt中的视频描述生成视频。
"""
import time
import logging
from zhipuai import ZhipuAI
from util.download_urlVideo import download_video
import os

# 初始化 API 客户端
client = ZhipuAI(api_key="5954cab1423165036aec35346b623907.NTEh4jhuqYX1nuaP")  # 替换为你的 API Key

# 配置生成参数
MODEL_NAME = "cogvideox"
VIDEO_SIZE = "720x480"
VIDEO_DURATION = 10  # 视频时长 (秒)

# 文件路径
DESCRIPTIONS_FILE = "Video_Discribtions/description_1_TAD/GPT_CN_line.csv"
OUTPUT_DIR = "GeneratedVideos/description_1_TAD"
START_LINE = 382  # 从第几行开始生成视频
LOG_FILE = "ErrorLog.log"  # 错误日志文件

# 配置日志
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,  # 只记录 ERROR 及以上级别的日志
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a",
)

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 读取描述文件
with open(DESCRIPTIONS_FILE, "r", encoding="utf-8") as file:
    descriptions = file.readlines()
print(f"find {len(descriptions)} descriptions")

# 遍历每条描述生成视频
for idx, prompt in enumerate(descriptions, start=1):
    if idx<START_LINE: continue
    prompt = prompt.strip().strip('"')
    if not prompt:
        continue

    print(f"Processing video {idx}: {prompt}")

    try:
        # 生成视频请求
        response = client.videos.generations(
            model=MODEL_NAME,
            prompt=prompt,
            size=VIDEO_SIZE,
            duration=VIDEO_DURATION,
        )
    except Exception as e:
        logging.error(f"Error during video generation for prompt '{prompt}': {e}")
        print(f"Error during video generation for prompt '{prompt}': {e}")
        continue
    # except:
    #     logging.error(f"Failed to initiate video generation.")
    #     print(f"Failed to initiate video generation.")
    #     continue
    video_id = response.id
    if not video_id:
        logging.error(f"Failed to initiate video generation for: {prompt}")
        print(f"Failed to initiate video generation for: {prompt}")
        continue

    print(f"Video generation started. Video ID: {video_id}")

    # 查询视频生成结果
    video_result = None
    inquire_interval = 10  # 查询
    for try_times in range(60):  # 最多查询 60 次(10min)，间隔 10 秒
        response = client.videos.retrieve_videos_result(id=video_id)
        if response==None:
            logging.error(f"Error during video generation for idx '{idx}.")
            print(f"Error during video generation for idx '{idx}.")
            break
        if response.task_status == "SUCCESS":
            video_result = response.video_result
            url_list = [vr.url for vr in response.video_result]
            save_path = os.path.join(OUTPUT_DIR, f'{idx}.mp4')
            for url in url_list: download_video(url, save_path)
            print(response)
            print(url)
            break
        elif response.task_status == "FAIL":
            print(f"Video generation failed for: {video_id}")
            break
        elif response.task_status == "PROCESSING":
            print(f"Video generation in progress for: {video_id}. Used {try_times * inquire_interval} seconds...")
            time.sleep(inquire_interval)
        else:
            raise RuntimeError(f"Unknown task status: {response.task_status}")
