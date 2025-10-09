"""
Q:我如何用python 将网络地址上的视频保存到本地，网络地址如：https://aigc-files.bigmodel.cn/api/cogvideo/7894729c-b6eb-11ef-8a7f-1a96d44b8a2b_0.mp4
"""
import requests
from tqdm import tqdm

def download_video(url, save_path):
    """
    下载视频并保存到本地文件
    :param url: 视频的网络地址
    :param save_path: 本地保存路径（包括文件名和扩展名）
    """
    try:
        # 发送 HTTP GET 请求
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 保存视频到本地
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"视频已保存到 {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"下载视频时出错: {e}")

def download_video_with_progress(url, save_path):
    """
    下载视频并显示下载进度
    :param url: 视频的网络地址
    :param save_path: 本地保存路径（包括文件名和扩展名）
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))
        print(f"视频已保存到 {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"下载视频时出错: {e}")

if __name__ == '__main__':
    video_url = "https://aigc-files.bigmodel.cn/api/cogvideo/7894729c-b6eb-11ef-8a7f-1a96d44b8a2b_0.mp4"
    save_path = "downloaded_video.mp4"
    download_video(video_url, save_path)
    download_video_with_progress(video_url, save_path)
