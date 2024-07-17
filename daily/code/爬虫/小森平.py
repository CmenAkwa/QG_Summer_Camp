import requests
from bs4 import BeautifulSoup
import os

headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}
response = requests.get("https://taira-komori.jpn.org/nature01cn.html", headers=headers)
if response.ok:
    print("获取成功", response.status_code)

else:
    print("获取失败")
    exit()

content = response.text
soup = BeautifulSoup(content, "html.parser")
a_tags = soup.find_all('a')
url = 'https://taira-komori.jpn.org'
sounds_dir = "natural_sound"

# 遍历所有的<a>标签
for a_tag in a_tags:
    # 检查<a>标签是否包含download属性
    if 'download' in a_tag.attrs:
        # 获取href属性的值，即音频文件的下载链接
        audio_url = a_tag['href']
        # 获取download属性的值，即音频文件的名称
        audio_filename = a_tag['download']
        download_url = url + '/' + audio_url
        download_response = requests.get(download_url, stream=True)
        save_path = os.path.join(sounds_dir, audio_filename)
        if download_response.ok:
            with open(save_path, 'wb') as file:
                # 写入内容到文件
                for chunk in download_response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"文件 '{audio_filename}' 下载成功并保存在 '{save_path}'")
        else:
            print(f"文件 '{audio_filename}' 下载失败，状态码: {download_response.status_code}")
