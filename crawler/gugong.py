import os
import time
import tqdm
import requests
from bs4 import BeautifulSoup

# 请求头
request_headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.183",
    "Accept":"*/*"
}

def get_info(url_str,directory_path):
    """
    获取图片信息
    """
    response = requests.get(url=url_str,headers=request_headers)
    response.encoding = 'gbk'
    statue_code = response.status_code
    if statue_code == 200:
        """
        判断响应状态码是否为200(成功),如果是200则继续执行,否则直接返回。
        """

        #将响应内容解析为HTML文档对象。
        html_text = response.content
        soup_doc =BeautifulSoup(html_text,"lxml",from_encoding="utf-8")

        #从HTML文档中选择包含图片URL和类型信息的元素
        img_elements = soup_doc.select("div.pic > img")
        type_elements = soup_doc.select("#bread-paints")

        #如果类型信息元素为空,则直接返回。
        if len(type_elements) == 0:
            return
        for img in img_elements:
            """遍历图片元素,获取图片URL和选项值,并打印出来"""
            option_value = img.get("data-option")
            img_url = "https://www.dpm.org.cn/" + img.get("src")
            #print(url_str,option_value)
            directory_path = directory_path
            # 确保目录存在，如果不存在则创建
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            # 完整的文件路径
            file_path = os.path.join(directory_path, option_value+".png")
            # 发送HTTP请求获取图片内容
            response = requests.get(img_url)

            # 确保请求成功
            if response.status_code == 200:
                # 将图片内容写入文件
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                print("失败")
    else:
        pass
if __name__ == "__main__":
    # 生成urllist
    directory_path="./gugong"
    urls = ['https://www.dpm.org.cn/collection/paint/{}.shtml'.format(str(i)) for i in range(220000,230000)]#22~23万
    #urls = ['https://www.dpm.org.cn/collection/paint/221032.shtml']
    #print(urls)
    for url_str in tqdm.tqdm(urls, desc="下载进度"):
        get_info(url_str,directory_path)
        time.sleep(1)
    print("End...")


