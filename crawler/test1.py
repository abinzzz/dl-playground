import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# 请求头
request_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.183",
    "Accept": "*/*"
}

def get_info(url_str, directory_path):
    with requests.Session() as session:
        session.headers.update(request_headers)
        response = session.get(url=url_str)
        print(f"访问 {url_str}，状态码：{response.status_code}")  # 调试打印
        if response.status_code == 200:
            soup_doc = BeautifulSoup(response.content, "lxml")
            img_elements = soup_doc.select("div.pic > img")
            print(f"找到 {len(img_elements)} 张图片")  # 调试打印

            for img in img_elements:
                option_value = img.get("data-option")
                img_url = "https://www.dpm.org.cn/" + img.get("src")
                print(f"图片完整 URL：{img_url}")  # 调试打印

                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                file_path = os.path.join(directory_path, option_value + ".png")
                response = session.get(img_url)
                print(f"下载 {img_url}，状态码：{response.status_code}")  # 调试打印

                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    print("图片下载失败:", img_url)
        else:
            print("请求失败，状态码：", response.status_code)

def download_images(urls, directory_path, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_url = {executor.submit(get_info, url, directory_path): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{url} 生成异常 {exc}")

if __name__ == "__main__":
    directory_path = "./gugong"
    #urls = ['https://www.dpm.org.cn/collection/paint/{}.shtml'.format(i) for i in range(230000, 240000)]#234597 #231643
    urls = ['https://www.dpm.org.cn/collection/paint/234597.shtml','https://www.dpm.org.cn/collection/paint/231643.shtml']
    num_threads = 2  # 可以根据实际情况调整线程数
    download_images(urls, directory_path, num_threads)
    print("下载完成...")
