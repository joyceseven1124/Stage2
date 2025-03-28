import requests
from bs4 import BeautifulSoup
import time
import random
import logging
import pandas as pd
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import os
from datetime import datetime

# 2025-03-11 14:30:01,123 - INFO - 爬蟲開始執行...
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_progress(board, current_url):
    progress = {}
    if os.path.exists('progress.json'):
        with open('progress.json', 'r') as f:
            progress = json.load(f)
    progress[board] = current_url
    with open('progress.json', 'w') as f:
        json.dump(progress, f)

def load_progress(board):
    if os.path.exists('progress.json'):
        with open('progress.json', 'r') as f:
            progress = json.load(f)
            return progress.get(board)
    return None

class PTTScraper:
    def __init__(self, board, sleep_time=(5, 10), batch_size=1000, start_url=None):
        self.domain = "https://www.ptt.cc"
        self.board = board
        # self.board_url = f"{self.domain}/bbs/{board}/index.html"
        self.board_url = start_url or f"{self.domain}/bbs/{board}/index.html"
        self.headers = {
            # 模擬瀏覽器
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Cookie': 'over18=1' # 處理年齡限制
        }
        self.sleep_time = sleep_time
        self.results = []
        self.batch_size = batch_size
        self.post_count = 0
        self.batch_number = 1
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")


    def _random_sleep(self):
        sleep_duration = random.uniform(*self.sleep_time)
        # logging.info(f"等待 {sleep_duration:.2f} 秒")
        time.sleep(sleep_duration)

    def _get_page(self, url, max_retries=3):
        for attempt in range(max_retries):
            try:
                # logging.info(f"請求頁面: {url} (嘗試 {attempt + 1}/{max_retries})")
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')
            except requests.exceptions.RequestException as e:
                logging.error(f"請求失敗: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 遞增等待時間
                    # logging.info(f"等待 {wait_time} 秒後重試")
                    time.sleep(wait_time)
                else:
                    logging.error(f"已達到最大重試次數，跳過此頁面")
        return None

    def scrape_posts(self, max_pages=1):
        # logging.info(f"開始爬取 {self.board} 版, 最多 {max_pages} 頁")
        page_url = self.board_url
        # 最後一筆資料
        if(page_url == -1):
            return
        
        page_count = 0

        with tqdm(total=max_pages, desc=f"爬取 {self.board} 版") as pbar:
            while page_count < max_pages:
                page_count += 1
                current_page_content = self._get_page(page_url)
                if not current_page_content:
                    break

                # class_=
                posts = current_page_content.find_all('div', class_="r-ent")
                for post in posts:
                    title_div = post.find('div', class_='title')
                    if '(本文已被刪除)' in title_div.text or not title_div.a:
                        continue

                    title = title_div.text.strip().replace('\n', ' ').replace('\r', ' ')
                    link = urljoin(self.domain, title_div.a['href'])
                    nrec = post.find('div', class_='nrec').text.strip()
                    author = post.find('div', class_='author').text.strip()
                    date = post.find('div', class_='date').text.strip()

                    self.results.append({
                        'title': title,
                        'nrec': nrec,
                        'author': author,
                        'date': date,
                        'link': link
                    })
                    # 分批儲存
                    self.post_count += 1
                    if self.post_count % self.batch_size == 0:
                        self._save_batch()

                # logging.info(f"爬取文章: {title}")

                prev_link = current_page_content.find('a', string='‹ 上頁')

                if not prev_link or 'href' not in prev_link.attrs:
                    save_progress(self.board, -1)
                    break

                page_url = urljoin(self.domain, prev_link['href'])
                self._random_sleep()
                save_progress(self.board, page_url)
                pbar.update(1)

        if self.results:  # 保存最後一批
            self._save_batch()

        # logging.info(f"{self.board} 版爬取完成，共爬取 {len(self.results)} 篇標題")
        return self.results

    def _save_batch(self):
        # filename = f"{self.board}_posts_batch_{self.batch_number}.csv"
        folder = './raw'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = f"{self.board}_posts_{self.start_time}_batch{self.batch_number:04d}.csv"
        file_path = os.path.join(folder, filename)
        df = pd.DataFrame(self.results)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        # logging.info(f"批次 {self.batch_number} 已保存至 {filename}，共 {len(self.results)} 篇文章")
        self.results = []  # 清空結果列表
        self.batch_number += 1

def scrape_board(board, max_pages=10001, batch_size=1000):
    start_url = load_progress(board)
    scraper = PTTScraper(board, sleep_time=(8, 15), batch_size=batch_size, start_url=start_url)
    total_posts = scraper.scrape_posts(max_pages=max_pages)
    if(total_posts):
        return board, len(total_posts)
    else:
        return board, 0

if __name__ == "__main__":
    board_names = ['baseball', 'Boy-Girl', 'c_chat', 'hatepolitics', 'Lifeismoney', 'Military', 'pc_shopping', 'stock', 'Tech_Job']
    max_threads = 5  # 限制最多 5 個執行緒
    max_pages = 1
    batch_size = 1000

    with ThreadPoolExecutor(max_threads) as executor:
        future_to_board = {executor.submit(scrape_board, board, max_pages, batch_size): board for board in board_names}
        for future in as_completed(future_to_board):
            board, num_posts = future.result()
            # logging.info(f"{board} 版爬取完成，總共 {num_posts} 篇文章")
