#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import os
import json
import time
import sys
import urllib.parse
import re
import jieba
import jieba.posseg as pseg
import logging
from requests.exceptions import RequestException, ConnectionError, Timeout
import tqdm
import random
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('crawler.log'), logging.StreamHandler()]
)
logger = logging.getLogger('zidian_crawler')

class ZidianCrawler:
    """从字典网(zidian.com.cn)爬取词语解释"""
    
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://www.zidian.com.cn"
        self.all_words_json = os.getenv(
            "TWPGEN_VERB_DICT_OUTPUT",
            os.path.join(os.path.dirname(__file__), "verb_sense_dict_zh.json"),
        )
        self.all_words_list = []
        
        # 添加缓存，避免重复判断词性
        self.verb_cache = {}
        
        # 线程安全的队列和锁
        self.verb_queue = queue.Queue()
        self.result_lock = threading.Lock()
        
        # 设置请求头，模拟浏览器
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.zidian.com.cn/'
        })
        
        # 加载已有的词语解释
        if os.path.exists(self.all_words_json):
            try:
                with open(self.all_words_json, 'r', encoding='utf-8') as f:
                    self.all_words_list = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"JSON文件 {self.all_words_json} 解析失败，将创建新文件")
                self.all_words_list = []
            except Exception as e:
                logger.error(f"加载词典文件时出错: {e}")
                self.all_words_list = []
        
        # 常用动词列表
        self.common_verbs = [
            "吃饭", "睡觉", "学习", "工作", "跑步", "游泳", "阅读", "写作", "思考", "分析",
            "研究", "观察", "讨论", "交流", "沟通", "合作", "帮助", "支持", "鼓励", "批评",
            "表扬", "责备", "指导", "教育", "培训", "学会", "掌握", "理解", "记忆", "忘记",
            "购买", "销售", "交易", "投资", "储蓄", "消费", "赚钱", "花钱", "借钱", "还钱",
            "旅行", "出差", "散步", "锻炼", "休息", "放松", "娱乐", "玩耍", "唱歌", "跳舞",
            "演奏", "欣赏", "创作", "设计", "绘画", "雕刻", "摄影", "录像", "编辑", "修改",
            "计划", "安排", "组织", "管理", "领导", "执行", "完成", "实现", "达成", "成功",
            "失败", "尝试", "努力", "坚持", "放弃", "开始", "继续", "停止", "结束", "暂停",
            "思念", "想念", "怀念", "回忆", "期待", "盼望", "希望", "渴望", "向往", "憧憬",
            "爱护", "关心", "照顾", "保护", "维护", "珍惜", "珍爱", "珍重", "爱惜", "喜爱"
        ]
        
        # 常用名词列表
        self.common_nouns = [
            "时间", "地点", "人物", "事件", "物品", "概念", "思想", "理论", "方法", "技术",
            "工具", "设备", "机器", "材料", "资源", "能源", "环境", "气候", "天气", "季节",
            "国家", "城市", "地区", "社区", "家庭", "学校", "公司", "组织", "政府", "机构",
            "文化", "艺术", "音乐", "文学", "电影", "戏剧", "舞蹈", "绘画", "雕塑", "建筑",
            "科学", "数学", "物理", "化学", "生物", "医学", "工程", "计算机", "网络", "软件",
            "经济", "政治", "法律", "教育", "健康", "医疗", "保险", "金融", "银行", "投资",
            "食物", "饮料", "服装", "鞋子", "家具", "电器", "交通", "通信", "信息", "数据",
            "问题", "困难", "挑战", "机会", "风险", "危机", "成功", "失败", "进步", "发展",
            "历史", "传统", "习俗", "礼仪", "道德", "伦理", "价值", "信仰", "宗教", "哲学",
            "感情", "情绪", "态度", "性格", "品质", "能力", "技能", "知识", "经验", "智慧"
        ]
    
    def save_all_words(self):
        """保存所有词语到统一的JSON文件"""
        try:
            # 筛选多义动词和所有名词
            filtered_words = []
            for entry in self.all_words_list:
                lemma = entry.get('lemma', '')
                # 如果是动词（以-v结尾），则只保留多义词
                if lemma.endswith('-v'):
                    if len(entry.get('senses', [])) > 1:
                        filtered_words.append(entry)
                        logger.debug(f"保留多义动词: {lemma} ({len(entry.get('senses', []))}个词义)")
                    else:
                        logger.debug(f"过滤单义动词: {lemma}")
                # 如果是名词，则全部保留
                else:
                    filtered_words.append(entry)
                    logger.debug(f"保留名词: {lemma}")
            
            # 更新词语列表
            self.all_words_list = filtered_words
            
            # 保存到文件
            with open(self.all_words_json, 'w', encoding='utf-8') as f:
                json.dump(self.all_words_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存词典文件，共 {len(self.all_words_list)} 个词条")
            return True
        except Exception as e:
            logger.error(f"保存词典文件时出错: {e}")
            return False
    
    def is_verb(self, word):
        """
        判断词语是否为动词
        :param word: 要判断的词语
        :return: 是否为动词
        """
        # 先检查缓存
        if word in self.verb_cache:
            return self.verb_cache[word]
        
        try:
            # 方法1：使用预定义的动词列表（最快）
            if word in self.common_verbs:
                self.verb_cache[word] = True
                return True
            
            # 基于词语特征的快速判断（避免使用jieba）
            # 方法2：检查词语是否包含常见的动词后缀
            verb_suffixes = ["化", "动", "行", "做", "想", "看", "听", "说", "读", "写", "用", "吃", "喝", "走", "跑", "跳", "学", "教", "买", "卖"]
            for suffix in verb_suffixes:
                if word.endswith(suffix) and len(word) > 1:
                    self.verb_cache[word] = True
                    return True
            
            # 方法3：检查词语是否以常见动词开头
            verb_prefixes = ["做", "去", "来", "想", "要", "能", "会", "可", "应", "该"]
            for prefix in verb_prefixes:
                if word.startswith(prefix) and len(word) > 1:
                    self.verb_cache[word] = True
                    return True
            
            # 方法4：使用jieba进行词性标注（最慢，但最准确）
            # 只有前面的快速方法都无法判断时才使用
            words = pseg.cut(word)
            for w, flag in words:
                if w == word and flag.startswith('v'):  # v开头的标记表示动词
                    self.verb_cache[word] = True
                    return True
            
            # 都不是，缓存结果
            self.verb_cache[word] = False
            return False
        except Exception as e:
            logger.error(f"判断词语 '{word}' 是否为动词时出错: {e}")
            # 出错时保守返回False
            self.verb_cache[word] = False
            return False
    
    def request_with_retry(self, url, method="get", params=None, max_retries=3, timeout=10):
        """
        发送请求并自动重试
        :param url: 请求URL
        :param method: 请求方法，默认为get
        :param params: 请求参数
        :param max_retries: 最大重试次数
        :param timeout: 超时时间（秒）
        :return: 响应对象或None
        """
        retries = 0
        while retries < max_retries:
            try:
                if method.lower() == "get":
                    response = self.session.get(url, params=params, timeout=timeout)
                else:
                    response = self.session.post(url, data=params, timeout=timeout)
                
                response.raise_for_status()
                return response
            except ConnectionError:
                logger.warning(f"连接错误，正在重试 ({retries+1}/{max_retries})...")
            except Timeout:
                logger.warning(f"请求超时，正在重试 ({retries+1}/{max_retries})...")
            except RequestException as e:
                logger.warning(f"请求异常: {e}，正在重试 ({retries+1}/{max_retries})...")
            
            retries += 1
            # 指数退避策略
            time.sleep(2 ** retries)
        
        logger.error(f"请求失败，已达到最大重试次数: {url}")
        return None
    
    def get_explanation(self, word, idx=0, total=0):
        """
        获取词语解释
        :param word: 要查询的词语
        :param idx: 当前词语索引
        :param total: 总词语数
        :return: 词语条目
        """
        # 检查词语是否已经查询过
        for entry in self.all_words_list:
            if entry.get('lemma') == word or entry.get('lemma') == f"{word}-v":
                return entry
        
        try:
            # 访问字典网主页
            main_url = "https://www.zidian.com.cn/"
            response = self.request_with_retry(main_url)
            if not response:
                return None
            
            # 搜索词语
            search_url = f"{self.base_url}/search"
            search_params = {'keyword': word}
            
            search_response = self.request_with_retry(search_url, params=search_params)
            if not search_response:
                return None
            
            # 解析搜索结果页面
            search_soup = BeautifulSoup(search_response.text, 'html.parser')
            
            # 查找词语链接
            word_link = None
            
            # 查找所有可能的链接，优先选择与搜索词完全匹配的结果
            all_links = []
            exact_match_link = None
            
            # 首先查找词语链接
            for a_tag in search_soup.find_all('a'):
                href = a_tag.get('href', '')
                # 检查链接是否包含词语相关路径
                if '/ci/' in href:
                    img = a_tag.find('img')
                    if img:
                        title = img.get('title', '')
                        all_links.append((href, title, 'ci'))
                        # 检查是否与搜索词完全匹配
                        if title == word:
                            exact_match_link = (href, title, 'ci')
                            break
            
            # 如果找到了精确匹配，使用它
            if exact_match_link:
                word_link = exact_match_link[0]
            # 否则，如果找到了词语链接，使用第一个
            elif all_links:
                word_link = all_links[0][0]
                # 更新词条名称为实际搜索到的词语
                word = all_links[0][1]
            # 如果没找到词语链接，查找汉字链接
            else:
                for a_tag in search_soup.find_all('a'):
                    href = a_tag.get('href', '')
                    if '/zi/' in href:
                        img = a_tag.find('img')
                        if img:
                            title = img.get('title', '')
                            all_links.append((href, title, 'zi'))
                            # 检查是否与搜索词完全匹配
                            if title == word:
                                exact_match_link = (href, title, 'zi')
                                break
                
                # 如果找到了精确匹配，使用它
                if exact_match_link:
                    word_link = exact_match_link[0]
                # 否则，如果找到了汉字链接，使用第一个
                elif all_links:
                    word_link = all_links[0][0]
                    # 更新词条名称为实际搜索到的词语
                    word = all_links[0][1]
                # 如果还没找到，查找成语链接
                else:
                    for a_tag in search_soup.find_all('a'):
                        href = a_tag.get('href', '')
                        if '/cy/' in href:
                            img = a_tag.find('img')
                            if img:
                                title = img.get('title', '')
                                all_links.append((href, title, 'cy'))
                                # 检查是否与搜索词完全匹配
                                if title == word:
                                    exact_match_link = (href, title, 'cy')
                                    break
                    
                    # 如果找到了精确匹配，使用它
                    if exact_match_link:
                        word_link = exact_match_link[0]
                    # 否则，如果找到了成语链接，使用第一个
                    elif all_links:
                        word_link = all_links[0][0]
                        # 更新词条名称为实际搜索到的词语
                        word = all_links[0][1]
            
            # 如果找到了链接，访问详情页
            if word_link:
                # 确保链接是完整的URL
                if not word_link.startswith('http'):
                    if word_link.startswith('//'):
                        word_link = 'https:' + word_link
                    else:
                        word_link = self.base_url + word_link
                
                detail_response = self.request_with_retry(word_link)
                if not detail_response:
                    return None
                
                # 解析详情页
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                
                # 判断是否为动词，并在lemma后添加"-v"
                is_verb_word = self.is_verb(word)
                lemma = f"{word}-v" if is_verb_word else word
                
                # 创建词语条目
                word_entry = {
                    "lemma": lemma,
                    "senses": []
                }
                
                # 收集所有解释和例句
                senses = []
                
                # 提取词语解释部分
                explanation_sections = detail_soup.select('.zidian-zi')
                
                for section in explanation_sections:
                    # 查找标题，确定是哪种解释
                    title_div = section.select_one('.fs-4')
                    if not title_div:
                        continue
                    
                    title = title_div.get_text(strip=True)
                    
                    # 只处理词语解释部分
                    if "词语解释" in title:
                        # 查找所有定义和例句
                        p_tags = section.select('p.indent')
                        
                        current_definition = None
                        current_examples = []
                        
                        for p in p_tags:
                            text = p.get_text(strip=True)
                            
                            # 跳过空行
                            if not text:
                                continue
                            
                            # 检查是否是新的释义
                            if text.startswith('⒈') or text.startswith('⒉') or re.match(r'^\d+\.', text):
                                # 如果已有当前释义，保存之前的释义和例句
                                if current_definition:
                                    senses.append({
                                        "definition": current_definition,
                                        "examples": current_examples
                                    })
                                    current_examples = []
                                
                                # 提取新的释义
                                definition = re.sub(r'^[⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑]|^\d+\.', '', text).strip()
                                current_definition = definition
                            
                            # 检查是否是例句
                            elif p.select('.attr_tag') and '例' in p.select_one('.attr_tag').text:
                                example_text = p.select_one('.attr_ext').text if p.select_one('.attr_ext') else ""
                                if example_text:
                                    # 处理例句
                                    example_text = example_text.strip()
                                    if example_text:
                                        current_examples.append(example_text)
                        
                        # 保存最后一个释义
                        if current_definition:
                            senses.append({
                                "definition": current_definition,
                                "examples": current_examples
                            })
                
                if not senses:
                    logger.warning(f"未找到词语 '{word}' 的解释")
                    return None
                
                # 检查是否为多义词（有多个词义的词）
                if is_verb_word and len(senses) <= 1:
                    logger.info(f"词语 '{word}' 只有一个词义，不保存")
                    return None
                
                # 创建最终的sense列表
                for i, sense in enumerate(senses, 1):
                    word_entry["senses"].append({
                        "definition": f"Sense Number {i}: {sense['definition']}",
                        "examples": sense['examples'],
                        "mappings": {}
                    })
                
                # 添加到所有词语列表（使用线程安全的方式）
                with self.result_lock:
                    self.all_words_list.append(word_entry)
                    
                    # 打印进度信息
                    if idx > 0 and total > 0:
                        if len(word_entry["senses"]) > 1:
                            print(f"多义动词 '{word}' 处理完毕，共 {len(word_entry['senses'])} 个词义 ({idx}/{total})")
                        else:
                            print(f"单义动词 '{word}' 已跳过 ({idx}/{total})")
                
                return word_entry
            else:
                logger.warning(f"未找到词语 '{word}' 的链接")
        
        except Exception as e:
            logger.error(f"处理词语 '{word}' 时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return None

    def worker_thread(self, total_lines):
        """
        工作线程，从队列中获取动词并处理
        """
        success_count = 0
        multi_sense_count = 0
        
        while True:
            try:
                # 从队列中获取词语，如果队列为空，退出线程
                item = self.verb_queue.get(block=False)
                if item is None:
                    break
                
                idx, word = item
                
                # 获取解释
                result = self.get_explanation(word, idx, total_lines)
                if result:
                    with self.result_lock:
                        success_count += 1
                        # 检查是否为多义词
                        if len(result["senses"]) > 1:
                            multi_sense_count += 1
                
                # 每处理10个词语，保存一次词典
                if success_count % 10 == 0:
                    with self.result_lock:
                        self.save_all_words()
                
                # 避免请求过于频繁
                time.sleep(random.uniform(0.5, 1))
                
                # 标记任务完成
                self.verb_queue.task_done()
            
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"工作线程处理出错: {e}")
                self.verb_queue.task_done()
        
        return success_count, multi_sense_count

    def process_words_from_file(self, file_path):
        """
        从文件读取词语并处理
        :param file_path: 词语文件路径
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件 {file_path} 不存在")
            return False
        
        # 统计文件总行数
        total_lines = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
        
        logger.info(f"文件 {file_path} 共 {total_lines} 行，开始处理")
        
        # 读取并处理词语
        processed_count = 0
        verb_count = 0
        
        # 读取所有词语并判断词性
        all_words = []
        verb_words = []
        
        logger.info("第一阶段：读取文件并判断词性")
        with open(file_path, 'r', encoding='utf-8') as f:
            # 使用tqdm创建进度条
            with tqdm.tqdm(total=total_lines, desc="判断词性") as pbar:
                for i, line in enumerate(f, start=1):
                    # 跳过注释行或空行
                    line = line.strip()
                    if not line or line.startswith('#'):
                        pbar.update(1)
                        continue
                    
                    # 处理词语
                    word = line.strip()
                    all_words.append((i, word))
                    processed_count += 1
                    
                    # 判断词性
                    if self.is_verb(word):
                        verb_count += 1
                        verb_words.append((i, word))
                        print(f"发现动词: '{word}' ({i}/{total_lines})")
                    else:
                        print(f"非动词 '{word}' 已跳过 ({i}/{total_lines})")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "已处理": processed_count,
                        "动词": verb_count
                    })
        
        logger.info(f"词性判断完成，共 {processed_count} 个词语，其中 {verb_count} 个动词")
        
        # 第二阶段：多线程爬取动词解释
        logger.info("第二阶段：多线程爬取动词解释")
        
        # 将所有动词放入队列
        for item in verb_words:
            self.verb_queue.put(item)
        
        # 创建工作线程
        num_threads = min(10, verb_count)  # 最多10个线程
        threads = []
        
        with tqdm.tqdm(total=verb_count, desc="爬取动词") as pbar:
            # 启动工作线程
            for _ in range(num_threads):
                thread = threading.Thread(target=self._crawler_worker, args=(total_lines, pbar))
                thread.daemon = True
                thread.start()
                threads.append(thread)
            
            # 等待所有任务完成
            self.verb_queue.join()
            
            # 向队列中添加结束信号
            for _ in range(num_threads):
                self.verb_queue.put(None)
            
            # 等待所有线程结束
            for thread in threads:
                thread.join()
        
        # 最后再次保存词典（并筛选）
        self.save_all_words()
        
        # 统计多义动词数量
        multi_sense_count = 0
        for entry in self.all_words_list:
            if entry.get('lemma', '').endswith('-v') and len(entry.get('senses', [])) > 1:
                multi_sense_count += 1
        
        logger.info(f"处理完成，共处理 {processed_count} 个词语，其中 {verb_count} 个动词，成功获取 {len(self.all_words_list)} 个词语的解释，{multi_sense_count} 个多义动词")
        return True
    
    def _crawler_worker(self, total_lines, pbar):
        """
        爬虫工作线程
        """
        while True:
            try:
                # 从队列中获取词语，如果队列为空，退出线程
                item = self.verb_queue.get(block=False)
                if item is None:
                    break
                
                idx, word = item
                
                # 获取解释
                result = self.get_explanation(word, idx, total_lines)
                
                # 更新进度条
                pbar.update(1)
                
                # 标记任务完成
                self.verb_queue.task_done()
            
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"工作线程处理出错: {e}")
                pbar.update(1)
                self.verb_queue.task_done()

def main():
    """主函数"""
    # 创建爬虫实例
    crawler = ZidianCrawler()
    
    file_path = os.getenv("TWPGEN_WORD_LIST", "data/words.txt")
    print(f"开始处理文件: {file_path}")
    crawler.process_words_from_file(file_path)

if __name__ == "__main__":
    main()
