import os
from dotenv import load_dotenv
import concurrent.futures
from zhipuai import ZhipuAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()
zhipu_api = os.getenv("ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=zhipu_api)

with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_text(text)


def process_chunk(chunk, index):
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "assistant", "content": "你是一个文档整理助手"},
            {"role": "user",
             "content": f"你是会议记录整理人员，以下是一段录音的逐字稿，请逐字将其整理成前后连贯的文字，需要注意："
                        f"1.保留完整保留原始录音的所有细节，不要进行概括和总结。"
                        f"2.尽量保留原文语义、语感。"
                        f"3.请修改错别字，符合中文语法规范。"
                        f"4.去掉说话人和时间戳。"
                        f"5.采用第一人称：我。"
                        f"6.请足够详细，字数越多越好。"
                        f"7.保持原始录音逐字稿的语言风格。"
                        f"8.直接输出处理后的文字"
                        f"以下是逐字稿：```{chunk}```"
             }
        ]
    )

    summary = response.choices[0].message.content
    with open(f'summary_chunk_{index + 1}.txt', 'w', encoding='utf-8') as file:
        file.write(summary)
    return summary


def summarize_text_concurrently(chunks):
    summaries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 使用tqdm创建进度条
        futures = [executor.submit(process_chunk, chunk, i) for i, chunk in enumerate(chunks)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            summaries.append(future.result())
    return summaries


# 执行并发摘要
summaries = summarize_text_concurrently(chunks)


# 将所有摘要合并到一个文件中
def merge_summaries():
    # 合并文件的名称
    merged_file_name = 'summary_output.txt'
    # 打开合并后的文件，准备写入
    with open(merged_file_name, 'w', encoding='utf-8') as merged_file:
        # 遍历所有可能的摘要片段文件
        for i in range(1, len(chunks) + 1):
            # 构建每个摘要片段的文件名
            chunk_file_name = f'summary_chunk_{i}.txt'
            # 打开每个摘要片段文件，读取其内容
            with open(chunk_file_name, 'r', encoding='utf-8') as chunk_file:
                # 将每个摘要片段的内容写入合并后的文件
                merged_file.write(chunk_file.read())
                # 如果不是最后一个文件，添加换行符以便区分不同的摘要片段
                if i != len(chunks):
                    merged_file.write('\n\n')


# 调用函数执行合并操作
merge_summaries()
print("Summarization complete. Check the summary_output.txt file.")
