import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time
from tqdm import tqdm
import sys
import traceback
import random

MODEL_PATH = os.getenv("TWPGEN_DICTIONARY_MODEL", "models/glm-4-9b-chat")
INPUT_DICT_PATH = os.getenv(
    "TWPGEN_VERB_DICT_INPUT", "experiments/dictionary/verb_sense_dict_zh.json"
)
OUTPUT_DICT_PATH = os.getenv(
    "TWPGEN_VERB_DICT_WITH_EXAMPLES",
    "experiments/dictionary/verb_sense_dict_with_examples.json",
)
######################################
#############使用环境NLP##############
######################################
# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 检查GPU是否可用
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    print(f"可用内存: {torch.cuda.mem_get_info()[0] / 1024 / 1024 / 1024:.2f} GB")

# 初始化 tokenizer 和模型
try:
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("分词器加载成功!")
    
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    print("模型加载成功!")
except Exception as e:
    print(f"加载模型出错: {e}")
    traceback.print_exc()
    sys.exit(1)

def generate_example(verb, definition, num_examples=1):
    """使用大模型生成多个例句"""
    try:
        # 根据要生成的例句数量调整提示
        if num_examples == 1:
            prompt_template = """请为以下中文动词及其定义生成一个简短的例句。
            
动词: {verb}
定义: {definition}

请直接给出一个自然、符合定义的例句，不要有任何解释或其他内容。例句应该简短、清晰地展示该动词在这个特定定义下的用法。"""
        else:
            prompt_template = """请为以下中文动词及其定义生成{num_examples}个不同的简短例句。

动词: {verb}
定义: {definition}

请直接给出{num_examples}个自然、符合定义的例句，每个例句一行，不要有任何解释或其他内容，也不要添加序号（如1.、2.等）。例句应该简短、清晰地展示该动词在这个特定定义下的用法。"""
        
        prompt = prompt_template.format(verb=verb, definition=definition, num_examples=num_examples)

        # 初始提示信息
        conversation_history = [{"role": "system", "content": "你是一个专业的中文例句生成助手，请根据用户提供的动词和定义生成简短、自然的例句。不要在例句前添加序号。"}]
        conversation_history.append({"role": "user", "content": prompt})
        
        print(f"准备模型输入...")
        # 应用聊天模板并转换为 PyTorch 张量
        inputs = tokenizer.apply_chat_template(conversation_history, 
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors="pt",
                                            return_dict=True)
        inputs = inputs.to(device)
        
        # 生成参数设置
        gen_kwargs = {
            "max_new_tokens": 300,
            "do_sample": True,
            "top_k": 1
        }
        
        print(f"生成例句中...")
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
        
        # 解码输出结果
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"例句生成成功")
        
        # 处理返回的例句
        examples = []
        if num_examples == 1:
            examples = [response.strip()]
        else:
            # 将响应按行分割，过滤掉空行
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            # 清理可能出现的序号
            cleaned_lines = []
            for line in lines:
                # 移除可能的序号格式 (1. 2. 等)
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    line = line[line.find('.')+1:].strip()
                cleaned_lines.append(line)
            
            # 最多取num_examples个例句
            examples = cleaned_lines[:num_examples]
            # 如果没有足够的例句，用第一个填充
            while len(examples) < num_examples and examples:
                examples.append(examples[0])
            
        return examples
    except Exception as e:
        print(f"生成例句时出错: {e}")
        traceback.print_exc()
        return [f"生成例句时出错: {str(e)}"]

def test_model():
    """测试模型是否正常工作"""
    print("使用示例动词测试模型...")
    # 测试生成1个例句
    example1 = generate_example("吃", "把食物等放到嘴里咽下去", 1)
    print(f"生成的1个例句: {example1}")
    
    # 测试生成3个例句
    example3 = generate_example("跑", "人或动物用脚或腿快速移动", 3)
    print(f"生成的3个例句: {example3}")
    
    print("测试完成.")

def save_dictionary(verb_list, dict_path, verb_idx, lemma, processed_count, total_count):
    """保存词典到文件"""
    try:
        with open(dict_path, 'w', encoding='utf-8') as f:
            json.dump(verb_list, f, ensure_ascii=False, indent=2)
        print(f"已保存词典到 {dict_path} - 处理了 {processed_count}/{total_count} 项，当前动词: '{lemma}'")
        return True
    except Exception as e:
        print(f"保存词典时出错: {e}")
        traceback.print_exc()
        return False

def process_dictionary():
    """处理词典文件，为所有词义添加例句，然后保留原有例句"""
    input_dict_path = INPUT_DICT_PATH
    output_dict_path = OUTPUT_DICT_PATH
    
    # 加载词典
    print(f"从 {input_dict_path} 加载词典")
    try:
        with open(input_dict_path, 'r', encoding='utf-8') as f:
            verb_list = json.load(f)
        print(f"词典加载成功，包含 {len(verb_list)} 个动词")
    except Exception as e:
        print(f"加载词典时出错: {e}")
        traceback.print_exc()
        return
    
    # 创建需要处理的项目列表 - 现在处理所有词条
    items_to_process = []
    for verb_idx, verb_entry in enumerate(verb_list):
        lemma = verb_entry.get("lemma", "").split('-')[0]  # 提取动词部分，去掉'-v'
        for sense_idx, sense in enumerate(verb_entry.get("senses", [])):
            items_to_process.append((verb_idx, lemma, sense_idx))
    
    total_items = len(items_to_process)
    print(f"共有 {total_items} 个词义需要处理")
    
    # 使用tqdm创建进度条
    pbar = tqdm(total=len(items_to_process), desc="处理进度")
    
    # 处理计数器
    processed_count = 0
    
    # 处理每个词义
    for verb_idx, lemma, sense_idx in items_to_process:
        try:
            sense = verb_list[verb_idx]["senses"][sense_idx]
            definition = sense.get("definition", "")
            # 清理定义文本，移除"Sense Number X: "前缀
            if "Sense Number" in definition:
                definition = definition.split(":", 1)[1].strip()
            
            # 保存原有的例句
            original_examples = sense.get("examples", [])
            
            # 随机决定生成1-3个例句
            num_examples = random.randint(1, 3)
            print(f"\n为动词 '{lemma}' 生成 {num_examples} 个例句，定义: {definition}")
            
            # 生成新例句
            new_examples = generate_example(lemma, definition, num_examples)
            print(f"生成的例句: {new_examples}")
            
            # 先添加新生成的例句，然后添加原有例句
            sense["examples"] = new_examples + original_examples
            
            processed_count += 1
            pbar.update(1)
            
            # 每处理10个词义就保存一次
            if processed_count % 10 == 0:
                save_dictionary(verb_list, output_dict_path, verb_idx, lemma, processed_count, total_items)
            
            # 添加短暂延迟，避免过于频繁的API调用
            time.sleep(1)
            
        except Exception as e:
            print(f"处理动词 '{lemma}' 时出错: {e}")
            traceback.print_exc()
    
    pbar.close()
    
    # 最终保存
    if save_dictionary(verb_list, output_dict_path, -1, "", processed_count, total_items):
        print(f"完成! 处理了 {processed_count} 个词义。")

if __name__ == "__main__":
    # 设置随机种子
    random.seed(time.time())
    
    # 直接处理词典
    print("\n开始处理词典...")
    process_dictionary() 
