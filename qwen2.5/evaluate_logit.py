# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import json
import csv
import random
from typing import List, Dict, Any
import torch
from tqdm import tqdm
import logging # 导入标准 logging 模块以设置

# 导入swift库中的相关模块
try:
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig
    from swift.utils import get_logger
    # 获取 swift 的 logger
    swift_logger = get_logger()
    # 设置 swift logger 的级别为 WARNING 或更高，以减少其自身的日志输出
    swift_logger.setLevel(logging.WARNING)

    # 我们也可以为我们自己的脚本创建一个独立的 logger，并独立控制其级别
    logger = logging.getLogger(__name__) # 创建一个当前模块的 logger
    logger.setLevel(logging.INFO) # 默认让我们自己的关键信息打印出来
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers(): # 防止重复添加 handler
        logger.addHandler(handler)

except ImportError as e:
    print(f"无法导入 swift.llm 或 swift.utils 的核心模块: {e}")
    print("请确保 swift 库已正确安装，特别是 PtEngine 相关的部分。")
    exit(1)
except Exception as e_general: # 捕获可能的其他 get_logger 相关错误
    print(f"初始化 logger 时发生错误: {e_general}")
    # 如果 logger 初始化失败，创建一个备用的简单打印 logger
    class SimpleLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    logger = SimpleLogger()


# ==============================================================================
# Функции, которые мы разработали для получения ответа и извлечения logprobs
# ==============================================================================

def get_model_response(engine: InferEngine,
                       prompt_messages: List[Dict[str, str]],
                       image_path_list: List[str],
                       model_max_tokens: int = 10,
                       temp: float = 0.0,
                       request_config_kwargs: Dict[str, Any] = None) -> (str | None, Any | None):
    """
    使用 swift.llm Engine 获取模型的文本响应及完整的响应对象。
    """
    infer_request = InferRequest(messages=prompt_messages, images=image_path_list)

    config_params = {
        "max_tokens": model_max_tokens,
        "temperature": temp,
        "logprobs": True,
        "top_logprobs": 5
    }
    if request_config_kwargs:
        config_params.update(request_config_kwargs)
    
    # logger.info(f"使用 RequestConfig 参数: {config_params}") # 可以注释掉以减少输出
    request_config = RequestConfig(**config_params)

    try:
        responses = engine.infer([infer_request], request_config=request_config)
    except Exception as e:
        logger.error(f"engine.infer 调用失败: {e}")
        return None, None

    if not responses:
        logger.error("推理引擎没有返回任何响应。")
        return None, None

    first_response = responses[0]

    if not first_response.choices:
        logger.error("响应中没有 choices。")
        # logger.info(f"原始响应对象内容: {vars(first_response)}") # 调试时开启
        return None, first_response

    text_content = first_response.choices[0].message.content.strip()
    return text_content, first_response

def extract_token_logprobs_from_swift(response_choice: Any, target_token_strings: List[str]) -> Dict[str, float]:
    """
    从 swift.llm 的响应中提取目标 token 字符串的 log probabilities。
    """
    token_logprobs_map = {token.lower(): -float('inf') for token in target_token_strings}

    if not hasattr(response_choice, 'logprobs') or response_choice.logprobs is None:
        # logger.warning("response_choice 对象中没有 'logprobs' 属性或其值为 None。") # 可以注释掉
        return token_logprobs_map

    logprobs_data = response_choice.logprobs
    # try: # 调试时开启下面的日志
    #     logger.info(f"收到的 response_choice.logprobs 原始结构: {json.dumps(logprobs_data, indent=2, ensure_ascii=False)}")
    # except TypeError:
    #     logger.info(f"收到的 response_choice.logprobs 原始结构 (直接打印): {logprobs_data}")


    try:
        if isinstance(logprobs_data, dict) and \
           'content' in logprobs_data and \
           isinstance(logprobs_data['content'], list) and \
           len(logprobs_data['content']) > 0 and \
           isinstance(logprobs_data['content'][0], dict) and \
           'top_logprobs' in logprobs_data['content'][0] and \
           isinstance(logprobs_data['content'][0]['top_logprobs'], list):
            
            first_step_top_logprobs_list = logprobs_data['content'][0]['top_logprobs']
            # logger.info(f"成功定位到第一个生成步骤的 top_logprobs 列表: {first_step_top_logprobs_list}") # 调试时可开启

            for item_dict in first_step_top_logprobs_list:
                if isinstance(item_dict, dict) and 'token' in item_dict and 'logprob' in item_dict:
                    api_token_str = str(item_dict['token']).strip() 
                    logprob_value = item_dict['logprob']
                    api_token_str_lower = api_token_str.lower()
                    
                    if api_token_str_lower in token_logprobs_map: 
                        if isinstance(logprob_value, (float, int)):
                            current_logprob_value = float(logprob_value)
                            if current_logprob_value > token_logprobs_map[api_token_str_lower]:
                                token_logprobs_map[api_token_str_lower] = current_logprob_value
                                # logger.info(f"更新/找到目标 token '{api_token_str_lower}' (来自API的'{api_token_str}') 的 logprob: {current_logprob_value}") # 调试时可开启
                        # else: # 可以注释掉不必要的警告
                            # logger.warning(f"Token '{api_token_str}' 的 logprob值 '{logprob_value}' 不是有效的数字。")
                # else: # 可以注释掉不必要的警告
                    # logger.warning(f"top_logprobs 列表中的项目格式不符合预期: {item_dict}")
        # else: # 可以注释掉不必要的警告
            # logger.warning("未能从 logprobs_data 中找到预期的 'content[0].top_logprobs' 结构。请再次核对原始结构。")

    except Exception as e:
        logger.error(f"解析 logprobs_data 时发生错误: {e}") # 保留错误信息

    return token_logprobs_map

# ==============================================================================
# 主评估逻辑
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="图片真伪性评估脚本 (基于Swift LLM)")
    parser.add_argument("--model_path", type=str,  default="./output_ckpt/v4-20250603-074103/checkpoint-merged", help="SWIFT格式的模型checkpoint路径。")
    parser.add_argument("--dataset_path", type=str, default="./datasets/img_test.json", help="包含图片路径和标注的JSON数据集文件路径。")
    parser.add_argument("--output_csv", type=str, default="deepfake_evaluation_resultsqwen25.csv", help="输出结果的CSV文件名。")
    # parser.add_argument("--output_json", type=str, default="deepfake_evaluation_summary.json", help="输出摘要信息的JSON文件名。") # 注释掉或移除JSON输出参数
    parser.add_argument("--infer_backend", type=str, default='pt', choices=['pt'], help="推理后端类型 (当前脚本优化为仅支持'pt')。")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="模型为A或B这样的回答生成的最大token数。")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度。")
    parser.add_argument("--request_config_extra", type=str, help="JSON字符串格式的额外RequestConfig参数, e.g., '{\"custom_param\": true}'")
    parser.add_argument("--quiet", action="store_true", help="启用安静模式，大幅减少控制台输出。")


    args = parser.parse_args()
    
    # 如果用户指定了 --quiet，则将我们自定义的 logger 级别也调高
    if args.quiet:
        logger.setLevel(logging.WARNING) # 或 logging.ERROR
        # 对于 swift_logger，我们已经在前面设置了 WARNING，如果想更安静可以设 ERROR
        # swift_logger.setLevel(logging.ERROR)

    if args.infer_backend != 'pt':
        logger.error(f"此脚本当前已优化为仅支持 'pt' 后端。你选择了 '{args.infer_backend}'。请修改脚本以支持其他后端或使用 'pt'。")
        exit(1)

    # logger.info(f"使用的推理后端: {args.infer_backend}") # 注释掉一些常规info
    # logger.info(f"从路径 '{args.model_path}' 初始化引擎...")
    try:
        engine = PtEngine(args.model_path)
    except Exception as e:
        logger.error(f"初始化 PtEngine 失败: {e}")
        exit(1)

    # logger.info(f"从 '{args.dataset_path}' 加载数据集...")
    try:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        exit(1)
    
    if not datas:
        logger.warning("数据集中没有数据。")
        exit(0)
        
    random.shuffle(datas)

    predictions_match_labels = 0
    processed_items_count = 0
    # evaluation_details = [] # 不再需要这个列表，因为我们不生成JSON摘要了

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "image_path", "predicted_reply", "true_label",
            "logprob_A", "logprob_B", "prob_A", "prob_B"
        ])

        # 如果不是 quiet 模式，保留 tqdm 进度条，否则禁用它
        iterable_datas = datas if args.quiet else tqdm(datas, desc="评估进度")

        for data_item in iterable_datas: # 修改这里
            image_path = data_item.get("images")
            messages_from_data = data_item.get("messages")

            if not image_path or not messages_from_data or len(messages_from_data) < 2:
                # logger.warning(f"数据项格式不完整，跳过: {data_item}") # 注释掉
                continue
            
            if not os.path.exists(image_path):
                # logger.warning(f"图片文件未找到，跳过: {image_path}") # 注释掉
                continue
            
            user_prompt_message = messages_from_data[0]
            true_label = messages_from_data[1]["content"].strip().upper()

            model_input_messages = [user_prompt_message]
            image_paths_for_request = [image_path]

            request_config_kwargs_parsed = {}
            if args.request_config_extra:
                try:
                    request_config_kwargs_parsed = json.loads(args.request_config_extra)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析 --request_config_extra 失败: {e}. 将使用默认 RequestConfig。")
            
            predicted_text, full_response = get_model_response(
                engine,
                model_input_messages,
                image_paths_for_request,
                model_max_tokens=args.max_new_tokens,
                temp=args.temperature,
                request_config_kwargs=request_config_kwargs_parsed
            )

            if predicted_text is None:
                # logger.warning(f"图片 {image_path} 未能获取模型回复。") # 注释掉
                csv_writer.writerow([image_path, "ERROR", true_label, "N/A", "N/A", "N/A", "N/A"])
                continue

            predicted_reply_cleaned = predicted_text.strip().upper()

            target_tokens = ["a", "b"]
            logprobs_map = {}
            prob_A, prob_B = 0.0, 0.0

            if full_response and hasattr(full_response, 'choices') and full_response.choices:
                logprobs_map = extract_token_logprobs_from_swift(full_response.choices[0], target_tokens)
            
            logprob_A = logprobs_map.get("a", -float('inf'))
            logprob_B = logprobs_map.get("b", -float('inf'))

            if logprob_A != -float('inf') and logprob_B != -float('inf'):
                logprobs_tensor = torch.tensor([logprob_A, logprob_B])
                probabilities = torch.softmax(logprobs_tensor, dim=0)
                prob_A = probabilities[0].item()
                prob_B = probabilities[1].item()
            
            csv_writer.writerow([
                image_path, predicted_reply_cleaned, true_label,
                f"{logprob_A:.4f}", f"{logprob_B:.4f}",
                f"{prob_A:.4f}", f"{prob_B:.4f}"
            ])

            if predicted_reply_cleaned == true_label:
                predictions_match_labels += 1
            processed_items_count += 1
            
            # 不再需要追加到 evaluation_details
            # evaluation_details.append({ ... })

    if processed_items_count > 0:
        accuracy = predictions_match_labels / processed_items_count
        # 最终的准确率信息可以考虑保留打印，或者也根据 quiet 参数决定
        final_summary_message = (
            f"\n--- 最终评估结果 ---\n"
            f"总处理图片数: {processed_items_count}\n"
            f"预测正确数: {predictions_match_labels}\n"
            f"准确率 (Accuracy): {accuracy:.4f}"
        )
        if not args.quiet:
            print(final_summary_message) # 如果不是安静模式，打印总结
        else: # 即使在安静模式，也可以考虑将总结信息用 logger.warning 输出，确保能看到
            logger.warning(final_summary_message)


    else:
        if not args.quiet:
            print("没有处理任何有效的图片数据，无法计算准确率。")
        else:
            logger.warning("没有处理任何有效的图片数据，无法计算准确率。")

    # 移除了JSON文件的保存部分
    # try:
    #     with open(args.output_json, 'w', encoding='utf-8') as f:
    #         json.dump(evaluation_details, f, indent=2, ensure_ascii=False)
    #     logger.info(f"详细评估结果已保存到: {args.output_json}")
    # except Exception as e:
    #     logger.error(f"保存详细评估结果到JSON文件失败: {e}")

    if not args.quiet:
        print(f"评估完成。CSV结果已保存到: {args.output_csv}")
    else:
        logger.warning(f"评估完成。CSV结果已保存到: {args.output_csv}")


if __name__ == '__main__':
    main()