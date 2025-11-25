#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多线程并行版实验1测试（适配新版多链部署优化器）

功能：
1. 读取 experiment1_data_generator_new.py 生成的 CSV 测试数据；
2. 将每一行（一个测试配置）转换为优化器需要的 test_data 字典；
3. 调用：
    - MultiFunctionOptimizer.optimize_for_profit()  （多功能多链部署）
    - SingleFunctionOptimizer.single_func_deployment()（单功能多链部署，节点不重复）
4. 收集结果，输出为一个汇总 CSV 并支持断点续跑。

注意：
- 已经适配新的返回格式（8 元组，增加 chain_count）。
- 带宽单位保持：链路容量为 MB/s，边界数据为 MB/s per user。
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import argparse
import json
import traceback
import concurrent.futures
import multiprocessing
from tqdm import tqdm

# ----------------------------------------------------------------------
# 路径设置：适配你的当前目录结构
# ----------------------------------------------------------------------
# 当前文件：根目录/src/experiment/test_experiment1.py
current_dir = os.path.dirname(os.path.abspath(__file__))    # .../src/experiment
src_dir = os.path.dirname(current_dir)                      # .../src
project_root = os.path.dirname(src_dir)                     # .../（根目录）

algorithm_dir = os.path.join(src_dir, 'algorithm')          # 根目录/src/algorithm

sys.path.append(src_dir)
sys.path.append(algorithm_dir)

# 输出目录：根目录/data/analysis/table
output_dir = os.path.join(project_root, 'data', 'analysis', 'table')
os.makedirs(output_dir, exist_ok=True)

# 导入优化器
from algorithm.multi_all_node import MultiFunctionOptimizer
from algorithm.single_all_node_multi import SingleFunctionOptimizer
from algorithm.shortest_path_all_node import ShortestPathOptimizer

# ----------------------------------------------------------------------
# 将一行 CSV 数据转换为优化器使用的 test_data 字典
# ----------------------------------------------------------------------
def build_test_data_from_row(row):
    """
    根据 experiment1_data_generator_new.py 输出的一行 CSV，
    构造 SingleFunctionOptimizer / MultiFunctionOptimizer 统一使用的 test_data 字典。

    关键单位约定：
    - computation_capacity: [TFLOPs/s, GB]
    - resource_demands: [TFLOPs/s per user, GB per user]
    - data_sizes: MB/s per user
    - bandwidth_matrix: MB/s 链路容量
    - node_costs: [gpu_cost_per_unit_compute, memory_cost_per_GB]
    - bandwidth_cost: $ / (MB/s · month)
    - profit_per_user: $ / user / month
    """
    # 1. 基本标量参数
    test_id = int(row.get('test_data_id', 0))
    node_count = int(row['node_count'])
    tokens_per_user = float(row['tokens_per_user'])
    link_bandwidth_gbps = float(row['link_bandwidth_gbps'])
    link_price_per_gbps_month = float(row['link_price_per_gbps_month'])
    user_price_per_month = float(row['user_price_per_month'])

    # 2. JSON 字段解析
    adjacency = json.loads(row['adjacency_json'])
    distance = json.loads(row['distance_json'])
    bandwidth_gbps_matrix = json.loads(row['bandwidth_json'])
    per_node_gpu = json.loads(row['per_node_gpu_json'])
    segments_layers = json.loads(row['segments_layers_json'])
    segments_detail = json.loads(row['segments_detail_json'])
    segments_summary = json.loads(row['segments_summary_json'])

    module_count = len(segments_detail)

    # 3. 节点资源与成本：computation_capacity / node_costs
    compute_utilization_factor = 0.4  # GPU 利用率比例，可根据需要调整

    computation_capacity = []
    node_costs = []
    total_compute_cap = 0.0
    total_mem_cap = 0.0

    for gpu_info in per_node_gpu:
        g_tflops = float(gpu_info['G_TFLOPS'])         # TFLOPs/s 理论峰值
        vram_bytes = float(gpu_info['VRAM_bytes'])     # 字节
        cost_per_gb_month = float(gpu_info['cost_per_GB_month'])

        compute_cap = g_tflops * compute_utilization_factor
        mem_cap_gb = vram_bytes / (1024.0 ** 3)

        computation_capacity.append([compute_cap, mem_cap_gb])
        # 当前建模：只对显存计价，算力成本为 0
        node_costs.append([0.0, cost_per_gb_month])

        total_compute_cap += compute_cap
        total_mem_cap += mem_cap_gb

    avg_compute_ability = total_compute_cap / node_count if node_count > 0 else 0.0
    avg_memory_ability = total_mem_cap / node_count if node_count > 0 else 0.0

    # 4. 模块资源需求（resource_demands）+ 边界数据速率（data_sizes）
    resource_demands = []
    data_sizes = []

    for idx, seg in enumerate(segments_detail):
        # 算力需求（TFLOPs/s per user）
        comp_tok = float(seg.get('compute_tflops_per_token', 0.0))
        comp_user = seg.get('compute_tflops_per_user_per_sec', None)
        if comp_user is None:
            comp_user = seg.get('compute_tflops_per_user', None)
        if comp_user is None:
            comp_user = comp_tok * tokens_per_user  # 兜底：按 token/sec 线性扩展
        comp_user = float(comp_user)

        # 显存需求（GB）
        mem_gb = float(seg.get('memory_gb', 0.0))

        resource_demands.append([comp_user, mem_gb])

        # 边界数据：MB/s per user（最后一个模块之后没有边界）
        if idx < module_count - 1:
            boundary_mb = float(seg.get('boundary_data_mb', 0.0))
            data_sizes.append(boundary_mb)

    # 5. 链路带宽矩阵：Gbps → MB/s
    GBPS_TO_MBPS = 125.0
    bandwidth_matrix = [
        [float(x) * GBPS_TO_MBPS for x in row_]
        for row_ in bandwidth_gbps_matrix
    ]

    # 6. 带宽成本：$/ (MB/s · month)
    bandwidth_cost = link_price_per_gbps_month / GBPS_TO_MBPS if GBPS_TO_MBPS > 0 else 0.0

    # 7. 利润 / 成本参数
    profit_per_user = user_price_per_month
    gpu_cost = 0.0 # TFLOPS
    memory_cost = (
        sum(cost[1] for cost in node_costs) / len(node_costs)
        if node_costs else 0.0
    )

    # 8. 拓扑平均度数（用于分析）
    degrees = [sum(1 for x in row_ if x != 0) for row_ in adjacency]
    topology_degree = float(sum(degrees) / len(degrees)) if degrees else 0.0

    # 9. 组装 test_data 字典
    test_data = {
        # 基础标识
        "test_data_id": test_id,

        # 拓扑与模型基础信息
        "topology_name": row.get('topology_name', ''),
        "node_count": node_count,
        "model_name": row.get('model_name', ''),
        "G_groups": int(row.get('G_groups', len(segments_layers))),
        "K_segments": int(row.get('K_segments', module_count)),
        "partition_index": int(row.get('partition_index', 0)),

        # 模型规格
        "model_num_layers": int(row.get('model_num_layers', 0)),
        "model_total_params": float(row.get('model_total_params', 0.0)),
        "model_hidden_size": int(row.get('model_hidden_size', 0)),
        "model_num_heads": int(row.get('model_num_heads', 0)),
        "model_num_kv_heads": int(row.get('model_num_kv_heads', 0)),

        # 简化分析字段
        "model_size": float(row.get('model_total_params', 0.0)),
        "bandwidth": float(row.get('link_bandwidth_gbps', 0.0)),  # 仍保留 Gbps 作为高层特征
        "topology_degree": topology_degree,
        "computation_ability": avg_compute_ability,
        "memory_ability": avg_memory_ability,

        # 用户套餐
        "user_price_per_month": user_price_per_month,
        "tokens_per_user": tokens_per_user,

        # 优化器核心字段
        "module_count": module_count,
        "computation_capacity": computation_capacity,
        "resource_demands": resource_demands,
        "data_sizes": data_sizes,
        "bandwidth_matrix": bandwidth_matrix,
        "memory_cost": memory_cost,
        "bandwidth_cost": bandwidth_cost,
        "profit_per_user": profit_per_user,
        "node_costs": node_costs,
        "distance_matrix": distance,
    }

    return test_data


# ----------------------------------------------------------------------
# 单个测试用例处理逻辑
# ----------------------------------------------------------------------
def process_test_case(row):
    """
    处理单个测试用例（CSV 中的一行记录）

    Args:
        row: dict，一行 CSV 转换得到的字典

    Returns:
        dict: 汇总结果（用于最终写入结果表）
    """
    # 从 CSV 行生成 test_data
    test_data = build_test_data_from_row(row)
    test_id = test_data.get('test_data_id', 0)

    print(f"\n正在处理测试数据 ID: {test_id}")
    print(
        f"参数: profit_per_user={test_data.get('profit_per_user', 0)}, "
        f"module_count={test_data.get('module_count', 0)}, "
        f"bandwidth={test_data.get('bandwidth', 0)}, "
        f"model_size={test_data.get('model_size', 0)}, "
        f"topology_degree={test_data.get('topology_degree', 0)}, "
        f"memory_cost={test_data.get('memory_cost', 0)}"
    )

    # 初始化结果字典（先写一些基础参数）
    result = {
        'test_data_id': test_id,
        'profit_per_user': test_data.get('profit_per_user', 0),
        'model_size': test_data.get('model_size', 0),
        'module_count': test_data.get('module_count', 0),
        'topology_degree': test_data.get('topology_degree', 0),
        'bandwidth': test_data.get('bandwidth', 0),
        'gpu_cost': test_data.get('gpu_cost', 0),
        'memory_cost': test_data.get('memory_cost', 0),
        'bandwidth_cost': test_data.get('bandwidth_cost', 0),
        'computation_ability': test_data.get('computation_ability', 0),
        'memory_ability': test_data.get('memory_ability', 0),
    }

    try:
        print(f"ID {test_id}: 开始处理任务...")

        # ---------------- 多功能部署优化 ----------------
        try:
            print(f"ID {test_id}: 开始多功能部署优化...")
            multi_optimizer = MultiFunctionOptimizer(test_data)
            print(f"ID {test_id}: 多功能部署优化器初始化完成")
            multi_func_result = multi_optimizer.optimize_for_profit()

            if multi_func_result:
                min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan = multi_func_result

                # 每个 plan 现在是 8 元组：
                # (total_cost, total_deploy_cost, total_comm_cost,
                #  total_profit, total_users, used_nodes_count,
                #  avg_modules_per_node, chain_count)

                # 1）最小成本策略
                if min_cost_plan is not None:
                    result['multi_func_min_cost_cost'] = min_cost_plan[0]
                    result['multi_func_min_cost_deploy_cost'] = min_cost_plan[1]
                    result['multi_func_min_cost_comm_cost'] = min_cost_plan[2]
                    result['multi_func_min_cost_profit'] = min_cost_plan[3]
                    result['multi_func_min_cost_users'] = min_cost_plan[4]
                    result['multi_func_min_cost_nodes'] = min_cost_plan[5]
                    result['multi_func_min_cost_avg_modules'] = min_cost_plan[6]
                    result['multi_func_min_cost_chain_count'] = min_cost_plan[7]
                else:
                    result['multi_func_min_cost_error'] = "无可行方案"

                # 2）最大利润策略
                if max_profit_plan is not None:
                    result['multi_func_profit_cost'] = max_profit_plan[0]
                    result['multi_func_profit_deploy_cost'] = max_profit_plan[1]
                    result['multi_func_profit_comm_cost'] = max_profit_plan[2]
                    result['multi_func_profit_profit'] = max_profit_plan[3]
                    result['multi_func_profit_users'] = max_profit_plan[4]
                    result['multi_func_profit_nodes'] = max_profit_plan[5]
                    result['multi_func_profit_avg_modules'] = max_profit_plan[6]
                    result['multi_func_profit_chain_count'] = max_profit_plan[7]
                    print(
                        f"ID {test_id}: 多功能部署优化完成（最大利润策略），"
                        f"总用户数: {max_profit_plan[4]}, 总利润: {max_profit_plan[3]}, "
                        f"部署链条数: {max_profit_plan[7]}"
                    )
                else:
                    result['multi_func_profit_error'] = "无可行方案"

                # 3）最差利润策略
                if min_profit_plan is not None:
                    result['multi_func_worst_profit_cost'] = min_profit_plan[0]
                    result['multi_func_worst_profit_deploy_cost'] = min_profit_plan[1]
                    result['multi_func_worst_profit_comm_cost'] = min_profit_plan[2]
                    result['multi_func_worst_profit_profit'] = min_profit_plan[3]
                    result['multi_func_worst_profit_users'] = min_profit_plan[4]
                    result['multi_func_worst_profit_nodes'] = min_profit_plan[5]
                    result['multi_func_worst_profit_avg_modules'] = min_profit_plan[6]
                    result['multi_func_worst_profit_chain_count'] = min_profit_plan[7]
                    print(
                        f"ID {test_id}: 多功能部署优化完成（最大用户策略），"
                        f"总用户数: {min_profit_plan[4]}, 总利润: {max_profit_plan[3]}, "
                        f"部署链条数: {min_profit_plan[7]}"
                    )
                else:
                    result['multi_func_worst_profit_error'] = "无可行方案"

                # 4）最大用户量策略
                if max_users_plan is not None:
                    result['multi_func_max_users_cost'] = max_users_plan[0]
                    result['multi_func_max_users_deploy_cost'] = max_users_plan[1]
                    result['multi_func_max_users_comm_cost'] = max_users_plan[2]
                    result['multi_func_max_users_profit'] = max_users_plan[3]
                    result['multi_func_max_users_users'] = max_users_plan[4]
                    result['multi_func_max_users_nodes'] = max_users_plan[5]
                    result['multi_func_max_users_avg_modules'] = max_users_plan[6]
                    result['multi_func_max_users_chain_count'] = max_users_plan[7]
                    print(
                        f"ID {test_id}: 多功能部署优化完成（最大用户策略），"
                        f"总用户数: {max_users_plan[4]}, 总利润: {max_users_plan[3]}, "
                        f"部署链条数: {max_users_plan[7]}"
                    )
                else:
                    result['multi_func_max_users_error'] = "无可行方案"
                    print(f"ID {test_id}: 多功能部署优化失败，无任何策略可行")
            else:
                result['multi_func_error'] = "无法找到任何多功能部署方案"
                print(f"ID {test_id}: 多功能部署优化失败：返回结果为空")
        except Exception as e:
            result['multi_func_error'] = str(e)
            print(f"ID {test_id}: 多功能部署优化出错: {str(e)}")
            traceback.print_exc()

            # ---------------- 分层图最短路部署优化 ----------------
        try:
            print(f"ID {test_id}: 开始分层图最短路部署优化...")
            sp_optimizer = ShortestPathOptimizer(test_data)
            sp_result = sp_optimizer.shortest_path_deployment()

            if sp_result is not None:
                # 成功：sp_result 是一个 dict
                # 先显式清空错误字段
                result['sp_error'] = ""

                result['sp_cost'] = sp_result["total_cost"]
                result['sp_deploy_cost'] = sp_result["total_deploy_cost"]
                result['sp_comm_cost'] = sp_result["total_comm_cost"]
                result['sp_profit'] = sp_result["total_profit"]
                result['sp_users'] = sp_result["total_users"]
                result['sp_nodes'] = sp_result["used_nodes"]
                result['sp_avg_modules'] = sp_result["avg_modules_per_node"]
                result['sp_chain_count'] = sp_result["chain_count"]

                print(
                    f"ID {test_id}: 分层图最短路部署完成，"
                    f"总用户数: {sp_result['total_users']}, 总利润: {sp_result['total_profit']}, "
                    f"部署链条数: {sp_result['chain_count']}"
                )
            else:
                # 无可行方案
                result['sp_error'] = "无法找到任何可行的最短路部署方案"
                print(f"ID {test_id}: 分层图最短路部署失败，无可行方案")
        except Exception as e:
            result['sp_error'] = str(e)
            print(f"ID {test_id}: 分层图最短路部署出错: {str(e)}")


        # ---------------- 单功能部署优化 ----------------
        try:
            print(f"ID {test_id}: 开始单功能部署优化...")
            single_optimizer = SingleFunctionOptimizer(test_data)
            print(f"ID {test_id}: 单功能部署优化器初始化完成")
            single_func_result = single_optimizer.single_func_deployment()

            if single_func_result:
                # 8 元组：
                # (total_cost, total_deploy_cost, total_comm_cost,
                #  total_profit, total_users, used_nodes_count,
                #  avg_modules_per_node, chain_count)
                result['single_func_cost'] = single_func_result[0]
                result['single_func_deploy_cost'] = single_func_result[1]
                result['single_func_comm_cost'] = single_func_result[2]
                result['single_func_profit'] = single_func_result[3]
                result['single_func_users'] = single_func_result[4]
                result['single_func_nodes'] = single_func_result[5]
                result['single_func_avg_modules'] = single_func_result[6]
                result['single_func_chain_count'] = single_func_result[7]

                print(
                    f"ID {test_id}: 单功能部署优化完成，总用户数: {single_func_result[4]}, "
                    f"总利润: {single_func_result[3]}, 部署链条数: {single_func_result[7]}"
                )
            else:
                result['single_func_error'] = "无法找到有效的单功能部署方案"
                print(f"ID {test_id}: 单功能部署优化失败，无可行方案")
        except Exception as e:
            result['single_func_error'] = str(e)
            print(f"ID {test_id}: 单功能部署优化出错: {str(e)}")
            traceback.print_exc()

    except Exception as e:
        result['process_error'] = str(e)
        print(f"ID {test_id}: 处理过程发生错误: {str(e)}")
        traceback.print_exc()

    print(f"ID {test_id}: 处理完成\n" + "-" * 60)
    return result


# ----------------------------------------------------------------------
# 结果保存 & 检查点
# ----------------------------------------------------------------------
def save_checkpoint(results, output_file, checkpoint_file):
    """保存结果和检查点"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    processed_ids = df['test_data_id'].tolist()
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(processed_ids, f, ensure_ascii=False)

    print(f"已保存结果至: {output_file}")
    print(f"已保存检查点至: {checkpoint_file}")


def load_checkpoint(checkpoint_file):
    """加载检查点，返回已处理的测试ID列表"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                processed_ids = json.load(f)
            print(f"已加载检查点，发现 {len(processed_ids)} 条已处理记录")
            return processed_ids
        except Exception as e:
            print(f"加载检查点出错: {e}")
    return []


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    """主函数：读取测试数据，启动多进程处理，生成结果"""
    parser = argparse.ArgumentParser(description='多线程并行处理实验1测试数据（多链部署版）')
    parser.add_argument('--processes', type=int, default=None, help='并行进程数（默认为CPU核心数-1）')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的数据条数（用于调试）')
    parser.add_argument('--output', type=str, default='experiment1_results_parallel_{input_name}.csv',
                        help='输出文件名模式，使用{input_name}表示输入文件名（不含扩展名）')
    parser.add_argument('--batch_size', type=int, default=500, help='批处理大小（每处理多少条保存一次）')
    parser.add_argument('--input', type=str, default='experiment1_all.csv',
                        help='输入文件名或路径（默认 data/test_data/experiment1_all.csv ）')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='输入文件夹路径，如果指定则处理该文件夹下所有CSV文件')

    args = parser.parse_args()

    # 设置进程数
    if args.processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_processes = args.processes

    print(f"=== 开始多线程测试（多链部署版） ===")
    print(f"使用 {num_processes} 个进程并行处理数据")

    input_files = []

    # 1. 若指定 input_dir，则扫描该目录下所有 CSV
    if args.input_dir:
        input_dir = args.input_dir
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(current_dir, input_dir)

        if os.path.exists(input_dir) and os.path.isdir(input_dir):
            print(f"扫描输入文件夹: {input_dir}")
            for filename in os.listdir(input_dir):
                if filename.lower().endswith('.csv'):
                    input_files.append(os.path.join(input_dir, filename))
            print(f"在 {input_dir} 中找到 {len(input_files)} 个CSV文件")
        else:
            print(f"指定的输入文件夹 {input_dir} 不存在或不是目录")
            sys.exit(1)
    else:
        # 2. 单文件模式：根据你的项目结构尝试多个候选路径
        candidate_paths = []
        if os.path.isabs(args.input):
            candidate_paths.append(args.input)
        else:
            # src/experiment 下
            candidate_paths.append(os.path.join(current_dir, args.input))
            # src 下
            candidate_paths.append(os.path.join(src_dir, args.input))
            # 根目录下
            candidate_paths.append(os.path.join(project_root, args.input))
            # 根目录/data/test_data 下（你现在实际存放的位置）
            candidate_paths.append(os.path.join(project_root, 'data', 'test_data', args.input))
            # 根目录/data 下
            candidate_paths.append(os.path.join(project_root, 'data', args.input))

        experiment_file = None
        for p in candidate_paths:
            if os.path.exists(p):
                experiment_file = p
                break

        if experiment_file is None:
            print("未找到测试数据文件。尝试了以下路径：")
            for p in candidate_paths:
                print("  -", p)
            print("请确认 experiment1_all.csv 的实际位置，或使用 --input 传入绝对路径。")
            sys.exit(1)

        input_files.append(experiment_file)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------
    # 逐个输入文件处理
    # --------------------------------------------
    for input_file in input_files:
        input_filename = os.path.basename(input_file)
        input_name = os.path.splitext(input_filename)[0]

        output_filename = args.output.replace('{input_name}', input_name)
        output_file = os.path.join(output_dir, output_filename)
        checkpoint_file = f"{output_file}.checkpoint"

        print(f"\n=== 处理输入文件: {input_file} ===")
        print(f"输出结果将保存至: {output_file}")

        # 读取测试数据文件
        print(f"读取测试数据: {input_file}")
        try:
            df = pd.read_csv(input_file)
            print(f"共 {len(df)} 条测试记录")
        except Exception as e:
            print(f"读取 {input_file} 失败: {str(e)}")
            continue

        # 如果没有 test_data_id 列，自动添加
        if 'test_data_id' not in df.columns:
            df.insert(0, 'test_data_id', range(len(df)))
            print("输入数据中未发现 test_data_id 列，已自动添加顺序 ID。")

        # 加载已处理的检查点
        processed_ids = load_checkpoint(checkpoint_file)

        previous_results = []
        if processed_ids and os.path.exists(output_file):
            try:
                previous_df = pd.read_csv(output_file)
                previous_results = previous_df.to_dict('records')
                print(f"已加载 {len(previous_results)} 条已处理结果")
            except Exception as e:
                print(f"加载之前结果文件出错: {e}")

        # 过滤出未处理数据
        df_filtered = df[~df['test_data_id'].isin(processed_ids)].copy()
        print(f"过滤掉 {len(processed_ids)} 条已处理记录，剩余 {len(df_filtered)} 条待处理")

        if args.limit is not None:
            df_filtered = df_filtered.head(args.limit)
            print(f"限制处理前 {args.limit} 条记录（当前实际处理 {len(df_filtered)} 条）")

        test_data_records = df_filtered.to_dict('records')
        print(f"开始处理 {len(test_data_records)} 条测试记录...")

        all_results = previous_results.copy()
        start_time = time.time()

        if len(test_data_records) > 0:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = {executor.submit(process_test_case, record): record for record in test_data_records}

                batch_results = []
                batch_count = 0

                print(f"\n总共需要处理 {len(futures)} 条记录")

                for i, future in enumerate(
                    tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理测试用例")
                ):
                    try:
                        result = future.result()
                        batch_results.append(result)
                        all_results.append(result)
                        batch_count += 1

                        if batch_count >= args.batch_size:
                            save_checkpoint(all_results, output_file, checkpoint_file)
                            print(
                                f">>> 进度: {i+1}/{len(futures)} "
                                f"({(i+1)/len(futures)*100:.1f}%)"
                            )
                            print(f">>> 已完成 {batch_count} 条记录处理，已保存中间结果")
                            batch_count = 0
                    except Exception as e:
                        print(f"处理任务时出错: {str(e)}")
                        traceback.print_exc()

                # 处理完最后一批
                if batch_results:
                    save_checkpoint(all_results, output_file, checkpoint_file)

        end_time = time.time()
        total_time = end_time - start_time
        records_processed = len(test_data_records)

        print("\n=== 处理完成 ===")
        print(f"处理了 {records_processed} 条记录，用时 {total_time:.2f} 秒")
        if records_processed > 0 and total_time > 0:
            print(f"处理速度: {records_processed / total_time:.2f} 条记录/秒")

        save_checkpoint(all_results, output_file, checkpoint_file)

        # 简单统计
        print("\n=== 结果统计 ===")
        df_results = pd.DataFrame(all_results)
        print(f"总记录数: {len(df_results)}")

        profit_columns = [
            'multi_func_profit_profit',
            'multi_func_max_users_profit',
            'single_func_profit'
        ]

        for col in profit_columns:
            if col in df_results.columns:
                avg_profit = df_results[col].mean()
                print(f"{col}: 平均值 = {avg_profit:.2f}")

        print(f"=== 测试文件 {input_filename} 处理完成 ===")

    print("\n=== 所有文件处理完成 ===")


if __name__ == "__main__":
    main()