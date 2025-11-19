
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
experiment1_data_generator_new.py

单文件、一键运行的测试数据生成程序（最终版）：

1. 从预定义的“已有拓扑库”中选择网络拓扑（非随机、可复现）；
2. 根据 GPU 型录与“按度排序”的分配策略，为每个节点分配显卡，得到节点算力 & 显存能力；
3. 选择一个固定的模型 model_name 和聚合组数 G，把 L 层聚合为 G 组，
   再在 G 上用隔板法“枚举所有切分方案”（你通过 K 和 G 控制组合规模）；
4. 对每个切分方案，估算每个子模块的资源需求：
   - 每 token 算力需求（TFLOPs/token）
   - 显存需求（权重 + KV）（GB）
   - 模块边界输出数据量（boundary_data_mb，单位：MB/s/用户，Activation + KV）
5. 以全排列方式遍历以下维度：
   - 拓扑结构 topology_name
   - 链路带宽 link_bandwidth_gbps
   - 链路带宽价格 link_price_per_gbps_month
   - GPU 型号集合 gpu_set
   - GPU 分配策略 map_mode（高连接度配高算力/高连接度配低算力）
   - 切分子模块数 K
   - 用户套餐 pricing_profile: {user_price_per_month, tokens_per_user}
     - user_price_per_month: 用户月费（$/month）
     - tokens_per_user: 单个用户的最低输出速率下限（token/sec）
6. 模型名称 model_name 和 G 组数不参与全排列，你在配置区选定一个即可，想换就改一行。
7. 支持分批写盘 + 最终合并为一个 CSV。

所有可调参数都在“全局配置区 CONFIG”中，只需改顶部即可，一键运行。
"""

import os
import json
import time
import itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 一、全局配置区（所有需要手动调的参数都放这里）
# ============================================================

# 1. 输出路径与批次设置
OUTPUT_DIR = "data\\Correctness_test\\test_data"      # 所有输出的根目录
BATCH_SIZE = 5000                       # 每多少条记录写一个批次文件
FINAL_CSV_NAME = "experiment_3n_3m_all.csv"  # 合并后的总表名

# 2. 拓扑库（已有拓扑，非随机、可复现）
#    当前示例：ny20_deg5（20节点、平均度数约5），可以在这里再加别的拓扑。
TOPOLOGY_LIBRARY: Dict[str, Dict[str, Any]] = {
    "toy3_chain": {
        "description": "3-node line topology: 0-1-2",
        "adjacency": [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
    }
}

# 3. GPU 型录（TFLOPs / 显存 / 显存价格）
GPU_CATALOG: Dict[str, Dict[str, Any]] = {
    "ToyGPU_Small": {
        "name": "ToyGPU_Small",
        "G_TFLOPS": 4.0,                     # 算力随便给一个比较大的，避免算力成为瓶颈
        "VRAM_bytes": int(4 * 1024**3),      # 4 GB 显存
        "cost_per_GB_month": 1.0,           # 便宜一点
    },
    "ToyGPU_Big": {
        "name": "ToyGPU_Big",
        "G_TFLOPS": 8.0,                     # 更大算力
        "VRAM_bytes": int(8 * 1024**3),      # 8 GB 显存
        "cost_per_GB_month": 2.0,           # 显存更贵
    }
}

# 4. GPU 排序策略（rank_order 固定，不参与全排列）
GPU_ASSIGNMENT_CONFIG: Dict[str, Any] = {
    # 度排序方式：desc = 度高在前；asc = 度低在前
    "rank_order": "desc",
}

# 5. 模型型录（多模型、多参数量），你可以在下面添加更多模型
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "ToyModel-3L": {
        "name": "ToyModel-3L",
        "num_layers": 3,                 # 就 3 层
        "total_params": float(1e6),      # 100 万参数，权重显存可以忽略
        "bytes_per_param": 2,            # 按 FP16 算
        "hidden_size": 64,               # 小一点的 hidden_size
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        # 让每层的 flops_per_token 比较简单：比如 ~1e9 FLOPs/层/token
        "flops_per_token_per_layer": float(1e9),
        # 每个边界的 Activation 大致 1 MB 级别，方便带宽起作用
        "activation_mb_per_boundary": 1.0,
    }
}

# 6. 运行时配置（KV / Activation 计算用），不参与全排列
RUN_CONFIG: Dict[str, Any] = {
    "seq_len": 16,   # KV cache 中的上下文长度（用于显存估算）
    "batch_size": 1,   # 批量大小
    # tokens_per_user 不在这里固定，而是在 pricing_profiles 中按套餐来设置
}

# 7. 实验组合配置（真正参与全排列的维度）
GENERATION_CONFIG: Dict[str, Any] = {
    # （1）拓扑结构：从已有拓扑库中选择（可以多个）
    "topology_names": ["toy3_chain"],

    # （2）模型名称（固定，不参与全排列；想换模型改这里）
    "model_name": "ToyModel-3L",

    # （3）聚合组数 G（固定，不参与全排列；想换 G 改这里）
    "G": 3,

    # （4）切分子模块数 K（参与全排列）
    "K_list": [3],

    # （5）链路带宽 Gbps（参与全排列）
    "link_bandwidth_gbps_list": [1.0],

    # （6）链路带宽价格 $/Gbps/月（参与全排列）
    "link_price_per_gbps_month_list": [1.0],

    # （7）GPU 型号集合（参与全排列）——每个元素是一个 GPU 型号列表
    "gpu_set_list": [
        ["ToyGPU_Small", "ToyGPU_Big"]
    ],

    # （8）GPU 分配策略 map_mode（参与全排列）
    #     - "high_to_high": 度高 → 分配 gpu_set 中“高档”的显卡
    #     - "high_to_low":  度高 → 分配 gpu_set 中“低档”的显卡
    "gpu_map_mode_list": ["high_to_high"],

    # （9）用户套餐 profiles（参与全排列）
    #     每个元素是 {user_price_per_month, tokens_per_user}
    #     - user_price_per_month: 单个用户月费（单位：$/month）
    #     - tokens_per_user: 每个用户的最低输出速率下限（单位：token/sec）
    "pricing_profiles": [
        {"user_price_per_month": 10.0, "tokens_per_user": 2}
    ],
}

# ============================================================
# 二、通用工具函数：拓扑 & GPU 分配
# ============================================================

def get_adjacency_matrix(topology_name: str) -> np.ndarray:
    """从 TOPOLOGY_LIBRARY 中取出邻接矩阵"""
    topo = TOPOLOGY_LIBRARY[topology_name]
    adj = np.array(topo["adjacency"], dtype=int)
    return adj


def compute_distance_matrix(adj: np.ndarray) -> np.ndarray:
    """
    根据邻接矩阵计算无权图最短路径距离（以跳数为单位）
    若不连通则距离设为一个较大值（这里用 999）。
    """
    n = adj.shape[0]
    dist = np.full((n, n), 999, dtype=int)
    np.fill_diagonal(dist, 0)

    for src in range(n):
        visited = [False] * n
        visited[src] = True
        queue = [(src, 0)]
        while queue:
            u, d = queue.pop(0)
            dist[src, u] = d
            for v in range(n):
                if adj[u, v] and not visited[v]:
                    visited[v] = True
                    queue.append((v, d + 1))
    return dist


def compute_bandwidth_matrix(adj: np.ndarray, link_bandwidth_gbps: float) -> np.ndarray:
    """
    根据邻接矩阵生成带宽矩阵：
    - 有边的位置 = link_bandwidth_gbps
    - 无边的位置 = 0
    """
    n = adj.shape[0]
    bw = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                bw[i, j] = link_bandwidth_gbps
    return bw


def assign_gpus_by_degree(
    adj: np.ndarray,
    gpu_set_names: List[str],
    rank_order: str,
    map_mode: str,
) -> List[Dict[str, Any]]:
    """
    按节点度数（degree_rank）为每个节点分配 GPU：
    - 先算出每个节点的度数；
    - 按度数排序（asc/desc）；
    - 使用“段法”分配 gpu_set_names 中的型号；
    - map_mode 决定“度高配高档显卡”还是“度高配低档显卡”。

    返回一个长度为 N 的列表，每个元素是节点的 GPU 信息 dict：
    {
        "node_index": i,
        "gpu_type": "...",
        "G_TFLOPS": ...,
        "VRAM_bytes": ...,
        "cost_per_GB_month": ...,
        "degree": ...
    }
    """
    n = adj.shape[0]
    degrees = adj.sum(axis=1).astype(int)

    # 度排序：desc = 度高在前；asc = 度低在前
    reverse = (rank_order == "desc")
    node_indices = list(range(n))
    node_indices_sorted = sorted(node_indices, key=lambda i: (degrees[i], i), reverse=reverse)

    # 段法分配数量
    S = len(gpu_set_names)
    base = n // S
    rem = n % S
    counts = [base] * S
    for i in range(rem):
        counts[i] += 1

    # gpu_set_names 通常按“从低档到高档”列出
    if map_mode == "high_to_high":
        # 度高的节点分配到 gpu_set_names 的“高档”（右端）
        band_indices = list(reversed(range(S)))
    elif map_mode == "high_to_low":
        band_indices = list(range(S))
    else:
        raise ValueError(f"Unknown map_mode: {map_mode}")

    # 生成 (node_index, gpu_name) 的对应关系
    assignment_pairs: List[Tuple[int, str]] = []
    pos = 0
    for band_idx in band_indices:
        c = counts[band_idx]
        gpu_name = gpu_set_names[band_idx]
        for _ in range(c):
            node_index = node_indices_sorted[pos]
            pos += 1
            assignment_pairs.append((node_index, gpu_name))

    # 写成按 node_index 顺序的列表
    result = []
    for i in range(n):
        gpu_name = next(name for idx, name in assignment_pairs if idx == i)
        gpu_info = GPU_CATALOG[gpu_name]
        result.append({
            "node_index": i,
            "gpu_type": gpu_info["name"],
            "G_TFLOPS": gpu_info["G_TFLOPS"],
            "VRAM_bytes": gpu_info["VRAM_bytes"],
            "cost_per_GB_month": gpu_info["cost_per_GB_month"],
            "degree": int(degrees[i]),
        })
    return result


# ============================================================
# 三、模型切分：G 组 + 隔板法枚举全部切分方案 + 资源需求估算
# ============================================================

def group_layers_evenly(num_layers: int, G: int) -> List[int]:
    """
    把 num_layers 层，尽量平均地分为 G 组。
    例如 L=80, G=10 -> [8,8,8,8,8,8,8,8,8,8]；
    L=82, G=10 -> [9,9,9,9,8,8,8,8,8,8]
    """
    base = num_layers // G
    rem = num_layers % G
    groups = [base + (1 if i < rem else 0) for i in range(G)]
    assert sum(groups) == num_layers
    return groups


def enumerate_compositions(G: int, K: int) -> List[List[int]]:
    """
    在整数 G 上枚举所有将其划分为 K 个正整数段的有序分法（隔板法）。
    例如 G=5,K=2:
      [1,4], [2,3], [3,2], [4,1]
    """
    if K < 1 or K > G:
        raise ValueError(f"Invalid composition: G={G}, K={K}")
    if K == 1:
        return [[G]]
    positions = range(1, G)
    compositions = []
    for bars in itertools.combinations(positions, K - 1):
        prev = 0
        segs = []
        for b in bars:
            segs.append(b - prev)
            prev = b
        segs.append(G - prev)
        compositions.append(segs)
    return compositions


def expand_groups_to_layers(group_sizes: List[int], comp: List[int]) -> List[int]:
    """
    已知每组层数 group_sizes[g]，以及“每段包含多少组”的 composition comp,
    展开为“每段包含多少层”的列表。
    """
    layers = []
    idx = 0
    for g_count in comp:
        s = sum(group_sizes[idx: idx + g_count])
        layers.append(s)
        idx += g_count
    return layers


def estimate_partition_resources(
    model_cfg: Dict[str, Any],
    layers_per_segment: List[int],
    run_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    对给定模型 & 切分方案，估算每个子模块的资源需求：
    - compute_tflops_per_token_i
    - memory_gb_i（权重 + KV）
    - boundary_data_mb_i：单用户在该边界上的数据流量速率（MB/s/user）

    这里 tokens_per_user 被理解为“每个用户的最低输出速率下限（token/sec）”。

    对于边界 i -> i+1：
      - Activation：每生成一个 token，隐藏状态过一次边界 → act_bytes_per_token * tokens_per_user (bytes/sec)
      - KV：每生成一个 token，需要为“之前所有层”生成新的 KV 并传给下游 →
            per-token KV bytes ≈ kv_bytes_per_token_per_layer * layers_before
            再乘 tokens_per_user 得到 KV bytes/sec
    """
    L = int(model_cfg["num_layers"])
    total_params = float(model_cfg["total_params"])
    bytes_per_param = int(model_cfg["bytes_per_param"])
    hidden_size = int(model_cfg["hidden_size"])
    num_heads = int(model_cfg["num_attention_heads"])
    num_kv_heads = int(model_cfg["num_key_value_heads"])
    flops_per_token_per_layer = float(model_cfg["flops_per_token_per_layer"])

    seq_len = int(run_cfg["seq_len"])
    batch_size = int(run_cfg["batch_size"])
    tokens_per_user = int(run_cfg["tokens_per_user"])  # token/sec

    bytes_per_act = 2  # 假设激活/KV 都用 FP16

    # 1. 权重显存总量
    total_weights_bytes = total_params * bytes_per_param
    total_weights_gb = total_weights_bytes / (1024**3)

    # 2. KV：每层每 token 的 KV 字节（用于显存与跨边界开销估算）
    head_dim = hidden_size / num_heads
    kv_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * bytes_per_param  # K+V 两份

    # 3. Activation：每 token 隐藏状态的字节数
    act_bytes_per_token = hidden_size * bytes_per_act

    segments = []
    K = len(layers_per_segment)

    # 前缀层数 prefix_layers[i] = 前 i 段总层数
    prefix_layers = [0]
    for li in layers_per_segment:
        prefix_layers.append(prefix_layers[-1] + li)

    for i, li in enumerate(layers_per_segment):
        # 3.1 计算量：flops_per_token_per_layer × li
        compute_tflops_per_token = flops_per_token_per_layer * li / (10**12)

        # 3.2 权重显存：按层数比例分配
        frac = li / L
        weights_gb_i = total_weights_gb * frac

        # 3.3 KV 显存（驻留）：每层 KV 保存 seq_len × batch_size 的上下文
        kv_bytes_i = kv_bytes_per_token_per_layer * seq_len * batch_size * li
        kv_gb_i = kv_bytes_i / (1024**3)

        memory_gb = weights_gb_i + kv_gb_i

        # 3.4 边界数据量（Activation + KV），只对 i < K-1 的边界有意义
        if i < K - 1:
            # Activation 部分：每个 token 的 act 都要过一次边界 → bytes/sec
            act_bytes_per_sec = act_bytes_per_token * tokens_per_user

            # KV 部分：
            # 每生成一个 token，需要为之前所有层生成新的 KV 并传给下游
            layers_before = prefix_layers[i + 1]  # 前 i+1 段的总层数
            kv_bytes_per_token_cross = kv_bytes_per_token_per_layer * layers_before
            kv_bytes_per_sec = kv_bytes_per_token_cross * tokens_per_user

            boundary_bytes_per_sec = act_bytes_per_sec + kv_bytes_per_sec
            boundary_mb = boundary_bytes_per_sec / (1024**2)  # MB/sec per user
        else:
            boundary_mb = 0.0

        segments.append({
            "segment_index": i,
            "layers": li,
            "compute_tflops_per_token": float(round(compute_tflops_per_token, 6)),
            "compute_tflops_per_user_per_sec": float(round(compute_tflops_per_token * tokens_per_user, 6)),
            "memory_gb": float(round(memory_gb, 6)),
            # 含义：单用户在该边界的流量速率（MB/s）
            "boundary_data_mb": float(round(boundary_mb, 6)),
        })

    summary = {
        "total_compute_tflops_per_token": float(round(sum(s["compute_tflops_per_token"] for s in segments), 6)),
        "total_memory_gb": float(round(sum(s["memory_gb"] for s in segments), 6)),
        "max_boundary_data_mb": float(round(max(s["boundary_data_mb"] for s in segments), 6)),
    }

    return {
        "segments_layers": layers_per_segment,
        "segments": segments,
        "summary": summary,
    }


# ============================================================
# 四、生成主逻辑（全排列 + 批次写盘 + 合并）
# ============================================================

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def generate_experiment_data() -> None:
    """
    核心生成函数：
    - 遍历以下维度的笛卡尔积：
        * topology_name
        * K_segments
        * link_bandwidth_gbps
        * link_price_per_gbps_month
        * gpu_set
        * gpu_map_mode
        * pricing_profile (user_price_per_month, tokens_per_user)
      model_name 与 G 固定。
    - 对每个组合：
        * 使用已有拓扑邻接矩阵；
        * 计算距离矩阵与带宽矩阵；
        * 按度排序 + gpu_set + map_mode 分配 GPU；
        * 使用模型与 G 聚合层 → 枚举 G->K 的所有隔板法切分；
        * 对每种切分方案估算模块资源需求；
        * 生成一行记录写入 buffer。
    - 按 BATCH_SIZE 写批次 CSV，最后合并为一个总表。
    """
    ensure_dir(OUTPUT_DIR)
    batch_files: List[str] = []
    buffer: List[Dict[str, Any]] = []
    batch_idx = 0

    topology_names = GENERATION_CONFIG["topology_names"]
    model_name = GENERATION_CONFIG["model_name"]
    G = GENERATION_CONFIG["G"]
    K_list = GENERATION_CONFIG["K_list"]
    link_bw_list = GENERATION_CONFIG["link_bandwidth_gbps_list"]
    link_price_list = GENERATION_CONFIG["link_price_per_gbps_month_list"]
    gpu_set_list = GENERATION_CONFIG["gpu_set_list"]
    gpu_map_mode_list = GENERATION_CONFIG["gpu_map_mode_list"]
    pricing_profiles = GENERATION_CONFIG["pricing_profiles"]

    rank_order = GPU_ASSIGNMENT_CONFIG["rank_order"]

    total_combos = (
        len(topology_names)
        * len(K_list)
        * len(link_bw_list)
        * len(link_price_list)
        * len(gpu_set_list)
        * len(gpu_map_mode_list)
        * len(pricing_profiles)
    )

    print("=== 实验1数据生成开始 ===")
    print(f"配置组合数（不含 G->K 切分方案）约为：{total_combos}")
    start_time = time.time()

    model_cfg = MODEL_CATALOG[model_name]
    L = int(model_cfg["num_layers"])
    if G > L:
        raise RuntimeError(f"G={G} 不能大于模型 {model_name} 的总层数 L={L}")

    for topo_name, K, link_bw, link_price, gpu_set, map_mode, pricing in itertools.product(
        topology_names, K_list, link_bw_list, link_price_list, gpu_set_list, gpu_map_mode_list, pricing_profiles
    ):
        if K > G:
            print(f"[跳过] 拓扑={topo_name}, K={K} > G={G}")
            continue

        # 拿拓扑
        adj = get_adjacency_matrix(topo_name)
        n_nodes = adj.shape[0]
        dist = compute_distance_matrix(adj)
        bw_mat = compute_bandwidth_matrix(adj, link_bw)

        # GPU 分配
        per_node_gpu = assign_gpus_by_degree(adj, gpu_set, rank_order, map_mode)

        # 聚合层为 G 组
        group_sizes = group_layers_evenly(L, G)
        # G 上的隔板法组合
        compositions = enumerate_compositions(G, K)

        # 构造运行配置（注入 tokens_per_user）
        run_cfg_base = dict(RUN_CONFIG)
        run_cfg_base["tokens_per_user"] = int(pricing["tokens_per_user"])

        for part_idx, comp in enumerate(compositions):
            layers_per_seg = expand_groups_to_layers(group_sizes, comp)
            part_res = estimate_partition_resources(model_cfg, layers_per_seg, run_cfg_base)

            row = {
                # 场景参数
                "topology_name": topo_name,
                "node_count": n_nodes,
                "model_name": model_name,
                "G_groups": G,
                "K_segments": K,
                "partition_index": part_idx,

                # 网络参数
                "link_bandwidth_gbps": link_bw,
                "link_price_per_gbps_month": link_price,

                # GPU 分配策略参数
                "gpu_set": ",".join(gpu_set),
                "gpu_map_mode": map_mode,
                "gpu_rank_order": rank_order,

                # 用户套餐参数（均为“按月计费 & token/sec 下限”语义）
                "user_price_per_month": pricing["user_price_per_month"],
                "tokens_per_user": pricing["tokens_per_user"],

                # 运行态参数
                "seq_len": RUN_CONFIG["seq_len"],
                "batch_size": RUN_CONFIG["batch_size"],

                # 模型信息
                "model_num_layers": L,
                "model_total_params": model_cfg["total_params"],
                "model_hidden_size": model_cfg["hidden_size"],
                "model_num_heads": model_cfg["num_attention_heads"],
                "model_num_kv_heads": model_cfg["num_key_value_heads"],

                # 拓扑矩阵（JSON 序列化）
                "adjacency_json": json.dumps(adj.tolist(), ensure_ascii=False),
                "distance_json": json.dumps(dist.tolist(), ensure_ascii=False),
                "bandwidth_json": json.dumps(bw_mat.tolist(), ensure_ascii=False),

                # 节点 GPU 分配（JSON）
                "per_node_gpu_json": json.dumps(per_node_gpu, ensure_ascii=False),

                # 切分方案 & 子模块资源（JSON）
                "segments_layers_json": json.dumps(part_res["segments_layers"], ensure_ascii=False),
                "segments_detail_json": json.dumps(part_res["segments"], ensure_ascii=False),
                "segments_summary_json": json.dumps(part_res["summary"], ensure_ascii=False),
            }

            buffer.append(row)

            if len(buffer) >= BATCH_SIZE:
                batch_path = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:04d}.csv")
                df = pd.DataFrame(buffer)
                df.to_csv(batch_path, index=False, encoding="utf-8")
                print(f"[写入批次] {batch_path} ({len(buffer)} 行)")
                buffer.clear()
                batch_files.append(batch_path)
                batch_idx += 1

    # 写最后一个批次
    if buffer:
        batch_path = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:04d}.csv")
        df = pd.DataFrame(buffer)
        df.to_csv(batch_path, index=False, encoding="utf-8")
        print(f"[写入批次] {batch_path} ({len(buffer)} 行)")
        buffer.clear()
        batch_files.append(batch_path)

    # 合并
    if batch_files:
        all_dfs = [pd.read_csv(p) for p in batch_files]
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_path = os.path.join(OUTPUT_DIR, FINAL_CSV_NAME)
        final_df.to_csv(final_path, index=False, encoding="utf-8")
        print(f"[合并完成] {final_path} 共 {len(final_df)} 行")
    else:
        print("[警告] 未生成任何批次文件，可能是配置导致没有有效组合。")

    elapsed = time.time() - start_time
    print(f"=== 实验1数据生成结束，用时 {elapsed:.2f} 秒 ===")


# ============================================================
# 五、入口
# ============================================================

if __name__ == "__main__":
    generate_experiment_data()
