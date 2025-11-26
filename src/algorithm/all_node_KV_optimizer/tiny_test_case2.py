# tiny_test_case2.py
# -*- coding: utf-8 -*-

"""
[不对称、多个策略分叉，适合检查「策略行为是否符合预期」]
Tiny Case 2（不对称资源 + 不对称价格）
用于同时调用：
- MultiFunctionOptimizer（multi_all_node_KV_optimizer.py）
- SingleFunctionOptimizer（single_all_node_KV_optimizer.py）
- ShortestPathOptimizer（shortest_path_all_node_KV_optimizer.py）

目标：
1. 节点显存容量、显存价格不一样；
2. 模块的 KV/权重不一样；
3. 带宽足够大，带宽价格为 0，避免干扰；
4. 让不同策略有可能给出不同的部署方案；
5. 方便手算验证 max_users / cost / profit 是否一致。
"""

from multi_all_node_KV_optimizer import MultiFunctionOptimizer
from single_all_node_KV_optimizer import SingleFunctionOptimizer
from shortest_path_all_node_KV_optimizer import ShortestPathOptimizer


def build_tiny_test_case2():
    """
    构造不对称的 test_case2：

    节点：
      - node0: [compute=100, mem=10], mem_cost=1.0  （容量中等，单价中等）
      - node1: [compute=100, mem=6],  mem_cost=0.5  （容量小，单价便宜）
      - node2: [compute=100, mem=20], mem_cost=2.0  （容量大，单价贵）

    模块（顺序：m0 -> m1）：
      - m0: compute=1.0, KV=0.5 GB/user, weight=4 GB   （权重重、KV少）
      - m1: compute=1.0, KV=1.0 GB/user, weight=1 GB   （权重轻、KV多）

    其他：
      - 带宽矩阵：节点之间 1000 MB/s（足够大，不形成瓶颈）
      - data_sizes = [1.0] （边界数据量）
      - bandwidth_cost = 0.0 （不计链路成本）
      - profit_per_user = 10.0
      - GPU 成本为 0，只用显存成本作部署成本
    """
    node_count = 3
    module_count = 2

    # 节点算力+显存：[compute_cap, mem_cap_GB]
    computation_capacity = [
        [100.0, 10.0],  # node0
        [100.0, 6.0],   # node1
        [100.0, 20.0],  # node2
    ]

    # 模块 per-user 需求：[compute_demand, kv_mem_demand_GB]
    resource_demands = [
        [1.0, 0.5],   # module 0（m0）：compute=1, KV=0.5
        [1.0, 1.0],   # module 1（m1）：compute=1, KV=1.0
    ]

    # 模块权重显存（一次加载，多链复用）
    weight_memory_demands = [
        4.0,  # m0 权重 4 GB（重）
        1.0,  # m1 权重 1 GB（轻）
    ]

    # 边界数据量（MB/s per user）
    data_sizes = [1.0]  # m0 -> m1

    # 带宽矩阵（MB/s）
    # 对角为 0，自身不走链路；其他都是 1000 MB/s，足够大
    bandwidth_matrix = [
        [0.0, 1000.0, 1000.0],
        [1000.0, 0.0, 1000.0],
        [1000.0, 1000.0, 0.0],
    ]

    # 距离矩阵（hop，简单起见，全 1）
    distance_matrix = [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]

    # 节点成本：[gpu_cost, memory_cost]
    node_costs = [
        [0.0, 1.0],  # node0：显存 1$/GB
        [0.0, 0.5],  # node1：显存 0.5$/GB（便宜）
        [0.0, 2.0],  # node2：显存 2$/GB（贵）
    ]

    test_case = {
        "test_data_id": 2,  # 小 ID，会打印部分调试日志
        "node_count": node_count,
        "module_count": module_count,
        "computation_capacity": computation_capacity,
        "resource_demands": resource_demands,
        "weight_memory_demands": weight_memory_demands,
        "data_sizes": data_sizes,
        "bandwidth_matrix": bandwidth_matrix,
        "distance_matrix": distance_matrix,
        "node_costs": node_costs,
        # 下面是全局成本参数（multi/single 作为默认值也会用到）
        "gpu_cost": 0.0,
        "memory_cost": 1.0,       # 这里给个默认值（但我们已经在 node_costs 中指定了）
        "bandwidth_cost": 0.0,    # 不计链路成本
        "profit_per_user": 10.0,  # 每个用户收益 $10
    }

    return test_case


def main():
    test_case2 = build_tiny_test_case2()

    print("========== Tiny Case 2: MultiFunctionOptimizer ==========")
    multi_opt = MultiFunctionOptimizer(test_case2)
    min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan = multi_opt.optimize_for_profit()
    print("min_cost_plan     :", min_cost_plan)
    print("max_profit_plan   :", max_profit_plan)
    print("min_profit_plan   :", min_profit_plan)
    print("max_users_plan    :", max_users_plan)

    print("\n========== Tiny Case 2: SingleFunctionOptimizer ==========")
    single_opt = SingleFunctionOptimizer(test_case2)
    min_cost_plan_s, max_profit_plan_s, min_profit_plan_s, max_users_plan_s = single_opt.optimize_for_profit()
    print("min_cost_plan     :", min_cost_plan_s)
    print("max_profit_plan   :", max_profit_plan_s)
    print("min_profit_plan   :", min_profit_plan_s)
    print("max_users_plan    :", max_users_plan_s)

    print("\n========== Tiny Case 2: ShortestPathOptimizer ==========")
    sp_opt = ShortestPathOptimizer(test_case2)
    sp_result = sp_opt.shortest_path_deployment()
    print("shortest_path_plan:", sp_result)


if __name__ == "__main__":
    main()
