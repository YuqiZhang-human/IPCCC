# tiny_test.py
# -*- coding: utf-8 -*-

"""
[对称、简单，适合 sanity check]
一个非常小的测试例，用来同时调用：
- MultiFunctionOptimizer（multi_all_node.py）
- SingleFunctionOptimizer（single_all_node_multi.py）
- ShortestPathOptimizer（shortest_path_all_node.py）

测试目标：
1. 所有参数极小且对称，方便手算。
2. 三个优化器都使用「权重复用 + KV 不复用」模型。
3. 观察多链部署结果是否与手算一致。
"""

from multi_all_node_KV_optimizer import MultiFunctionOptimizer
from single_all_node_KV_optimizer import SingleFunctionOptimizer
from shortest_path_all_node_KV_optimizer import ShortestPathOptimizer


def build_tiny_test_case():
    """
    构造一个极小且对称的 test_case：
    - 2 个节点（0、1）
    - 2 个模块（m0、m1）
    - 两个节点完全对称（算力、显存、单价都一样）
    - 两个模块完全对称（算力、KV、权重都一样）
    - 带宽足够大，不形成瓶颈，也不计成本（bandwidth_cost=0）
    - GPU 成本为 0，只看显存成本（memory_cost=1）
    - 每个用户收益为 10
    """
    node_count = 2
    module_count = 2

    # 每个节点：算力 100（随便足够大），显存 10 GB
    computation_capacity = [
        [100.0, 10.0],  # node 0
        [100.0, 10.0],  # node 1
    ]

    # 每个模块：每用户需要 1 单位算力 + 1 GB KV 显存
    # 注意：这里的第二项只表示 KV，不包含权重
    resource_demands = [
        [1.0, 1.0],  # module 0
        [1.0, 1.0],  # module 1
    ]

    # 每个模块的权重显存：2 GB（一次加载，多链复用）
    weight_memory_demands = [2.0, 2.0]

    # 相邻模块之间的数据量（MB/s per user）
    data_sizes = [1.0]  # m0 -> m1

    # 带宽矩阵：节点 0 和 1 之间 1000 MB/s，足够大
    bandwidth_matrix = [
        [0.0, 1000.0],
        [1000.0, 0.0],
    ]

    # 距离矩阵：hop 都为 1（对角 0）
    distance_matrix = [
        [0.0, 1.0],
        [1.0, 0.0],
    ]

    # 节点成本：GPU 不计价（0），显存 1$/GB（月）
    node_costs = [
        [0.0, 1.0],  # node 0
        [0.0, 1.0],  # node 1
    ]

    test_case = {
        "test_data_id": 1,                # 小 ID，会输出一些调试信息；想安静一点可以改成 99
        "node_count": node_count,
        "module_count": module_count,
        "computation_capacity": computation_capacity,
        "resource_demands": resource_demands,
        "weight_memory_demands": weight_memory_demands,
        "data_sizes": data_sizes,
        "bandwidth_matrix": bandwidth_matrix,
        "distance_matrix": distance_matrix,
        "node_costs": node_costs,
        # 全局成本参数（multi/single 会用到 gpu_cost/memory_cost 作为默认）
        "gpu_cost": 0.0,
        "memory_cost": 1.0,
        "bandwidth_cost": 0.0,           # 不考虑链路成本，简化
        "profit_per_user": 10.0,         # 每个用户收益 $10
    }

    return test_case


def main():
    test_case = build_tiny_test_case()

    print("========== MultiFunctionOptimizer ==========")
    multi_opt = MultiFunctionOptimizer(test_case)
    min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan = multi_opt.optimize_for_profit()
    print("min_cost_plan     :", min_cost_plan)
    print("max_profit_plan   :", max_profit_plan)
    print("min_profit_plan   :", min_profit_plan)
    print("max_users_plan    :", max_users_plan)

    print("\n========== SingleFunctionOptimizer ==========")
    single_opt = SingleFunctionOptimizer(test_case)
    min_cost_plan_s, max_profit_plan_s, min_profit_plan_s, max_users_plan_s = single_opt.optimize_for_profit()
    print("min_cost_plan     :", min_cost_plan_s)
    print("max_profit_plan   :", max_profit_plan_s)
    print("min_profit_plan   :", min_profit_plan_s)
    print("max_users_plan    :", max_users_plan_s)

    print("\n========== ShortestPathOptimizer ==========")
    sp_opt = ShortestPathOptimizer(test_case)
    sp_result = sp_opt.shortest_path_deployment()
    print("shortest_path_plan:", sp_result)


if __name__ == "__main__":
    main()
