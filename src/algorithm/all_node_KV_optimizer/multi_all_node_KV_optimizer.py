#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多功能部署优化器（多链部署版，按模块贪心 + 权重复用 + KV 不复用）

【核心逻辑对齐说明】

1. 资源模型（真实版 Serving 简化）：
   - resource_demands[module] = [compute_demand_per_user, kv_memory_per_user_GB]
     * 第二个维度只表示 KV 显存（随用户数线性增长）
   - weight_memory_demands[module] = 权重显存 (GB)，一次加载，多链复用
   - 每个节点 j 的显存约束：
       mem_used_j = sum_{m 已在 j 上加载} weight_mem[m]
                    + sum_{所有已部署链 k} u_k * sum_{m 属于链 k 且部署在 j 的模块} kv_mem_per_user[m]
       <= mem_cap_j
     在实现里，已部署链的 (权重 + KV) 都体现在剩余显存容量中，
     对于新链只再考虑「本链新增权重 + 本链 KV」。

2. 单条链部署（单策略）：
   - 按模块顺序 m_0, m_1, ..., m_{K-1} 贪心：
       对当前模块 m_i：
           遍历所有节点 j，评估：
             - 在当前剩余资源 + 前面模块已选节点的前提下，
               将 m_i 放到 j 时，这个模块的「局部指标」：
                 * 能支持的本链局部用户数 users_local
                 * 本模块引入的局部成本 cost_local
                 * 局部利润 profit_local = users_local * profit_per_user - cost_local
           然后：
             - min_cost     : 选择 cost_local 最小的 j
             - max_profit   : 选择 profit_local 最大的 j
             - min_profit   : 选择 profit_local 最小的 j （最差情况）
             - max_users    : 选择 users_local 最大的 j
       如果某一层没有任何可行节点 → 当前策略下该链无法再部署。

   - 局部指标只考虑「当前模块的增量 + 与上一模块的链路」，
     符合你说的“当前子模块产生的成本 / 利润 / 用户量”的本地视角；
     整条链完成后再用全链约束重新算一次真实 max_users、cost、profit。

3. 多链部署（单策略）：
   - 对于某个 objective（min_cost / max_profit / min_profit / max_users）：
       1) 将网络资源恢复到初始状态：
            - remaining_computation_capacity
            - remaining_bandwidth_matrix
            - modules_loaded_per_node（哪些模块的权重已经在节点上）
       2) 循环：
            a. 用当前策略运行一次“单链贪心部署”
            b. 如果该链最终 max_users <= 0 或不可行 → 停止
            c. 否则：
                - 把该链的成本 / 利润 / 用户数 加到全局累计
                - 从剩余资源中扣除该链的算力 / KV / 权重 / 带宽占用
                - 更新 modules_loaded_per_node 中的已加载权重
       3) 得到该策略下“把网络吃干榨净”后的：
            (total_cost, total_deploy_cost, total_comm_cost,
             total_profit, total_users, used_nodes_count,
             avg_modules_per_node, chain_count)

4. 四种策略相互独立：
   - optimize_for_profit() 内部顺序执行：
       - min_cost  策略的多链部署（从初始资源开始）
       - max_profit 策略的多链部署（资源重置）
       - min_profit 策略的多链部署（资源重置）
       - max_users  策略的多链部署（资源重置）
   - 每种策略返回 8 元组，与你原程序接口保持一致：
       (total_cost, total_deploy_cost, total_comm_cost,
        total_profit, total_users, used_nodes_count,
        avg_modules_per_node, chain_count)
"""

import json
import math
import copy
from typing import Any, Dict, List, Optional, Tuple


class MultiFunctionOptimizer:
    """多功能部署优化器（多链部署版，按模块贪心）"""

    def __init__(self, test_data: Dict[str, Any]) -> None:
        """
        Args:
            test_data (dict): 测试数据字典，包含如下字段（由上游脚本构造）：
                - node_count: 节点数量 n
                - module_count: 模块数量 m
                - computation_capacity: n×2，[[compute_TFLOPs/s, memory_GB], ...]
                - resource_demands: m×2，[[compute_TFLOPs/s per user, kv_memory_GB per user], ...]
                - weight_memory_demands: 长度为 m，模块权重显存 (GB)，可选，没有则视为 0
                - data_sizes: 长度 m-1，每个边界的单用户数据速率 (MB/s)
                - bandwidth_matrix: n×n，链路带宽 (MB/s)
                - gpu_cost: GPU 成本系数（一般建议设为 0，让算力只作为约束）
                - memory_cost: 显存成本系数（$/GB/月）
                - bandwidth_cost: 带宽成本系数（$/ (MB/s·月)）
                - profit_per_user: 单用户月收益（$/user/月）
                - node_costs: 每个节点的 [gpu_cost, memory_cost]（可选）
                - distance_matrix: n×n 节点间距离（跳数）
                - test_data_id: 用于调试打印的标识（可选）
        """
        # test_data_id 便于调试时筛选输出
        self.test_data_id = int(test_data.get("test_data_id", 0))

        # 基本规模
        self.node_count = int(test_data["node_count"])
        self.module_count = int(test_data["module_count"])

        # 解析数组型配置
        self.computation_capacity = self._parse_array(test_data["computation_capacity"])
        self.resource_demands = self._parse_array(test_data["resource_demands"])
        self.data_sizes = self._parse_array(test_data["data_sizes"])
        self.bandwidth_matrix = self._parse_array(test_data["bandwidth_matrix"])

        # 成本与收益
        self.gpu_cost = float(test_data.get("gpu_cost", 0.0))
        self.memory_cost = float(test_data.get("memory_cost", 0.0))
        self.bandwidth_cost = float(test_data.get("bandwidth_cost", 0.0))
        self.profit_per_user = float(test_data.get("profit_per_user", 0.0))

        # 节点成本与距离矩阵
        node_costs = test_data.get("node_costs")
        distance_matrix = test_data.get("distance_matrix")

        self.node_costs = self._parse_array(node_costs) if node_costs is not None else None
        self.distance_matrix = self._parse_array(distance_matrix) if distance_matrix is not None else None

        # 如果未提供 per-node 成本，则统一使用全局成本
        if self.node_costs is None:
            self.node_costs = [[self.gpu_cost, self.memory_cost] for _ in range(self.node_count)]

        # 如果未提供距离矩阵，则默认距离为 1（对角线为 0）
        if self.distance_matrix is None:
            self.distance_matrix = [
                [0 if i == j else 1 for j in range(self.node_count)]
                for i in range(self.node_count)
            ]

        # 权重显存需求（静态，一次加载，多链复用），可选字段
        raw_weight_mem = test_data.get("weight_memory_demands")
        if raw_weight_mem is not None:
            parsed = self._parse_array(raw_weight_mem)
            if len(parsed) != self.module_count:
                raise ValueError("weight_memory_demands 长度必须等于 module_count")
            self.weight_memory_demands = [float(x) for x in parsed]
        else:
            # 如果没有提供，则默认所有模块权重显存为 0（兼容老配置）
            self.weight_memory_demands = [0.0 for _ in range(self.module_count)]

        # 初始资源快照（算力 & 显存 & 带宽）
        self.initial_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.initial_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 当前剩余资源（会在多链部署过程中不断扣减）
        self.remaining_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 当前策略下：每个节点上已经加载过哪些模块的权重（多链之间权重复用）
        self.modules_loaded_per_node: List[set] = [set() for _ in range(self.node_count)]

    # ----------------------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------------------
    def _parse_array(self, data):
        """解析数据，如果是 JSON 字符串则转换为列表"""
        if data is None:
            return None

        if isinstance(data, str):
            text = data.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                try:
                    return eval(text)
                except Exception:
                    raise ValueError(f"无法解析数组字段: {data}")
        return data

    def get_node_capacity(self, node: int) -> Tuple[float, float]:
        """获取节点当前剩余的 [算力, 显存] 容量"""
        if node < 0 or node >= self.node_count:
            return 0.0, 0.0
        return (
            float(self.remaining_computation_capacity[node][0]),
            float(self.remaining_computation_capacity[node][1]),
        )

    def get_module_demand(self, module: int) -> Tuple[float, float]:
        """
        获取模块单用户的 [算力需求, KV 显存需求(GB/user)]
        第二个维度只表示 KV 显存，不包含权重显存。
        """
        if module < 0 or module >= self.module_count:
            return 0.0, 0.0
        return (
            float(self.resource_demands[module][0]),
            float(self.resource_demands[module][1]),
        )

    def get_module_weight_memory(self, module: int) -> float:
        """获取模块权重显存 (GB)"""
        if module < 0 or module >= self.module_count:
            return 0.0
        return float(self.weight_memory_demands[module])

    def get_link_bandwidth(self, from_node: int, to_node: int) -> float:
        """获取两个节点之间当前剩余的带宽 (MB/s)"""
        if (
            from_node < 0
            or from_node >= self.node_count
            or to_node < 0
            or to_node >= self.node_count
        ):
            return 0.0
        return float(self.remaining_bandwidth_matrix[from_node][to_node])

    def get_link_distance(self, from_node: int, to_node: int) -> float:
        """获取两个节点之间的距离（跳数，用于通信成本）"""
        if (
            from_node < 0
            or from_node >= self.node_count
            or to_node < 0
            or to_node >= self.node_count
        ):
            return 1.0
        return float(self.distance_matrix[from_node][to_node])

    def get_data_size(self, boundary_index: int) -> float:
        """获取相邻两个模块之间的单用户数据速率 (MB/s)"""
        if boundary_index < 0 or boundary_index >= len(self.data_sizes):
            return 0.0
        return float(self.data_sizes[boundary_index])

    # ----------------------------------------------------------------------
    # 全链的资源极限与成本计算（使用权重复用 + KV 不复用模型）
    # ----------------------------------------------------------------------
    def calculate_max_users_for_deployment(self, deployment: List[int]) -> int:
        """
        在当前剩余资源 + 已加载权重 self.modules_loaded_per_node 的前提下，
        计算给定多功能部署方案能支持的最大用户数。

        - 节点限制：算力 / 显存(权重 + KV)
        - 链路限制：带宽
        """
        n = self.node_count
        m = self.module_count

        if len(deployment) != m:
            return 0

        # 1. 逐节点统计单用户资源需求 & 本链新增权重
        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n  # 本链新增的权重显存（只对尚未在该节点加载过的模块计一次）

        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand

            weight_mem = self.get_module_weight_memory(module_idx)
            # 若该模块权重尚未在该节点加载过，则本链需要新增这份权重显存
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem

        limits: List[int] = []

        # 节点算力 & 显存限制
        for node in range(n):
            comp_cap, mem_cap = self.get_node_capacity(node)
            comp_use = node_compute_per_user[node]
            kv_use = node_kv_per_user[node]
            new_weight = node_new_weight[node]

            # 完全未使用该节点
            if comp_use <= 0 and kv_use <= 0 and new_weight <= 0:
                continue

            # 如果连新增权重都放不下，则不可行
            if new_weight > mem_cap + 1e-9:
                return 0

            comp_limit = float("inf")
            mem_limit = float("inf")

            if comp_use > 0:
                if comp_cap <= 0:
                    return 0
                comp_limit = comp_cap / comp_use

            if kv_use > 0:
                avail_mem = mem_cap - new_weight
                if avail_mem <= 0:
                    return 0
                mem_limit = avail_mem / kv_use

            node_limit = min(comp_limit, mem_limit)
            if node_limit <= 0:
                return 0
            limits.append(int(math.floor(node_limit)))

        # 2. 链路带宽限制
        for boundary_idx in range(m - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]

            if from_node == to_node:
                continue

            bandwidth = self.get_link_bandwidth(from_node, to_node)
            if bandwidth <= 0:
                return 0

            data_size = self.get_data_size(boundary_idx)  # MB/s per user
            if data_size <= 0:
                continue

            users_by_bandwidth = bandwidth / data_size
            if users_by_bandwidth <= 0:
                return 0

            limits.append(int(math.floor(users_by_bandwidth)))

        if not limits:
            return 0

        return max(0, min(limits))

    def calculate_costs_for_deployment(
        self, deployment: List[int], user_count: int
    ) -> Tuple[float, float, float, float]:
        """
        计算单条链（在当前剩余资源视角下）的：
        - total_cost         总成本
        - deploy_cost        部署成本（显存 + 可选算力）
        - communication_cost 通信成本
        - profit             利润
        """
        n = self.node_count
        m = self.module_count

        if user_count <= 0 or len(deployment) != m:
            return 0.0, 0.0, 0.0, 0.0

        # 1. 节点资源使用（per user） + 本链新增权重
        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand

            weight_mem = self.get_module_weight_memory(module_idx)
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem

        # 2. 部署成本（GPU + 显存）
        deploy_cost = 0.0
        for node in range(n):
            comp_use = node_compute_per_user[node] * user_count
            kv_use = node_kv_per_user[node] * user_count
            weight_use = node_new_weight[node]

            gpu_cost_node, mem_cost_node = self.node_costs[node]

            # 注意：如果你希望算力不计成本，可以在上游把 gpu_cost_node 设为 0
            deploy_cost += comp_use * float(gpu_cost_node)
            deploy_cost += (kv_use + weight_use) * float(mem_cost_node)

        # 3. 通信成本
        comm_cost = 0.0
        for boundary_idx in range(m - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            distance = self.get_link_distance(from_node, to_node)
            comm_cost += data_size * distance * self.bandwidth_cost * user_count

        total_cost = deploy_cost + comm_cost
        profit = self.profit_per_user * user_count - total_cost

        return total_cost, deploy_cost, comm_cost, profit

    # ----------------------------------------------------------------------
    # 多链部署时：扣减资源 & 更新权重加载状态
    # ----------------------------------------------------------------------
    def apply_chain_consumption(self, deployment: List[int], user_count: int) -> None:
        """
        将某条已经部署的链，对应的资源占用，从“剩余资源”中扣除，
        并在 modules_loaded_per_node 中记录权重已加载。
        """
        if user_count <= 0:
            return

        n = self.node_count
        m = self.module_count

        # 1. 节点资源扣减（算力 + KV + 新权重）
        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand

            weight_mem = self.get_module_weight_memory(module_idx)
            # 仅对本次首次在该节点加载的模块增加权重显存
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem
                self.modules_loaded_per_node[node].add(module_idx)

        for node in range(n):
            comp_cap, mem_cap = self.remaining_computation_capacity[node]
            comp_cap = float(comp_cap) - node_compute_per_user[node] * user_count
            mem_cap = (
                float(mem_cap)
                - node_kv_per_user[node] * user_count
                - node_new_weight[node]
            )

            if comp_cap < 0.0:
                comp_cap = 0.0
            if mem_cap < 0.0:
                mem_cap = 0.0

            self.remaining_computation_capacity[node][0] = comp_cap
            self.remaining_computation_capacity[node][1] = mem_cap

        # 2. 链路带宽扣减（按 KV 流量估计）
        for boundary_idx in range(m - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            consume = data_size * user_count
            bw_ft = float(self.remaining_bandwidth_matrix[from_node][to_node]) - consume
            bw_tf = float(self.remaining_bandwidth_matrix[to_node][from_node]) - consume
            if bw_ft < 0.0:
                bw_ft = 0.0
            if bw_tf < 0.0:
                bw_tf = 0.0
            self.remaining_bandwidth_matrix[from_node][to_node] = bw_ft
            self.remaining_bandwidth_matrix[to_node][from_node] = bw_tf

    # ----------------------------------------------------------------------
    # 单条链：按模块贪心部署（当前策略 & 当前剩余资源）
    # ----------------------------------------------------------------------
    def _deploy_single_chain_greedy(
        self, objective: str
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int]]]:
        """
        在当前剩余资源 + 当前策略已加载权重状态下，
        使用「按模块顺序的局部贪心」部署一条新链。

        objective:
            - "min_cost"
            - "max_profit"
            - "min_profit"
            - "max_users"

        返回:
            (cost, deploy_cost, comm_cost, profit,
             user_count, used_nodes_count, avg_modules_per_node, deployment)
            或 None（本策略在当前资源下无法再部署新链）
        """
        n = self.node_count
        m = self.module_count

        if n <= 0 or m <= 0:
            return None

        deployment = [-1] * m

        # 逐模块贪心
        for module_idx in range(m):
            best_candidate = None  # (score_tuple, node, users_local, cost_local, profit_local)

            for node in range(n):
                # 计算当前模块放到 node 上的局部可行性和局部指标
                comp_demand, kv_demand = self.get_module_demand(module_idx)
                weight_mem = self.get_module_weight_memory(module_idx)
                comp_cap, mem_cap = self.get_node_capacity(node)

                if comp_demand > 0 and comp_cap <= 0:
                    continue

                # 判断权重是否为新增
                is_new_weight = (module_idx not in self.modules_loaded_per_node[node])
                static_weight_add = weight_mem if is_new_weight else 0.0

                # 显存：至少要放下新增权重
                if static_weight_add > mem_cap + 1e-9:
                    continue

                # 基于当前模块本身的节点限制
                comp_limit = float("inf")
                mem_limit = float("inf")

                if comp_demand > 0:
                    comp_limit = comp_cap / comp_demand

                if kv_demand > 0:
                    avail_mem = mem_cap - static_weight_add
                    if avail_mem <= 0:
                        continue
                    mem_limit = avail_mem / kv_demand

                # 链路限制：考虑与上一个模块之间的带宽
                link_limit = float("inf")
                if module_idx > 0:
                    prev_node = deployment[module_idx - 1]
                    if prev_node == -1:
                        # 理论上不会出现
                        prev_node = node
                    if prev_node != node:
                        bandwidth = self.get_link_bandwidth(prev_node, node)
                        if bandwidth <= 0:
                            continue
                        data_size = self.get_data_size(module_idx - 1)
                        if data_size > 0:
                            link_limit = bandwidth / data_size
                        else:
                            link_limit = float("inf")

                # 计算局部可支持用户数（只看当前模块 + 与前一模块的链路）
                limits = []
                if comp_limit < float("inf"):
                    limits.append(comp_limit)
                if mem_limit < float("inf"):
                    limits.append(mem_limit)
                if link_limit < float("inf"):
                    limits.append(link_limit)

                if not limits:
                    # 对该模块来说，节点不形成约束（comp=0 & kv=0 & data_size=0），
                    # 但为了安全起见，认为至少要能支持1个用户
                    users_local = 1
                else:
                    users_local = int(math.floor(min(limits)))
                    if users_local <= 0:
                        continue

                # 计算局部成本和利润（仅考虑本模块的新增占用 + 与上一模块的通信）
                gpu_cost_node, mem_cost_node = self.node_costs[node]

                comp_use_local = comp_demand * users_local
                kv_use_local = kv_demand * users_local

                deploy_cost_local = comp_use_local * float(gpu_cost_node)
                deploy_cost_local += (kv_use_local + static_weight_add) * float(mem_cost_node)

                comm_cost_local = 0.0
                if module_idx > 0:
                    prev_node = deployment[module_idx - 1]
                    if prev_node != node:
                        data_size = self.get_data_size(module_idx - 1)
                        if data_size > 0:
                            distance = self.get_link_distance(prev_node, node)
                            comm_cost_local = (
                                data_size * distance * self.bandwidth_cost * users_local
                            )

                total_cost_local = deploy_cost_local + comm_cost_local
                profit_local = self.profit_per_user * users_local - total_cost_local

                # 根据 objective 选择最优节点
                # score_tuple 统一写清楚，方便阅读：
                #   对min_cost:   (total_cost_local, -profit_local, -users_local)
                #   对max_profit: (-profit_local, -users_local, total_cost_local)
                #   对min_profit: (profit_local, total_cost_local, users_local)
                #   对max_users:  (-users_local, total_cost_local, -profit_local)
                if objective == "min_cost":
                    score = (
                        total_cost_local,
                        -profit_local,
                        -users_local,
                    )
                elif objective == "max_profit":
                    score = (
                        -profit_local,
                        -users_local,
                        total_cost_local,
                    )
                elif objective == "min_profit":
                    score = (
                        profit_local,
                        total_cost_local,
                        users_local,
                    )
                elif objective == "max_users":
                    score = (
                        -users_local,
                        total_cost_local,
                        -profit_local,
                    )
                else:
                    # 不支持的目标，直接跳过
                    continue

                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (
                        score,
                        node,
                        users_local,
                        total_cost_local,
                        profit_local,
                    )

            if best_candidate is None:
                # 当前模块找不到任何可行节点 → 在当前资源下无法再部署一条新链
                if self.test_data_id <= 5:
                    print(
                        f"[test_id={self.test_data_id}] objective={objective}, "
                        f"在部署模块 {module_idx} 时无可行节点，停止单链部署"
                    )
                return None

            _, chosen_node, users_local, local_cost, local_profit = best_candidate
            deployment[module_idx] = chosen_node

            if self.test_data_id <= 3:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, "
                    f"模块 {module_idx} 选择节点 {chosen_node} "
                    f"(局部 users={users_local}, local_cost={local_cost:.4f}, local_profit={local_profit:.4f})"
                )

        # 所有模块都贪心选完后，用完整的资源模型再算一次这条链的 max_users / 成本 / 利润
        max_users = self.calculate_max_users_for_deployment(deployment)
        if max_users <= 0:
            if self.test_data_id <= 5:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, "
                    f"完整链部署后 max_users={max_users}，视为不可行"
                )
            return None

        total_cost, deploy_cost, comm_cost, profit = self.calculate_costs_for_deployment(
            deployment, max_users
        )

        used_nodes_count = len(set(deployment))
        avg_mods_per_node = (
            self.module_count / used_nodes_count if used_nodes_count > 0 else 0.0
        )

        if self.test_data_id <= 3:
            print(
                f"[test_id={self.test_data_id}] objective={objective}, "
                f"完整链：users={max_users}, total_cost={total_cost:.4f}, profit={profit:.4f}, "
                f"used_nodes={used_nodes_count}, avg_mods_per_node={avg_mods_per_node:.3f}"
            )

        return (
            total_cost,
            deploy_cost,
            comm_cost,
            profit,
            max_users,
            used_nodes_count,
            avg_mods_per_node,
            deployment.copy(),
        )

    def find_best_single_chain_for_objective(
        self, objective: str
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int]]]:
        """
        保留原函数名以兼容旧调用逻辑，
        现在内部直接调用按模块贪心的单链部署。
        """
        return self._deploy_single_chain_greedy(objective)

    # ----------------------------------------------------------------------
    # 多链循环：在指定 objective 下吃光网络资源
    # ----------------------------------------------------------------------
    def deploy_until_exhaustion(self, objective: str):
        """
        在指定优化目标下，从初始资源出发，循环部署多条链直到无法再部署任何一条可行链。

        返回:
            (total_cost, total_deploy_cost, total_comm_cost,
             total_profit, total_users, used_nodes_count,
             avg_modules_per_node, chain_count)
        或 None（完全无法部署）
        """
        # 重置剩余资源为初始状态，并清空已加载权重
        self.remaining_computation_capacity = copy.deepcopy(self.initial_computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.initial_bandwidth_matrix)
        self.modules_loaded_per_node = [set() for _ in range(self.node_count)]

        total_cost = 0.0
        total_deploy_cost = 0.0
        total_comm_cost = 0.0
        total_profit = 0.0
        total_users = 0
        used_nodes_union = set()
        chain_count = 0

        if self.test_data_id <= 3:
            print(f"\n[test_id={self.test_data_id}] === 开始 objective={objective} 的多链部署 ===")

        while True:
            best = self.find_best_single_chain_for_objective(objective)
            if best is None:
                break

            (
                cost,
                deploy_cost,
                comm_cost,
                profit,
                user_count,
                used_nodes_count,
                avg_mods_per_node,
                deployment,
            ) = best

            # 停止条件：没有任何可行的用户（max_users = 0）时停止
            if user_count <= 0:
                break

            # 累计这一条链
            total_cost += cost
            total_deploy_cost += deploy_cost
            total_comm_cost += comm_cost
            total_profit += profit
            total_users += user_count
            used_nodes_union.update(deployment)
            chain_count += 1

            if self.test_data_id <= 3:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, "
                    f"部署第 {chain_count} 条链：users={user_count}, cost={cost:.4f}, profit={profit:.4f}"
                )

            # 从剩余资源中扣除本链消耗，并更新权重加载状态
            self.apply_chain_consumption(deployment, user_count)

        if chain_count == 0:
            if self.test_data_id <= 3:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, 完全无法部署任何链"
                )
            return None

        used_nodes_count = len(used_nodes_union)
        avg_modules_per_node = (
            self.module_count * chain_count / used_nodes_count
            if used_nodes_count > 0
            else 0.0
        )

        if self.test_data_id <= 3:
            print(
                f"[test_id={self.test_data_id}] === objective={objective} 多链部署结束："
                f"chain_count={chain_count}, total_users={total_users}, "
                f"total_profit={total_profit:.4f}, total_cost={total_cost:.4f} ==="
            )

        return (
            total_cost,
            total_deploy_cost,
            total_comm_cost,
            total_profit,
            total_users,
            used_nodes_count,
            avg_modules_per_node,
            chain_count,
        )

    # ----------------------------------------------------------------------
    # 对外主接口：返回 4 种策略吃干榨净后的全局效果
    # ----------------------------------------------------------------------
    def optimize_for_profit(self):
        """
        对同一网络 / 配置，在以下四种目标下分别
        「从初始资源出发，循环部署直到资源耗尽」：

        1. min_cost    : 总成本最小
        2. max_profit  : 总利润最大
        3. min_profit  : 总利润最小（最差情况）
        4. max_users   : 总用户量最大

        返回:
            (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)

        其中每个 plan 是 8 元组：
            (total_cost, total_deploy_cost, total_comm_cost,
             total_profit, total_users, used_nodes_count,
             avg_modules_per_node, chain_count)
        """
        min_cost_plan = self.deploy_until_exhaustion("min_cost")
        max_profit_plan = self.deploy_until_exhaustion("max_profit")
        min_profit_plan = self.deploy_until_exhaustion("min_profit")
        max_users_plan = self.deploy_until_exhaustion("max_users")

        return (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)
