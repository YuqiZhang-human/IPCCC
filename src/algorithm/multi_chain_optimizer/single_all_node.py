#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单功能部署优化器（节点独占 + 多链部署版）

特性：
- 单条链内部：每个节点最多部署 1 个模块（single-function）
- 多条链之间：某个节点一旦部署过模块，就被视为“永久占用”，后续链不能再使用该节点
- 资源建模：
    * 节点容量：算力（TFLOPs/s）、显存（GB）
    * 模块需求：每用户算力需求（TFLOPs/s per user）、显存需求（GB per user）
    * 链路容量：带宽（MB/s）
    * 边界负载：每用户数据速率（MB/s per user）
- 多链部署流程：
    * 从初始资源 + 所有节点可用开始
    * 反复：
        1. 在当前“剩余资源 + 剩余可用节点”下求一条单链的最优方案（最大利润）
        2. 计算该链的最大支持用户数 user_count
        3. 累计成本 / 利润 / 用户数等指标
        4. 从剩余资源中扣除该链的资源占用，并将相关节点标记为“不可再用”
    * 当无法再找到任何可行链（max_users == 0 或可用节点数不足）时结束

返回：
- single_func_deployment() 返回 8 元组：
    (total_cost, total_deploy_cost, total_comm_cost,
     total_profit, total_users, used_nodes_count,
     avg_modules_per_node, chain_count)
"""

import math
import copy
import json
from typing import Any, Dict, List, Optional, Tuple


class SingleFunctionOptimizer:
    """单功能部署优化器（节点独占 + 多链部署）"""

    def __init__(self, test_data: Dict[str, Any]) -> None:
        """
        Args:
            test_data: 字典，包含以下键（由 test_experiment1.py 构造）：
                - node_count: 节点数量 n
                - module_count: 模块数量 m
                - computation_capacity: n×2，[[compute_TFLOPs_per_s, memory_GB], ...]
                - resource_demands: m×2，[[compute_TFLOPs_per_s_per_user, memory_GB_per_user], ...]
                - data_sizes: 长度 m-1，每个边界的单用户数据速率 (MB/s)
                - bandwidth_matrix: n×n，链路带宽 (MB/s)
                - gpu_cost: GPU 成本系数（当前版本一般为 0）
                - memory_cost: 显存成本系数（$/GB/month）
                - bandwidth_cost: 带宽成本系数（$/ (MB/s·month)）
                - profit_per_user: 单用户月收益（$/user/month）
                - node_costs: n×2，每个节点 [gpu_cost, memory_cost]（可选）
                - distance_matrix: n×n，节点间距离（跳数）
        """
        # 基本 ID
        self.test_data_id = test_data.get("test_data_id", 0)

        # 基础规模
        self.node_count = int(test_data["node_count"])
        self.module_count = int(test_data["module_count"])

        # 解析数组型数据
        self.computation_capacity = self._parse_array(test_data["computation_capacity"])
        self.resource_demands = self._parse_array(test_data["resource_demands"])
        self.data_sizes = self._parse_array(test_data["data_sizes"])
        self.bandwidth_matrix = self._parse_array(test_data["bandwidth_matrix"])

        # 成本与收益
        self.gpu_cost = float(test_data.get("gpu_cost", 0.0))
        self.memory_cost = float(test_data.get("memory_cost", 0.0))
        self.bandwidth_cost = float(test_data.get("bandwidth_cost", 0.0))
        self.profit_per_user = float(test_data.get("profit_per_user", 0.0))

        # 节点特定成本与距离矩阵
        node_costs = test_data.get("node_costs")
        distance_matrix = test_data.get("distance_matrix")

        self.node_costs = self._parse_array(node_costs) if node_costs is not None else None
        self.distance_matrix = self._parse_array(distance_matrix) if distance_matrix is not None else None

        # 如果未提供 per-node 成本，则使用全局成本
        if self.node_costs is None:
            self.node_costs = [[self.gpu_cost, self.memory_cost] for _ in range(self.node_count)]

        # 如果未提供距离矩阵，使用默认距离 1（对角线为 0）
        if self.distance_matrix is None:
            self.distance_matrix = [
                [0 if i == j else 1 for j in range(self.node_count)]
                for i in range(self.node_count)
            ]

        # 简单 sanity check
        if self.node_count < self.module_count:
            print(
                f"警告: 节点数量({self.node_count})少于模块数量({self.module_count})，"
                "single-function 模型下可能无法找到可行解（每条链需要至少 m 个节点）"
            )

        # 初始资源（备份）
        self.initial_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.initial_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 剩余资源（循环部署中动态更新）
        self.remaining_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 节点可用状态：节点独占模型的关键
        # True 表示该节点还从未部署过任何链上的模块，可以使用
        # False 表示该节点已经在某条链中部署过模块，后续链不能再用
        self.node_available: List[bool] = [True] * self.node_count

        # 可选：记录所有单链方案
        self.all_solutions: List[Tuple] = []

    # ----------------------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------------------
    def _parse_array(self, value):
        """将可能为 JSON 字符串的数组转换为 Python 列表"""
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return json.loads(value)
            except Exception:
                try:
                    return eval(value)
                except Exception:
                    raise ValueError(f"无法解析数组字段: {value}")
        return value

    def get_node_capacity(self, node: int) -> Tuple[float, float]:
        """获取节点剩余的 [算力, 显存] 容量"""
        if node < 0 or node >= self.node_count:
            return 0.0, 0.0
        # 使用剩余资源，而不是初始资源
        return (
            float(self.remaining_computation_capacity[node][0]),
            float(self.remaining_computation_capacity[node][1]),
        )

    def get_module_demand(self, module: int) -> Tuple[float, float]:
        """获取单个模块的 [算力需求/用户, 显存需求/用户]"""
        if module < 0 or module >= self.module_count:
            return 0.0, 0.0
        return (
            float(self.resource_demands[module][0]),
            float(self.resource_demands[module][1]),
        )

    def get_link_bandwidth(self, from_node: int, to_node: int) -> float:
        """获取两个节点之间的剩余带宽 (MB/s)"""
        if (
            from_node < 0 or from_node >= self.node_count
            or to_node < 0 or to_node >= self.node_count
        ):
            return 0.0
        return float(self.remaining_bandwidth_matrix[from_node][to_node])

    def get_link_distance(self, from_node: int, to_node: int) -> float:
        """获取两个节点之间的距离（跳数，用于通信成本）"""
        if (
            from_node < 0 or from_node >= self.node_count
            or to_node < 0 or to_node >= self.node_count
        ):
            return 1.0
        return float(self.distance_matrix[from_node][to_node])

    def get_data_size(self, boundary_index: int) -> float:
        """获取相邻两个模块之间的单用户数据速率 (MB/s)"""
        if boundary_index < 0 or boundary_index >= len(self.data_sizes):
            return 0.0
        return float(self.data_sizes[boundary_index])

    # ----------------------------------------------------------------------
    # 单条链的资源极限与成本计算
    # ----------------------------------------------------------------------
    def calculate_max_users_for_deployment(self, deployment: List[int]) -> int:
        """
        计算在当前剩余资源下，给定部署方案能支持的最大用户数。

        - 节点限制：算力、显存
        - 链路限制：带宽
        """
        # 统计每个节点的资源使用（单个用户）
        node_compute_usage = [0.0] * self.node_count
        node_memory_usage = [0.0] * self.node_count

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        node_limits: List[int] = []

        # 1. 节点算力 & 显存限制
        for node in range(self.node_count):
            comp_cap, mem_cap = self.get_node_capacity(node)
            comp_use = node_compute_usage[node]
            mem_use = node_memory_usage[node]

            if comp_use <= 0 and mem_use <= 0:
                # 该节点未使用，不构成限制
                continue

            # 计算算力可支持的最大用户数
            if comp_use > 0 and comp_cap > 0:
                users_by_compute = comp_cap / comp_use
            else:
                users_by_compute = float("inf")

            # 计算显存可支持的最大用户数
            if mem_use > 0 and mem_cap > 0:
                users_by_memory = mem_cap / mem_use
            else:
                users_by_memory = float("inf")

            node_limit = min(users_by_compute, users_by_memory)
            if node_limit <= 0:
                return 0

            node_limits.append(int(math.floor(node_limit)))

        # 2. 链路带宽限制
        link_limits: List[int] = []
        for boundary_idx in range(self.module_count - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]

            if from_node == to_node:
                # 同一节点，无跨节点通信
                continue

            bandwidth = self.get_link_bandwidth(from_node, to_node)
            if bandwidth <= 0:
                return 0

            data_size = self.get_data_size(boundary_idx)  # MB/s per user
            if data_size <= 0:
                # 没有实际通信负载，不构成限制
                continue

            users_by_bandwidth = bandwidth / data_size
            if users_by_bandwidth <= 0:
                return 0

            link_limits.append(int(math.floor(users_by_bandwidth)))

        all_limits = node_limits + link_limits
        if not all_limits:
            # 理论上不会发生（至少会有节点限制）
            return 0

        return max(0, min(all_limits))

    def calculate_costs_for_deployment(
        self, deployment: List[int], user_count: int
    ) -> Tuple[float, float, float, float]:
        """
        计算单条链在当前资源视角下的：
        - 总成本（cost）
        - 部署成本（compute+memory）（deploy_cost）
        - 通信成本（comm_cost）
        - 利润（profit）
        """
        if user_count <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # 1. 计算各节点上的 per-user 资源使用
        node_compute_usage = [0.0] * self.node_count
        node_memory_usage = [0.0] * self.node_count

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        # 2. 部署成本（这里主要是显存成本；GPU 成本通常为 0）
        deploy_cost = 0.0
        for node in range(self.node_count):
            comp_use = node_compute_usage[node] * user_count      # TFLOPs/s * users（逻辑单位）
            mem_use = node_memory_usage[node] * user_count        # GB * users

            gpu_cost_node, mem_cost_node = self.node_costs[node]

            # 当前版本通常 gpu_cost_node = 0，只计显存成本
            deploy_cost += comp_use * float(gpu_cost_node)
            deploy_cost += mem_use * float(mem_cost_node)

        # 3. 通信成本：对每个跨节点边界，按 data_size × distance × bandwidth_cost × user_count
        comm_cost = 0.0
        for boundary_idx in range(self.module_count - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)  # MB/s per user
            if data_size <= 0:
                continue

            distance = self.get_link_distance(from_node, to_node)
            comm_cost += data_size * distance * self.bandwidth_cost * user_count

        total_cost = deploy_cost + comm_cost
        profit = self.profit_per_user * user_count - total_cost

        return total_cost, deploy_cost, comm_cost, profit

    # ----------------------------------------------------------------------
    # 扣减资源（用于多链部署 + 节点独占）
    # ----------------------------------------------------------------------
    def apply_chain_consumption(self, deployment: List[int], user_count: int) -> None:
        """
        将一条已经部署的链对应的资源占用，从“剩余资源”中扣除。

        额外：将使用过的节点标记为 node_available=False，
        并将其剩余算力 / 显存置 0，从而实现“节点独占型”模型。
        """
        if user_count <= 0:
            return

        # 1. 节点资源扣减 + 节点禁用
        node_compute_usage = [0.0] * self.node_count
        node_memory_usage = [0.0] * self.node_count

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        for node in range(self.node_count):
            comp_use = node_compute_usage[node]
            mem_use = node_memory_usage[node]

            if comp_use <= 0 and mem_use <= 0:
                # 该节点在本条链中未使用
                continue

            # 扣减剩余资源
            comp_cap, mem_cap = self.remaining_computation_capacity[node]
            comp_cap = float(comp_cap) - comp_use * user_count
            mem_cap = float(mem_cap) - mem_use * user_count
            if comp_cap < 0.0:
                comp_cap = 0.0
            if mem_cap < 0.0:
                mem_cap = 0.0

            # 将节点资源清零 + 标记为不可再用
            self.remaining_computation_capacity[node][0] = 0.0
            self.remaining_computation_capacity[node][1] = 0.0
            self.node_available[node] = False

        # 2. 链路带宽扣减（理论上后续链不再使用这些节点，但保留带宽变化更符合物理直觉）
        for boundary_idx in range(self.module_count - 1):
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
    # 单条链搜索（在当前剩余资源 + 节点可用状态下）
    # ----------------------------------------------------------------------
    def find_best_single_chain(
        self,
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int]]]:
        """
        在当前剩余资源 + 节点可用状态下，枚举所有单条链部署方案，
        返回最优方案（以利润最大为目标）。

        返回:
            (cost, deploy_cost, comm_cost, profit,
             user_count, used_nodes_count, avg_modules_per_node, deployment)
            或 None（无可行方案）
        """
        n = self.node_count
        m = self.module_count

        if m == 0 or n == 0:
            return None

        # 节点独占模型下，如果可用节点数 < 模块数，必然无解
        available_nodes_count = sum(1 for v in self.node_available if v)
        if available_nodes_count < m:
            return None

        best_solution: Optional[
            Tuple[float, float, float, float, int, int, float, List[int]]
        ] = None

        # 回溯搜索：为每个模块选择一个“当前仍可用且在本链未用过”的节点
        used_nodes = [False] * n
        deployment = [0] * m

        def dfs(mod_idx: int):
            nonlocal best_solution

            if mod_idx == m:
                # 完整部署方案
                user_count = self.calculate_max_users_for_deployment(deployment)
                if user_count <= 0:
                    return

                cost, deploy_cost, comm_cost, profit = self.calculate_costs_for_deployment(
                    deployment, user_count
                )

                used_nodes_count = len(set(deployment))
                avg_mods_per_node = m / used_nodes_count if used_nodes_count > 0 else 0.0

                cand = (
                    cost,
                    deploy_cost,
                    comm_cost,
                    profit,
                    user_count,
                    used_nodes_count,
                    avg_mods_per_node,
                    deployment.copy(),
                )

                # 选择逻辑：利润优先，其次用户数，再次总成本
                if best_solution is None:
                    best_solution = cand
                else:
                    (
                        b_cost,
                        _b_dep,
                        _b_comm,
                        b_profit,
                        b_users,
                        _b_used_nodes,
                        _b_avg_mods,
                        _b_deploy,
                    ) = best_solution

                    if (
                        profit > b_profit
                        or (profit == b_profit and user_count > b_users)
                        or (profit == b_profit and user_count == b_users and cost < b_cost)
                    ):
                        best_solution = cand
                return

            # 为第 mod_idx 个模块选择节点
            for node in range(n):
                # 节点独占：只能选当前仍可用的节点
                if not self.node_available[node]:
                    continue
                # 单链 single-function：本链内一个节点只能放一个模块
                if used_nodes[node]:
                    continue

                used_nodes[node] = True
                deployment[mod_idx] = node
                dfs(mod_idx + 1)
                used_nodes[node] = False

        dfs(0)

        return best_solution

    # ----------------------------------------------------------------------
    # 对外主入口：循环部署多条链直到资源 / 节点耗尽
    # ----------------------------------------------------------------------
    def single_func_deployment(self):
        """
        在给定网络 / 资源约束下，循环部署多条单功能链，直到无法再部署一条可行链。

        节点独占：一旦某节点在某条链中部署过模块，该节点在后续链中无法再使用。

        返回 8 元组:
            (total_cost, total_deploy_cost, total_comm_cost,
             total_profit, total_users, used_nodes_count,
             avg_modules_per_node, chain_count)
        """
        # 每次调用都从“初始资源 + 所有节点可用”开始
        self.remaining_computation_capacity = copy.deepcopy(self.initial_computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.initial_bandwidth_matrix)
        self.node_available = [True] * self.node_count

        total_cost = 0.0
        total_deploy_cost = 0.0
        total_comm_cost = 0.0
        total_profit = 0.0
        total_users = 0
        used_nodes_union = set()
        chain_count = 0

        while True:
            best = self.find_best_single_chain()
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

            # 如果最大可支持用户数为 0，则停止
            if user_count <= 0:
                break

            # 累计
            total_cost += cost
            total_deploy_cost += deploy_cost
            total_comm_cost += comm_cost
            total_profit += profit
            total_users += user_count
            used_nodes_union.update(deployment)
            chain_count += 1

            # 扣减资源 + 禁用节点
            self.apply_chain_consumption(deployment, user_count)

        if chain_count == 0:
            # 无法部署任何链，返回 None（保持调用逻辑兼容）
            return None

        used_nodes_count = len(used_nodes_union)
        avg_modules_per_node = (
            self.module_count * chain_count / used_nodes_count
            if used_nodes_count > 0
            else 0.0
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
