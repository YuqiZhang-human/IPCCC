#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多功能部署优化器（新版，多链部署版）

- 与 SingleFunctionOptimizer 的区别：
    * 多功能：一个节点可以部署多个模块（没有 "每节点最多1个模块" 的限制）
- 单次求解：
    * 在当前剩余资源下，枚举所有「模块 → 节点」的部署方案
    * 对每个方案：
        - 基于算力 / 显存 / 带宽计算该方案可支持的最大用户数
        - 计算该方案在该用户数下的成本 / 通信成本 / 总利润
    * 根据目标策略（最小成本 / 最大利润 / 最小利润 / 最大用户数）选出最佳单条链方案
- 多链循环：
    * 对每种策略（min_cost / max_profit / min_profit / max_users）：
        - 从初始资源开始
        - 反复：
            1. 找一条单链的最优方案
            2. 如果该方案的最大用户数 == 0，则停止
            3. 否则，累计成本 / 利润 / 用户数等，并从剩余资源中扣除对应资源占用
        - 得到在该策略下将网络“吃干榨净”后的全局结果
- optimize_for_profit() 返回 4 个 8 元组：
    (total_cost, total_deploy_cost, total_comm_cost,
     total_profit, total_users, used_nodes_count,
     avg_modules_per_node, chain_count)
"""

import json
import math
import copy
from typing import Any, Dict, List, Optional, Tuple


class MultiFunctionOptimizer:
    """多功能部署优化器（多链部署版）"""

    def __init__(self, test_data: Dict[str, Any]) -> None:
        """
        Args:
            test_data (dict): 测试数据字典，包含如下字段（由 test_experiment1.py 构造）：
                - node_count: 节点数量 n
                - module_count: 模块数量 m
                - computation_capacity: n×2，[[compute_TFLOPs/s, memory_GB], ...]
                - resource_demands: m×2，[[compute_TFLOPs/s per user, memory_GB per user], ...]
                - data_sizes: 长度 m-1，每个边界的单用户数据速率 (MB/s)
                - bandwidth_matrix: n×n，链路带宽 (MB/s)
                - gpu_cost: GPU 成本系数
                - memory_cost: 显存成本系数（$/GB/月）
                - bandwidth_cost: 带宽成本系数（$/ (MB/s·月)）
                - profit_per_user: 单用户月收益（$/user/月）
                - node_costs: 每个节点的 [gpu_cost, memory_cost]（可选）
                - distance_matrix: n×n 节点间距离（跳数）
        """
        # 保存 test_data_id 便于调试
        self.test_data_id = test_data.get("test_data_id", 0)

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

        # 初始资源快照
        self.initial_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.initial_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 当前剩余资源（会在多链部署过程中不断扣减）
        self.remaining_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 可选：记录所有单链方案（一般不用）
        self.all_solutions = []

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
        """获取模块单用户的 [算力需求, 显存需求]"""
        if module < 0 or module >= self.module_count:
            return 0.0, 0.0
        return (
            float(self.resource_demands[module][0]),
            float(self.resource_demands[module][1]),
        )

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
    # 单条链的资源极限与成本计算
    # ----------------------------------------------------------------------
    def calculate_max_users_for_deployment(self, deployment: List[int]) -> int:
        """
        在当前剩余资源下，计算给定多功能部署方案能支持的最大用户数。

        - 节点限制：算力 / 显存
        - 链路限制：带宽
        """
        n = self.node_count
        m = self.module_count

        if len(deployment) != m:
            return 0

        # 1. 逐节点统计单用户资源需求
        node_compute_usage = [0.0] * n
        node_memory_usage = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        node_limits: List[int] = []

        # 节点算力 & 显存限制
        for node in range(n):
            comp_cap, mem_cap = self.get_node_capacity(node)
            comp_use = node_compute_usage[node]
            mem_use = node_memory_usage[node]

            if comp_use <= 0 and mem_use <= 0:
                # 未使用该节点
                continue

            # 基于算力的限制
            if comp_use > 0 and comp_cap > 0:
                users_by_compute = comp_cap / comp_use
            else:
                users_by_compute = float("inf")

            # 基于显存的限制
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

            link_limits.append(int(math.floor(users_by_bandwidth)))

        all_limits = node_limits + link_limits
        if not all_limits:
            return 0

        return max(0, min(all_limits))

    def calculate_costs_for_deployment(
        self, deployment: List[int], user_count: int
    ) -> Tuple[float, float, float, float]:
        """
        计算单条链（在当前剩余资源视角下）的：
        - total_cost       总成本
        - deploy_cost      部署成本（算力+显存）
        - communication_cost 通信成本
        - profit           利润
        """
        n = self.node_count
        m = self.module_count

        if user_count <= 0 or len(deployment) != m:
            return 0.0, 0.0, 0.0, 0.0

        # 1. 节点资源使用（per user）
        node_compute_usage = [0.0] * n
        node_memory_usage = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        # 2. 部署成本（GPU + 显存）
        deploy_cost = 0.0
        for node in range(n):
            comp_use = node_compute_usage[node] * user_count
            mem_use = node_memory_usage[node] * user_count

            gpu_cost_node, mem_cost_node = self.node_costs[node]
            deploy_cost += comp_use * float(gpu_cost_node)
            deploy_cost += mem_use * float(mem_cost_node)

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
    # 扣减资源（多链部署）
    # ----------------------------------------------------------------------
    def apply_chain_consumption(self, deployment: List[int], user_count: int) -> None:
        """
        将某条已经部署的链，对应的资源占用，从“剩余资源”中扣除。
        """
        if user_count <= 0:
            return

        n = self.node_count
        m = self.module_count

        # 1. 节点资源扣减
        node_compute_usage = [0.0] * n
        node_memory_usage = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        for node in range(n):
            comp_cap, mem_cap = self.remaining_computation_capacity[node]
            comp_cap = float(comp_cap) - node_compute_usage[node] * user_count
            mem_cap = float(mem_cap) - node_memory_usage[node] * user_count

            if comp_cap < 0.0:
                comp_cap = 0.0
            if mem_cap < 0.0:
                mem_cap = 0.0

            self.remaining_computation_capacity[node][0] = comp_cap
            self.remaining_computation_capacity[node][1] = mem_cap

        # 2. 链路带宽扣减
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
    # 单条链搜索（在当前剩余资源下）
    # ----------------------------------------------------------------------
    def find_best_single_chain_for_objective(
        self, objective: str
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int]]]:
        """
        在当前剩余资源下，枚举所有部署方案，返回在给定目标下的最优单条链方案。

        objective:
            - "min_cost"
            - "max_profit"
            - "min_profit"
            - "max_users"

        返回:
            (cost, deploy_cost, comm_cost, profit,
             user_count, used_nodes_count, avg_modules_per_node, deployment)
            或 None（无可行方案）
        """
        n = self.node_count
        m = self.module_count

        if n <= 0 or m <= 0:
            return None

        best_solution: Optional[
            Tuple[float, float, float, float, int, int, float, List[int]]
        ] = None

        # 为了剪枝，维护 partial 下的 per-node usage 与部分带宽可行性
        node_compute_usage = [0.0] * n
        node_memory_usage = [0.0] * n
        deployment = [0] * m

        def partial_feasible(up_to_module_idx: int) -> bool:
            """
            对当前已经分配了 [0..up_to_module_idx] 的模块，做简单剪枝：
            - 任一节点的 per-user usage > 剩余 capacity，则即便 1 个用户也支持不了 → 不可行
            - 对已经形成的跨节点边界，如果对应链路带宽为 0 → 不可行
            """
            # 节点资源剪枝
            for node in range(n):
                comp_cap, mem_cap = self.get_node_capacity(node)
                comp_use = node_compute_usage[node]
                mem_use = node_memory_usage[node]

                # 如果对某节点 comp_use > comp_cap 或 mem_use > mem_cap，则连 1 用户也不行
                if comp_use > 0 and comp_cap <= 0:
                    return False
                if mem_use > 0 and mem_cap <= 0:
                    return False

            # 链路连通性 & 带宽剪枝
            for boundary_idx in range(up_to_module_idx):
                from_node = deployment[boundary_idx]
                to_node = deployment[boundary_idx + 1]
                if from_node == to_node:
                    continue
                bw = self.get_link_bandwidth(from_node, to_node)
                if bw <= 0:
                    return False

            return True

        def dfs(module_idx: int):
            nonlocal best_solution

            if module_idx == m:
                # 完整部署方案
                max_users = self.calculate_max_users_for_deployment(deployment)
                if max_users <= 0:
                    return

                cost, deploy_cost, comm_cost, profit = self.calculate_costs_for_deployment(
                    deployment, max_users
                )
                used_nodes_count = len(set(deployment))
                avg_mods_per_node = m / used_nodes_count if used_nodes_count > 0 else 0.0

                cand = (
                    cost,
                    deploy_cost,
                    comm_cost,
                    profit,
                    max_users,
                    used_nodes_count,
                    avg_mods_per_node,
                    deployment.copy(),
                )

                # 按 objective 更新 best_solution
                if best_solution is None:
                    best_solution = cand
                else:
                    (
                        b_cost,
                        _b_dep_c,
                        _b_comm_c,
                        b_profit,
                        b_users,
                        _b_used_nodes,
                        _b_avg,
                        _b_deploy,
                    ) = best_solution
                    c_cost, _, _, c_profit, c_users, _, _, _ = cand

                    if objective == "min_cost":
                        # 成本越小越好；成本相同，利润越高越好，其次用户数越多
                        if (
                            c_cost < b_cost
                            or (c_cost == b_cost and c_profit > b_profit)
                            or (c_cost == b_cost and c_profit == b_profit and c_users > b_users)
                        ):
                            best_solution = cand

                    elif objective == "max_profit":
                        # 利润越高越好；利润相同，用户数越多越好，其次成本越低
                        if (
                            c_profit > b_profit
                            or (c_profit == b_profit and c_users > b_users)
                            or (c_profit == b_profit and c_users == b_users and c_cost < b_cost)
                        ):
                            best_solution = cand

                    elif objective == "min_profit":
                        # 利润越低越差，找最差的；利润相同，成本越低
                        if (
                            c_profit < b_profit
                            or (c_profit == b_profit and c_cost < b_cost)
                        ):
                            best_solution = cand

                    elif objective == "max_users":
                        # 用户数越多越好；用户数相同，成本越低越好，其次利润越高
                        if (
                            c_users > b_users
                            or (c_users == b_users and c_cost < b_cost)
                            or (c_users == b_users and c_cost == b_cost and c_profit > b_profit)
                        ):
                            best_solution = cand

                return

            # 为第 module_idx 个模块选择节点（多功能：节点可以重复使用）
            for node in range(n):
                # 增量更新节点 usage
                comp_demand, mem_demand = self.get_module_demand(module_idx)
                node_compute_usage[node] += comp_demand
                node_memory_usage[node] += mem_demand
                deployment[module_idx] = node

                if partial_feasible(module_idx):
                    dfs(module_idx + 1)

                # 回溯
                node_compute_usage[node] -= comp_demand
                node_memory_usage[node] -= mem_demand

        dfs(0)
        return best_solution

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
        # 重置剩余资源为初始状态
        self.remaining_computation_capacity = copy.deepcopy(self.initial_computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.initial_bandwidth_matrix)

        total_cost = 0.0
        total_deploy_cost = 0.0
        total_comm_cost = 0.0
        total_profit = 0.0
        total_users = 0
        used_nodes_union = set()
        chain_count = 0

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

            # 从剩余资源中扣除本链消耗
            self.apply_chain_consumption(deployment, user_count)

        if chain_count == 0:
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

    # ----------------------------------------------------------------------
    # 对外主接口：返回 4 种策略吃干榨净后的全局效果
    # ----------------------------------------------------------------------
    def optimize_for_profit(self):
        """
        对同一网络 / 配置，在以下四种目标下分别「从初始资源出发，循环部署直到资源耗尽」：

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
