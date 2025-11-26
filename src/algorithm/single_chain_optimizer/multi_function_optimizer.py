#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多功能部署优化器
基于分支定界法实现，完全基于物理限制计算最大用户量，没有随机因素
"""

import os
import sys
import json
import numpy as np
import math
from collections import defaultdict
import heapq
import copy

class MultiFunctionOptimizer:
    """
    多功能部署优化器类，每个节点可以部署多个功能
    基于分支定界法实现
    """
    
    def __init__(self, test_data):
        """
        初始化多功能部署优化器
        
        Args:
            test_data (dict): 测试数据字典，包含如下字段：
                - node_count: 节点数量
                - module_count: 模块数量
                - computation_capacity: 计算能力列表 n×2的矩阵，第1列是算力，第2列是存储
                - resource_demands: 资源需求列表 m×2的矩阵，第1列是算力，第2列是存储
                - data_sizes: 数据大小列表，长度为m-1
                - bandwidth_matrix: 带宽矩阵 n×n的矩阵
                - gpu_cost: GPU成本系数
                - memory_cost: 内存成本系数
                - bandwidth_cost: 带宽成本系数
                - profit_per_user: 每个用户的利润
                - node_costs: 每个节点的GPU和内存成本 n×2的矩阵
                - distance_matrix: 节点间距离矩阵 n×n的矩阵
        """
        # 保存test_data_id用于调试
        self.test_data_id = test_data.get('test_data_id', 0)
        
        # 基础配置
        self.node_count = int(test_data['node_count'])
        self.module_count = int(test_data['module_count'])
        
        # 处理可能是字符串的输入
        self.computation_capacity = self._parse_array(test_data['computation_capacity'])
        self.resource_demands = self._parse_array(test_data['resource_demands'])
        self.data_sizes = self._parse_array(test_data['data_sizes'])
        self.bandwidth_matrix = self._parse_array(test_data['bandwidth_matrix'])
        
        # 成本参数
        self.gpu_cost = float(test_data['gpu_cost'])
        self.memory_cost = float(test_data['memory_cost'])
        self.bandwidth_cost = float(test_data['bandwidth_cost'])
        self.profit_per_user = float(test_data['profit_per_user'])
        
        # 节点特定的成本和距离
        self.node_costs = self._parse_array(test_data.get('node_costs', None))
        self.distance_matrix = self._parse_array(test_data.get('distance_matrix', None))
        
        # 如果未提供节点特定成本，使用全局成本
        if self.node_costs is None:
            self.node_costs = [[self.gpu_cost, self.memory_cost] for _ in range(self.node_count)]
        
        # 如果未提供距离矩阵，使用默认距离1
        if self.distance_matrix is None:
            self.distance_matrix = [[1 if i != j else 0 for j in range(self.node_count)] for i in range(self.node_count)]
        
        # 初始化最优解
        self.best_deployment = None  # 最优部署方案
        self.best_cost = float('inf')  # 最小成本
        self.best_profit = float('-inf')  # 最大利润
        self.best_user_count = 0  # 最大用户数
        
        # 记录所有满足条件的解
        self.all_solutions = []
    
    def _parse_array(self, data):
        """解析数据，如果是JSON字符串则转换为列表"""
        if data is None:
            return None
            
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return data
    
    def get_node_capacity(self, node):
        """获取节点的计算和存储能力"""
        if node < 0 or node >= self.node_count:
            return 0, 0
        return self.computation_capacity[node][0], self.computation_capacity[node][1]
    
    def get_module_demand(self, module):
        """获取模块的计算和存储需求"""
        if module < 0 or module >= self.module_count:
            return 0, 0
        return self.resource_demands[module][0], self.resource_demands[module][1]
    
    def get_link_bandwidth(self, from_node, to_node):
        """获取两个节点之间的带宽"""
        if from_node < 0 or from_node >= self.node_count or to_node < 0 or to_node >= self.node_count:
            return 0
        return self.bandwidth_matrix[from_node][to_node]
    
    def get_link_distance(self, from_node, to_node):
        """获取两个节点之间的距离"""
        if from_node < 0 or from_node >= self.node_count or to_node < 0 or to_node >= self.node_count:
            return 1  # 默认距离为1
        return self.distance_matrix[from_node][to_node]
    
    def get_data_size(self, module):
        """获取模块输出的数据大小"""
        if module < 0 or module >= self.module_count - 1:
            return 0
        return self.data_sizes[module]
    
    def get_node_costs(self, node):
        """获取节点的GPU和内存成本"""
        if node < 0 or node >= self.node_count:
            return self.gpu_cost, self.memory_cost  # 默认使用全局成本
        return self.node_costs[node][0], self.node_costs[node][1]
    
    def calculate_max_users(self, deployment):
        """
        计算给定部署方案的最大用户数量
        基于木桶原理，找到所有资源限制中的最小值
        并向下取整以确保返回整数用户数
        
        Args:
            deployment (list): 部署方案，每个元素是节点索引
            
        Returns:
            int: 最大用户数量(整数)
        """
        # 1. 计算节点资源限制
        node_resource_limits = []
        
        # 统计每个节点的资源使用情况
        node_compute_usage = defaultdict(float)
        node_storage_usage = defaultdict(float)
        
        for module_idx, node_idx in enumerate(deployment):
            compute_demand, storage_demand = self.get_module_demand(module_idx)
            node_compute_usage[node_idx] += compute_demand
            node_storage_usage[node_idx] += storage_demand
        
        # 计算每个节点的最大用户数 - 按节点索引排序以确保确定性
        sorted_nodes = sorted(set(deployment))
        for node_idx in sorted_nodes:
            compute_capacity, storage_capacity = self.get_node_capacity(node_idx)
            
            # 如果算力需求大于0，计算基于算力的最大用户数
            if node_compute_usage[node_idx] > 0:
                compute_limit = math.floor(compute_capacity / node_compute_usage[node_idx])
                node_resource_limits.append(compute_limit)
            
            # 如果存储需求大于0，计算基于存储的最大用户数
            if node_storage_usage[node_idx] > 0:
                storage_limit = math.floor(storage_capacity / node_storage_usage[node_idx])
                node_resource_limits.append(storage_limit)
        
        # 2. 计算链路带宽限制
        link_bandwidth_limits = []
        
        # 检查每对相邻模块
        for i in range(self.module_count - 1):
            from_node = deployment[i]
            to_node = deployment[i + 1]
            
            # 如果相邻模块部署在同一节点上，不需要考虑带宽
            if from_node == to_node:
                continue
            
            # 检查带宽是否为0
            bandwidth = self.get_link_bandwidth(from_node, to_node)
            if bandwidth <= 0:
                # 如果带宽为0，不能支持任何用户
                return 0
            
            # 计算基于带宽的最大用户数
            data_size = self.get_data_size(i)
            if data_size > 0:  # 避免除以零
                link_limit = math.floor(bandwidth / data_size)
                link_bandwidth_limits.append(link_limit)
        
        # 3. 木桶原理：系统整体最大用户量 = 所有限制中的最小值
        all_limits = node_resource_limits + link_bandwidth_limits
        if not all_limits:
            return 0  # 如果没有有效限制，返回0
        
        return min(all_limits)  # 最小值已经是取整后的结果
    
    def calculate_total_cost(self, deployment, user_count):
        """
        计算给定部署方案的总成本
        
        Args:
            deployment (list): 部署方案，每个元素是节点索引
            user_count (float): 用户数量
            
        Returns:
            tuple: (total_cost, compute_storage_cost, communication_cost) 总成本、计算存储成本、通信成本
        """
        # 1. 计算计算和存储成本
        compute_storage_cost = 0
        
        # 统计每个节点的资源使用情况
        node_compute_usage = defaultdict(float)
        node_storage_usage = defaultdict(float)
        
        for module_idx, node_idx in enumerate(deployment):
            compute_demand, storage_demand = self.get_module_demand(module_idx)
            node_compute_usage[node_idx] += compute_demand * user_count
            node_storage_usage[node_idx] += storage_demand * user_count
        
        # 计算每个节点的成本 - 按节点索引排序以确保确定性
        sorted_nodes = sorted(set(deployment))
        for node_idx in sorted_nodes:
            gpu_cost, memory_cost = self.get_node_costs(node_idx)
            node_cost = (node_compute_usage[node_idx] * gpu_cost + 
                         node_storage_usage[node_idx] * memory_cost)
            compute_storage_cost += node_cost
        
        # 2. 计算通信成本
        communication_cost = 0
        
        # 检查每对相邻模块
        for i in range(self.module_count - 1):
            from_node = deployment[i]
            to_node = deployment[i + 1]
            
            # 如果相邻模块部署在同一节点上，不需要考虑通信成本
            if from_node == to_node:
                continue
            
            # 计算通信成本 = 数据大小 * 带宽成本 * 距离 * 用户数
            data_size = self.get_data_size(i)
            distance = self.get_link_distance(from_node, to_node)
            
            communication_cost += data_size * self.bandwidth_cost * distance * user_count
        
        # 3. 总成本 = 计算存储成本 + 通信成本
        total_cost = compute_storage_cost + communication_cost
        
        return total_cost, compute_storage_cost, communication_cost
    
    def is_deployment_feasible(self, deployment):
        """
        检查部署方案是否可行
        
        Args:
            deployment (list): 部署方案，每个元素是节点索引
            
        Returns:
            bool: 是否可行
        """
        # 长度检查
        if len(deployment) != self.module_count:
            return False
        
        # 节点索引范围检查
        for node_idx in deployment:
            if node_idx < 0 or node_idx >= self.node_count:
                return False
        
        # 检查每对相邻模块的连通性
        for i in range(self.module_count - 1):
            from_node = deployment[i]
            to_node = deployment[i + 1]
            
            # 如果在同一节点上，肯定连通
            if from_node == to_node:
                continue
            
            # 检查带宽是否大于0
            if self.get_link_bandwidth(from_node, to_node) <= 0:
                return False
        
        # 计算每个节点的资源使用情况
        node_compute_usage = defaultdict(float)
        node_storage_usage = defaultdict(float)
        
        for module_idx, node_idx in enumerate(deployment):
            compute_demand, storage_demand = self.get_module_demand(module_idx)
            node_compute_usage[node_idx] += compute_demand
            node_storage_usage[node_idx] += storage_demand
        
        # 检查每个节点的资源是否足够
        for node_idx, compute_usage in node_compute_usage.items():
            compute_capacity, storage_capacity = self.get_node_capacity(node_idx)
            
            if compute_usage > compute_capacity:
                return False
            
            if node_storage_usage[node_idx] > storage_capacity:
                return False
        
        return True
    
    def search_all_deployments(self):
        """
        使用深度优先搜索方法搜索所有可行的部署方案
        使用分支定界法进行剪枝
        """
        # 清空之前的结果
        self.all_solutions = []
        
        # 使用栈进行深度优先搜索
        stack = [([], 0)]  # (当前部署方案, 当前模块索引)
        attempted = 0
        
        print(f"开始搜索部署方案，test_data_id={self.test_data_id}")
        
        while stack:
            current_deployment, module_idx = stack.pop()
            
            # 如果已经部署了所有模块，评估该方案
            if module_idx == self.module_count:
                attempted += 1
                # 计算最大用户数量
                max_users = self.calculate_max_users(current_deployment)
                
                if max_users > 0:  # 如果方案可行
                    # 计算成本
                    total_cost, compute_storage_cost, communication_cost = self.calculate_total_cost(current_deployment, max_users)
                    # 计算利润
                    profit = max_users * self.profit_per_user - total_cost
                    
                    # 保存方案
                    self.all_solutions.append((total_cost, profit, max_users, current_deployment))
                    print(f"{self.test_data_id}找到可行方案，最大用户数：{max_users}，利润：{profit}")
                
                continue
            
            # 尝试将当前模块部署到每个节点 - 确保节点遍历顺序固定（从高到低）
            node_indices = list(range(self.node_count))
            for node_idx in reversed(node_indices):  # 倒序遍历确保栈弹出顺序是正序
                # 创建新的部署方案
                new_deployment = current_deployment + [node_idx]
                
                # 剪枝判断：如果已经不可行，就不继续探索
                if not self._is_partial_feasible(new_deployment, module_idx):
                    continue
                
                # 将新方案加入栈中
                stack.append((new_deployment, module_idx + 1))
        
        # 根据需要对所有方案进行排序 - 使用稳定排序
        if self.all_solutions:
            # 先按照确定的标准进行预排序，避免相同值顺序的不确定性
            self.all_solutions.sort(key=lambda x: tuple(x[3]))  # 首先按部署方案排序
            # 然后再按主要标准排序
            self.all_solutions.sort(key=lambda x: (x[0], -x[2]))  # 先按成本升序，再按用户数降序
            print(f"共找到 {len(self.all_solutions)} 个可行方案")
        else:
            print(f"尝试了 {attempted} 个方案，但未找到任何可行方案")
            # 打印节点容量和模块需求信息以便调试
            for node_idx in range(self.node_count):
                compute, storage = self.get_node_capacity(node_idx)
                print(f"节点 {node_idx} 容量：计算={compute}，存储={storage}")
            
            for module_idx in range(self.module_count):
                compute, storage = self.get_module_demand(module_idx)
                print(f"模块 {module_idx} 需求：计算={compute}，存储={storage}")
    
    def _is_partial_feasible(self, partial_deployment, current_module):
        """
        检查部分部署方案是否可行
        
        Args:
            partial_deployment: 部分部署方案
            current_module: 当前已部署到的模块索引
            
        Returns:
            bool: 是否可行
        """
        # 边界检查
        if not partial_deployment or current_module < 0:
            return True
        
        # 检查最后部署的模块
        latest_module = current_module
        node_idx = partial_deployment[latest_module]
        
        # 检查节点资源是否足够
        compute_capacity, storage_capacity = self.get_node_capacity(node_idx)
        
        # 计算节点上已经部署的模块的资源使用
        node_compute_usage = 0
        node_storage_usage = 0
        
        for module_idx, deployed_node in enumerate(partial_deployment):
            if deployed_node == node_idx:
                compute_demand, storage_demand = self.get_module_demand(module_idx)
                node_compute_usage += compute_demand
                node_storage_usage += storage_demand
        
        # 检查资源是否足够
        if node_compute_usage > compute_capacity or node_storage_usage > storage_capacity:
            if self.test_data_id <= 5:  # 只对前几个测试用例输出详细信息
                print(f"资源不足: 节点{node_idx} 计算能力={compute_capacity}, 计算需求={node_compute_usage}")
                print(f"资源不足: 节点{node_idx} 存储能力={storage_capacity}, 存储需求={node_storage_usage}")
            return False
        
        # 如果不是第一个模块，检查与前一个模块的连通性
        if latest_module > 0:
            prev_node = partial_deployment[latest_module - 1]
            
            # 如果不在同一节点，检查带宽
            if prev_node != node_idx and self.get_link_bandwidth(prev_node, node_idx) <= 0:
                if self.test_data_id <= 5:  # 只对前几个测试用例输出详细信息
                    print(f"带宽不足: 节点{prev_node}到节点{node_idx}之间没有带宽")
                return False
        
        return True
    
    def optimize_for_profit(self):
        """
        优化利润
        
        Returns:
            tuple: (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)
            每个方案是 (成本, 计算存储成本, 通信成本, 利润, 用户数, 使用节点数, 平均模块/节点)
        """
        # 搜索所有可行的部署方案
        self.search_all_deployments()
        
        # 如果没有找到任何解
        if not self.all_solutions:
            return None
        
        # 为每个方案计算详细成本
        detailed_solutions = []
        for solution in self.all_solutions:
            cost, profit, user_count, deployment = solution
            _, compute_storage_cost, communication_cost = self.calculate_total_cost(deployment, user_count)
            
            # 计算使用的节点数量
            used_nodes_count = len(set(deployment))
            
            # 计算平均每节点模块数
            avg_modules_per_node = self.module_count / used_nodes_count if used_nodes_count > 0 else 0
            
            detailed_solutions.append((cost, compute_storage_cost, communication_cost, profit, user_count, 
                                      used_nodes_count, avg_modules_per_node, deployment))
        
        # 分离成本、利润和用户数
        costs = [s[0] for s in detailed_solutions]
        profits = [s[3] for s in detailed_solutions]
        user_counts = [s[4] for s in detailed_solutions]
        
        # 找到最小成本方案 - 使用稳定的排序和选择逻辑
        # 现在按成本升序、利润降序排序，确保在相同成本下选择利润最高的
        min_cost = min(costs)
        min_cost_candidates = [(i, s) for i, s in enumerate(detailed_solutions) if s[0] == min_cost]
        min_cost_candidates.sort(key=lambda x: (-x[1][3], x[1][0], tuple(x[1][7])))  # 按负利润（即利润降序）、成本升序排序
        min_cost_idx = min_cost_candidates[0][0]
        min_cost_plan = detailed_solutions[min_cost_idx][:7]  # 取成本、计算存储成本、通信成本、利润、用户数、节点数、平均模块数
        
        # 找到最大利润方案 - 使用稳定的排序和选择逻辑
        max_profit = max(profits)
        max_profit_candidates = [(i, s) for i, s in enumerate(detailed_solutions) if s[3] == max_profit]
        max_profit_candidates.sort(key=lambda x: (x[1][0], tuple(x[1][7])))  # 按成本、部署方案排序
        max_profit_idx = max_profit_candidates[0][0]
        max_profit_plan = detailed_solutions[max_profit_idx][:7]
        
        # 找到最小利润方案 - 使用稳定的排序和选择逻辑
        min_profit = min(profits)
        min_profit_candidates = [(i, s) for i, s in enumerate(detailed_solutions) if s[3] == min_profit]
        min_profit_candidates.sort(key=lambda x: (x[1][0], tuple(x[1][7])))  # 按成本、部署方案排序
        min_profit_idx = min_profit_candidates[0][0]
        min_profit_plan = detailed_solutions[min_profit_idx][:7]
        
        # 找到最大用户数方案 - 使用稳定的排序和选择逻辑
        max_users = max(user_counts)
        max_users_candidates = [(i, s) for i, s in enumerate(detailed_solutions) if s[4] == max_users]
        # 按用户量降序、成本升序、利润降序排序
        max_users_candidates.sort(key=lambda x: (-x[1][4], x[1][0], -x[1][3], tuple(x[1][7])))
        max_users_idx = max_users_candidates[0][0]
        max_users_plan = detailed_solutions[max_users_idx][:7]
        
        return (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan) 