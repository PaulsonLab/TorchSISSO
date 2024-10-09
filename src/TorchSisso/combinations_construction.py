



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 00:56:39 2024

@author: muthyala.7
"""
import torch
import re

class FeatureConstructor:
    def __init__(self, df_feature_values, columns):
        
        self.columns = columns
        self.device = 'cpu'
        self.df_feature_values = df_feature_values

    def apply_function_tensor(self, tensor, func):
        
        if func.startswith('pow'):
            power_value = int(re.search(r'pow\((\d+)\)', func).group(1))
            return tensor ** power_value
        elif func == 'div':
            return torch.div(tensor, 1)  # Default division by 1 if no second argument
        elif func.startswith('-'):
            func_name = func[1:]
            return -getattr(torch, func_name)(tensor)
        else:
            return getattr(torch, func)(tensor)

    def apply_operation(self, tensor1, tensor2, op):
        # Apply operation defined in torch (op is now passed directly from feature_def)
        return op(tensor1, tensor2)

    def construct_generic_features(self, feature_definition):
        n_features = len(self.columns)

        # Create a grid of combinations
        x, y = torch.meshgrid(torch.arange(n_features), torch.arange(n_features), indexing='ij')

        # Stack and reshape the results
        combinations_indices = torch.stack((x.flatten(), y.flatten()), dim=1)

        # Create a 3D tensor of shape (n_combinations, n_samples, 2)
        comb_tensor = self.df_feature_values[:, combinations_indices].permute(1, 0, 2)

        results = []
        expressions = []

        for part in feature_definition:
            var1_idx = 0
            var2_idx = 1
            
            # Ensure the indices are within bounds of the 3D tensor's last dimension
            if var1_idx >= 2 or var2_idx >= 2:
                raise IndexError(f"var1_idx or var2_idx is out of bounds. Maximum index allowed is 1, got var1_idx={var1_idx}, var2_idx={var2_idx}")
            
            functions1 = part['functions1']
            functions2 = part['functions2']
            op = part['operation']  # Operation is now part of the feature definition
            final_functions = part.get('final_functions', [])  # Get final functions if they exist

            var1 = comb_tensor[:, :, var1_idx]
            var2 = comb_tensor[:, :, var2_idx]

            # Apply functions to the first variable only if there are any functions specified
            if functions1:
                for func in functions1:
                    var1 = self.apply_function_tensor(var1, func)

            # Apply functions to the second variable only if there are any functions specified
            if functions2:
                for func in functions2:
                    var2 = self.apply_function_tensor(var2, func)

            # Apply the operation between the two variables
            op_result = self.apply_operation(var1, var2, op)

            # Apply final functions on the operation result
            if len(final_functions)>=1:
                for func in final_functions:
                    op_result = self.apply_function_tensor(op_result, func)

            # Create expression strings
            expr1_template = [self.columns[idx.item()] for idx in combinations_indices[:, var1_idx]]
            expr2_template = [self.columns[idx.item()] for idx in combinations_indices[:, var2_idx]]

            # Apply functions to expression strings
            for func in functions1:
                if func.startswith('pow'):
                    power_value = re.search(r'pow\((\d+)\)', func).group(1)
                    expr1_template = [f"({expr})**{power_value}" for expr in expr1_template]
                elif func.startswith('-'):
                    expr1_template = [f"-{expr}" for expr in expr1_template]
                else:
                    expr1_template = [f"{func}({expr})" for expr in expr1_template]

            for func in functions2:
                if func.startswith('pow'):
                    power_value = re.search(r'pow\((\d+)\)', func).group(1)
                    expr2_template = [f"({expr})**{power_value}" for expr in expr2_template]
                elif func.startswith('-'):
                    expr2_template = [f"-{expr}" for expr in expr2_template]
                else:
                    expr2_template = [f"{func}({expr})" for expr in expr2_template]

            # Combine expressions with the operation
            combined_expr = [f"({expr1}) {op.__name__} ({expr2})" for expr1, expr2 in zip(expr1_template, expr2_template)]
            
            # Combine final expressions only if final functions are specified
            final_expr = combined_expr
            
            
            if final_functions:
                final_expr = []
                final_expr = [f"{func}(({expr1}) {op.__name__} ({expr2}))" for expr1, expr2 in zip(expr1_template, expr2_template) for func in final_functions]


            
            results.append(op_result)
            expressions.extend(final_expr)

        return torch.cat(results, dim=1).T, expressions


