#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:40:17 2025

@author: zoz
"""


import torch

from collections import defaultdict
from typing import List



# Aggregation strategies 
class Aggregator:
    def __init__(self):
        self.server_momentum = None
        self.server_opt_state = defaultdict(float)
        
    def fed_avg(self, local_updates: List[dict]) -> dict:
        avg_weights = {}
        for key in local_updates[0].keys():
            avg_weights[key] = torch.stack(
                [update[key].float() for update in local_updates]
            ).mean(dim=0)
        return avg_weights

    def fed_avg_momentum(self, local_updates: List[dict], beta=0.9) -> dict:
        current_update = self.fed_avg(local_updates)
        if self.server_momentum is None:
            self.server_momentum = current_update
        else:
            for key in current_update:
                self.server_momentum[key] = beta * self.server_momentum[key] + (1 - beta) * current_update[key]
        return self.server_momentum

    def fed_adagrad(self, local_updates: List[dict], eta=0.01, epsilon=1e-8) -> dict:
        current_update = self.fed_avg(local_updates)
        for key in current_update:
            self.server_opt_state[key] += current_update[key].float() ** 2
            current_update[key] = eta * current_update[key] / (torch.sqrt(self.server_opt_state[key]) + epsilon)
        return current_update