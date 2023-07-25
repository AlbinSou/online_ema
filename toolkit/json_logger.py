################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2020-01-25                                                             #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import os
from typing import TYPE_CHECKING, List
import jsonlines

import torch
from avalanche.logging import BaseLogger
import collections


class ParallelJSONLogger(BaseLogger):

    def __init__(self, filename, autoupdate=False):
        super().__init__()
        self.filename = filename
        self.metric_dict = collections.defaultdict(lambda: {})
        self.autoupdate = autoupdate

    def log_single_metric(self, name, value, x_plot):
        self.metric_dict[x_plot][name] = value

        if self.autoupdate:
            self.update_json()

    def _convert_to_records(self, metric_dict):
        records = []
        for step, mdict in metric_dict.items():
            new_dict = {"step": step}
            new_dict.update(mdict)
            records.append(new_dict)
        return records

    def update_json(self):
        # Reset metric dict and put info in file
        records = self._convert_to_records(self.metric_dict)
        with jsonlines.open(self.filename, mode="a") as writer:
            writer.write_all(records)
        self.metric_dict = collections.defaultdict(lambda: {})
