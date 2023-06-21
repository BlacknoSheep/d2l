import os
import sys
from datetime import datetime


class Logger:
    def __init__(self):
        self.logs = list()
        self.breaker = "="*40  # 行分隔符
        self.date = datetime.now().strftime("%Y-%m-%d %H-%M-%S")  # 当前日期和时间

        self.add(self.date)
        self.add_breaker()

    def add(self, *logs):
        for log in logs:
            self.logs.append(log)

    def add_breaker(self):
        """添加行分隔符"""
        self.add(self.breaker)

    def save(self, path):
        with open(path, 'w') as f:
            sys.stdout = f
            for log in self.logs:
                print(log)
            sys.stdout = sys.__stdout__

    def print(self):
        for log in self.logs:
            print(log)

logger = Logger()
