import numpy as np
import pandas as pd
from typing import List

class TSPData:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.distance_matrix = None
        self.num_cities = 0
        self.load_data()
    
    def load_data(self):
        print(f"Loading: {self.file_path}")
        df = pd.read_csv(self.file_path, index_col=0)
        self.distance_matrix = df.values.astype(float)
        self.num_cities = len(df)
        print(f"Loaded {self.num_cities} cities")
    
    def get_distance(self, city_a: int, city_b: int) -> float:
        return self.distance_matrix[city_a][city_b]
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        total = 0.0
        for i in range(len(tour) - 1):
            total += self.get_distance(tour[i], tour[i+1])
        total += self.get_distance(tour[-1], tour[0])
        return total