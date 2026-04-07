import pandas as pd
import numpy as np
from genetic_algorithm import TSPGeneticAlgorithm
import os
from datetime import datetime

def load_distance_matrix():
    possible_paths = [
        '../Data/tsp_data_100_distance_matrix.csv',
        'Data/tsp_data_100_distance_matrix.csv',
        '../data/tsp_data_100_distance_matrix.csv',
        'data/tsp_data_100_distance_matrix.csv',
        'C:/Users/piyum/OneDrive/Desktop/tsp-genetic-algorithm/Data/tsp_data_100_distance_matrix.csv'
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        print("Error: Could not find distance matrix file!")
        print("Creating sample distance matrix...")
        return create_sample_distance_matrix()
    
    print(f"Loading distance matrix from: {file_path}")
    
    try:
        df = pd.read_csv(file_path, header=None)
        
        print(f"Raw data shape: {df.shape}")
        
        # Clean the data - remove any non-numeric characters
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace('[^0-9.-]', '', regex=True)
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN with 0
        df = df.fillna(0)
        
        distance_matrix = df.values.astype(float)
        
        # Check if matrix is square
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            print(f"Warning: Matrix not square! Shape: {distance_matrix.shape}")
            # Take minimum dimension to make it square
            min_dim = min(distance_matrix.shape)
            distance_matrix = distance_matrix[:min_dim, :min_dim]
            print(f"Adjusted to square matrix: {distance_matrix.shape}")
        
        print(f"Successfully loaded {distance_matrix.shape[0]} cities")
        print(f"Sample distance (0,1): {distance_matrix[0,1] if distance_matrix.shape[1] > 1 else 'N/A'}")
        
        return distance_matrix
        
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Creating sample distance matrix instead...")
        return create_sample_distance_matrix()

def create_sample_distance_matrix():
    num_cities = 50
    np.random.seed(42)
    cities = np.random.rand(num_cities, 2) * 100
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.sqrt(((cities[i] - cities[j])**2).sum())
    print(f"Created sample distance matrix with {num_cities} cities")
    return distance_matrix

def save_results(results_df, best_distance, best_route, history, best_history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_folder = "C:/Users/piyum/OneDrive/Desktop/tsp-genetic-algorithm/results"
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"Created results folder: {results_folder}")
    
    if len(results_df) > 0:
        parameter_results_path = f"{results_folder}/parameter_test_results_{timestamp}.csv"
        results_df.to_csv(parameter_results_path, index=False)
        print(f"\n📁 Saved: {parameter_results_path}")
    
    summary_path = f"{results_folder}/best_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TSP GENETIC ALGORITHM RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Distance: {best_distance:.2f}\n")
        f.write(f"Best Route Length: {len(best_route)}\n")
        f.write(f"Best Route (first 20): {best_route[:20]}...\n")
    
    print(f"📁 Saved: {summary_path}")
    
    if history and best_history:
        history_path = f"{results_folder}/convergence_history_{timestamp}.csv"
        history_df = pd.DataFrame({
            'generation': range(len(history)), 
            'best_overall': history,
            'best_generation': best_history[:len(history)]
        })
        history_df.to_csv(history_path, index=False)
        print(f"📁 Saved: {history_path}")
    
    if best_route:
        best_route_path = f"{results_folder}/best_route_{timestamp}.txt"
        with open(best_route_path, 'w') as f:
            f.write(','.join(map(str, best_route)))
        print(f"📁 Saved: {best_route_path}")
    
    return results_folder

def test_parameters():
    distance_matrix = load_distance_matrix()
    num_cities = len(distance_matrix)
    
    configs = [
        {"name": "Config 1 - Small", "pop_size": 100, "generations": 500, "mut_rate": 0.01, "tourn_size": 3, "selection": "tournament", "crossover": "ox", "mutation": "swap"},
        {"name": "Config 2 - Medium", "pop_size": 200, "generations": 800, "mut_rate": 0.015, "tourn_size": 5, "selection": "tournament", "crossover": "ox", "mutation": "swap"},
        {"name": "Config 3 - With PMX", "pop_size": 200, "generations": 800, "mut_rate": 0.015, "tourn_size": 5, "selection": "tournament", "crossover": "pmx", "mutation": "swap"},
        {"name": "Config 4 - Rank Selection", "pop_size": 200, "generations": 800, "mut_rate": 0.015, "tourn_size": 5, "selection": "rank", "crossover": "ox", "mutation": "swap"}
    ]
    
    results = []
    
    print("=" * 60)
    print("TSP GENETIC ALGORITHM - TESTING")
    print("=" * 60)
    print(f"Cities: {num_cities}")
    print("=" * 60)
    
    for config in configs:
        try:
            print(f"\nTesting {config['name']}...")
            print(f"  Population: {config['pop_size']}, Generations: {config['generations']}")
            print(f"  Mutation: {config['mut_rate']}, Tournament: {config['tourn_size']}")
            
            ga = TSPGeneticAlgorithm(
                distance_matrix=distance_matrix,
                population_size=config['pop_size'],
                generations=config['generations'],
                mutation_rate=config['mut_rate'],
                tournament_size=config['tourn_size'],
                elite_size=2
            )
            
            best_route, best_distance, history, best_history = ga.evolve(
                selection_type=config['selection'],
                crossover_type=config['crossover'],
                mutation_type=config['mutation'],
                use_2opt=True,
                use_3opt=False
            )
            
            results.append({
                'config': config['name'],
                'best_distance': float(best_distance),
                'population_size': config['pop_size'],
                'generations': config['generations'],
                'mutation_rate': config['mut_rate'],
                'tournament_size': config['tourn_size'],
                'selection': config['selection'],
                'crossover': config['crossover']
            })
            
            print(f"  ✓ Result: {best_distance:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("\nNo successful results!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_distance')
    
    print("\n" + "=" * 60)
    print("BEST CONFIGURATIONS")
    print("=" * 60)
    for idx, row in results_df.iterrows():
        print(f"{row['config']}: {row['best_distance']:.2f}")
    
    return results_df

def run_best_config():
    distance_matrix = load_distance_matrix()
    
    try:
        ga = TSPGeneticAlgorithm(
            distance_matrix=distance_matrix,
            population_size=200,
            generations=500,
            mutation_rate=0.015,
            tournament_size=5,
            elite_size=2
        )
        
        best_route, best_distance, history, best_history = ga.evolve(
            selection_type="tournament",
            crossover_type="ox",
            mutation_type="swap",
            use_2opt=True,
            use_3opt=False
        )
        
        print("\n" + "=" * 60)
        print(f"✅ BEST DISTANCE: {best_distance:.2f}")
        print("=" * 60)
        
        return best_route, best_distance, history, best_history
        
    except Exception as e:
        print(f"Error in run_best_config: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    results_df = test_parameters()
    
    best_route, best_distance, history, best_history = run_best_config()
    
    if best_route is not None:
        save_results(results_df, best_distance, best_route, history, best_history)
        
        print("\n" + "=" * 60)
        print("✅ RESULTS SAVED IN 'results' FOLDER")
        print("=" * 60)
    else:
        print("No results to save!")