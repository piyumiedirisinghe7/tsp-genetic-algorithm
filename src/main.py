import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
import time
from tqdm import tqdm
import os
from datetime import datetime
from plyer import notification

def load_cities(file_path="../data/cities.csv"):
    try:
        for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        cities = df[['x', 'y']].values
        print(f"✅ Successfully loaded {len(cities)} cities")
        return cities
    except FileNotFoundError:
        print("📁 Cities file not found. Creating 101 random cities...")
        np.random.seed(42)
        cities = np.random.rand(101, 2) * 100
        return cities

def calculate_distance_matrix(cities):
    return cdist(cities, cities, metric='euclidean')

def calculate_route_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route)):
        distance += distance_matrix[route[i], route[(i+1) % len(route)]]
    return distance

def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

def greedy_initialization(num_cities, distance_matrix):
    route = [0]
    unvisited = set(range(1, num_cities))
    
    while unvisited:
        last = route[-1]
        nearest = min(unvisited, key=lambda city: distance_matrix[last, city])
        route.append(nearest)
        unvisited.remove(nearest)
    
    return route

def hybrid_initialization(pop_size, num_cities, distance_matrix):
    population = []
    
    for i in range(pop_size):
        if i < pop_size // 4:
            route = greedy_initialization(num_cities, distance_matrix)
            random.shuffle(route)
        else:
            route = list(range(num_cities))
            random.shuffle(route)
        population.append(route)
    
    return population

def tournament_selection(population, fitness_scores, tournament_size=5):
    selected = []
    for _ in range(2):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx].copy())
    return selected[0], selected[1]

def roulette_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [f/total_fitness for f in fitness_scores]
    
    selected = []
    for _ in range(2):
        r = random.random()
        cumsum = 0
        for i, prob in enumerate(probabilities):
            cumsum += prob
            if r <= cumsum:
                selected.append(population[i].copy())
                break
    return selected[0], selected[1]

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]
    
    position = (end + 1) % size
    for gene in parent2:
        if gene not in child:
            child[position] = gene
            position = (position + 1) % size
    return child

def pmx_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child1 = [-1] * size
    child2 = [-1] * size
    
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]
    
    for i in range(start, end+1):
        if parent2[i] not in child1:
            pos = i
            while start <= pos <= end:
                try:
                    pos = parent1.index(parent2[pos])
                except ValueError:
                    break
            if start <= pos <= end:
                continue
            child1[pos] = parent2[i]
        
        if parent1[i] not in child2:
            pos = i
            while start <= pos <= end:
                try:
                    pos = parent2.index(parent1[pos])
                except ValueError:
                    break
            if start <= pos <= end:
                continue
            child2[pos] = parent1[i]
    
    for i in range(size):
        if child1[i] == -1:
            child1[i] = parent2[i]
        if child2[i] == -1:
            child2[i] = parent1[i]
    
    return child1, child2

def cycle_crossover(parent1, parent2):
    size = len(parent1)
    child1 = [-1] * size
    child2 = [-1] * size
    
    idx = 0
    while True:
        if child1[idx] == -1:
            while child1[idx] == -1:
                child1[idx] = parent1[idx]
                value = parent2[idx]
                idx = parent1.index(value)
            break
        idx = (idx + 1) % size
    
    for i in range(size):
        if child1[i] == -1:
            child1[i] = parent2[i]
        child2[i] = parent1[i] if child1[i] == parent2[i] else parent2[i]
    
    return child1, child2

def swap_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def inversion_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(route)), 2))
        route[i:j+1] = reversed(route[i:j+1])
    return route

def scramble_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(route)), 2))
        subset = route[i:j+1]
        random.shuffle(subset)
        route[i:j+1] = subset
    return route

def adaptive_mutation_rate(generation, total_generations, base_rate=0.02, final_rate=0.005):
    return base_rate - (base_rate - final_rate) * (generation / total_generations)

def calculate_fitness(population, distance_matrix):
    fitness_scores = []
    distances = []
    
    for route in population:
        distance = calculate_route_distance(route, distance_matrix)
        distances.append(distance)
        fitness_scores.append(1.0 / (distance ** 2))
    
    return fitness_scores, distances

def two_opt_improvement(route, distance_matrix):
    improved = True
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route, distance_matrix)
    
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                
                new_route = best_route.copy()
                new_route[i:j+1] = reversed(new_route[i:j+1])
                new_distance = calculate_route_distance(new_route, distance_matrix)
                
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
    
    return best_route

def send_victory_notification(distance, generation, improvement_percent):
    try:
        message = f"🥳💫🎊🎉 TARGET ACHIEVED! 🎉🎊💫🥳\n\n✨ Distance: {distance:.2f}\n⭐ Generation: {generation}\n📈 Improvement: {improvement_percent:.1f}%\n\n🏆 OPTIMAL ROUTE FOUND! 🏆"
        
        notification.notify(
            title="🎉 TSP GENETIC ALGORITHM - VICTORY! 🎉",
            message=message,
            app_name="TSP Solver",
            timeout=10
        )
        print("\n🔔✨ Desktop notification sent! ✨🔔")
    except:
        print("\n⚠️ Could not send desktop notification (plyer not configured properly)")

def genetic_algorithm(cities, pop_size=300, generations=1000, 
                     mutation_rate=0.02, tournament_size=5,
                     selection_type='tournament', 
                     crossover_type='ox',
                     mutation_type='inversion',
                     elite_size=5,
                     use_two_opt=True,
                     use_adaptive_mutation=True,
                     verbose=True):
    
    num_cities = len(cities)
    distance_matrix = calculate_distance_matrix(cities)
    
    print(f"\n{'='*60}")
    print(f"🎯 ENHANCED GENETIC ALGORITHM FOR TSP 🎯")
    print(f"{'='*60}")
    print(f"📍 Cities: {num_cities}")
    print(f"👥 Population size: {pop_size}")
    print(f"🔄 Generations: {generations}")
    print(f"🧬 Base mutation rate: {mutation_rate}")
    print(f"🏆 Tournament size: {tournament_size}")
    print(f"✅ Selection: {selection_type}")
    print(f"🔄 Crossover: {crossover_type}")
    print(f"⚡ Mutation: {mutation_type}")
    print(f"👑 Elite size: {elite_size}")
    print(f"🔧 2-opt improvement: {use_two_opt}")
    print(f"📊 Adaptive mutation: {use_adaptive_mutation}")
    print(f"{'='*60}\n")
    
    print("🌱 Initializing population with hybrid method...")
    population = hybrid_initialization(pop_size, num_cities, distance_matrix)
    
    fitness_scores, distances = calculate_fitness(population, distance_matrix)
    best_distance = min(distances)
    best_route = population[np.argmin(distances)].copy()
    target_achieved_notified = False
    
    if use_two_opt:
        print("🔧 Applying 2-opt to initial best solution...")
        best_route = two_opt_improvement(best_route, distance_matrix)
        best_distance = calculate_route_distance(best_route, distance_matrix)
    
    history = {
        'best_distances': [best_distance],
        'avg_distances': [np.mean(distances)],
        'worst_distances': [max(distances)]
    }
    
    start_time = time.time()
    no_improvement_count = 0
    
    print("🚀 Running Enhanced Genetic Algorithm...")
    print("🎯 Target: Distance < 800\n")
    
    for generation in tqdm(range(generations), desc="Progress", unit="gen"):
        
        current_mutation_rate = mutation_rate
        if use_adaptive_mutation:
            current_mutation_rate = adaptive_mutation_rate(generation, generations, mutation_rate, 0.005)
        
        new_population = []
        
        if elite_size > 0:
            elite_indices = np.argsort(distances)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
        
        while len(new_population) < pop_size:
            if selection_type == 'tournament':
                parent1, parent2 = tournament_selection(population, fitness_scores, tournament_size)
            else:
                parent1, parent2 = roulette_selection(population, fitness_scores)
            
            if crossover_type == 'ox':
                child1 = ordered_crossover(parent1, parent2)
                child2 = ordered_crossover(parent2, parent1)
            elif crossover_type == 'pmx':
                child1, child2 = pmx_crossover(parent1, parent2)
            elif crossover_type == 'cycle':
                child1, child2 = cycle_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            if mutation_type == 'swap':
                child1 = swap_mutation(child1, current_mutation_rate)
                child2 = swap_mutation(child2, current_mutation_rate)
            elif mutation_type == 'inversion':
                child1 = inversion_mutation(child1, current_mutation_rate)
                child2 = inversion_mutation(child2, current_mutation_rate)
            elif mutation_type == 'scramble':
                child1 = scramble_mutation(child1, current_mutation_rate)
                child2 = scramble_mutation(child2, current_mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
        
        fitness_scores, distances = calculate_fitness(population, distance_matrix)
        
        current_best_idx = np.argmin(distances)
        current_best_distance = distances[current_best_idx]
        current_best_route = population[current_best_idx].copy()
        
        if use_two_opt and (generation % 50 == 0 or current_best_distance < best_distance):
            current_best_route = two_opt_improvement(current_best_route, distance_matrix)
            current_best_distance = calculate_route_distance(current_best_route, distance_matrix)
        
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route.copy()
            no_improvement_count = 0
            
            if best_distance < 800 and not target_achieved_notified:
                improvement_pct = ((history['best_distances'][0] - best_distance) / history['best_distances'][0]) * 100
                print(f"\n{'='*60}")
                print(f"🥳💫🎊🎉 TARGET ACHIEVED! 🎉🎊💫🥳")
                print(f"📍 Best Distance: {best_distance:.2f}")
                print(f"🔄 Generation: {generation + 1}")
                print(f"📈 Improvement: {improvement_pct:.1f}%")
                print(f"🏆 CONGRATULATIONS! OPTIMAL ROUTE FOUND! 🏆")
                print(f"{'='*60}\n")
                send_victory_notification(best_distance, generation + 1, improvement_pct)
                target_achieved_notified = True
        else:
            no_improvement_count += 1
        
        history['best_distances'].append(best_distance)
        history['avg_distances'].append(np.mean(distances))
        history['worst_distances'].append(max(distances))
        
        if verbose and (generation + 1) % 100 == 0:
            elapsed = time.time() - start_time
            current_best = best_distance
            if current_best < 800:
                target_status = "✅🥳 TARGET ACHIEVED! 🎉✨"
            else:
                target_status = f"🎯 Need {current_best - 800:.1f} more"
            print(f"\n  📊 Gen {generation+1}/{generations} | 🏆 Best: {current_best:.2f} | 📈 Avg: {np.mean(distances):.2f} | {target_status}")
        
        if no_improvement_count > 200:
            print(f"\n  ⏹️ Early stopping at generation {generation+1} (no improvement for 200 gens)")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    if best_distance < 800:
        print(f"🥳💫🎊🎉 VICTORY! TARGET ACHIEVED! 🎉🎊💫🥳")
        print(f"✨ Best Distance: {best_distance:.2f} (Below 800!) ✨")
        print(f"🏆 CONGRATULATIONS! 🏆")
    else:
        print(f"📊 ALGORITHM COMPLETED!")
        print(f"📍 Best Distance: {best_distance:.2f}")
        print(f"🎯 Target: 800 (Need {best_distance - 800:.1f} less)")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    return best_route, best_distance, history

def plot_route(cities, route, title="TSP Route"):
    plt.figure(figsize=(12, 8))
    
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=2)
    
    route_cities = cities[route]
    plt.plot(route_cities[:, 0], route_cities[:, 1], 'b-', linewidth=2, alpha=0.7)
    
    plt.scatter(cities[route[0], 0], cities[route[0], 1], 
               c='green', s=200, marker='*', zorder=3, label='Start')
    
    plt.scatter(cities[route[-1], 0], cities[route[-1], 1], 
               c='orange', s=200, marker='s', zorder=3, label='End')
    
    plt.title(title, fontsize=16)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def plot_convergence(history, title="Convergence Plot"):
    plt.figure(figsize=(12, 6))
    
    generations = range(len(history['best_distances']))
    
    plt.plot(generations, history['best_distances'], 'g-', linewidth=2, label='Best Distance')
    plt.plot(generations, history['avg_distances'], 'b-', linewidth=1.5, alpha=0.7, label='Average Distance')
    plt.plot(generations, history['worst_distances'], 'r-', linewidth=1, alpha=0.5, label='Worst Distance')
    
    plt.axhline(y=800, color='orange', linestyle='--', linewidth=2, label='Target (800)')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def save_results(results_df, best_distance, best_route, history, config_name):
    try:
        results_dir = 'results'
        
        absolute_path = os.path.join(os.getcwd(), results_dir)
        print(f"💾 Saving to: {absolute_path}")
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"📁 Created '{results_dir}' directory")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_file = os.path.join(results_dir, f'summary_{config_name}_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write(f"TSP ENHANCED GENETIC ALGORITHM RESULTS\n")
            f.write(f"{'='*60}\n")
            f.write(f"Config: {config_name}\n")
            f.write(f"Best Distance: {best_distance:.2f}\n")
            if best_distance < 800:
                f.write(f"STATUS: 🥳💫🎊🎉 TARGET ACHIEVED! 🎉🎊💫🥳\n")
            else:
                f.write(f"STATUS: Target not reached (Need {best_distance - 800:.1f} less)\n")
            f.write(f"Best Route: {best_route}\n")
            f.write(f"Final Generation: {len(history['best_distances'])-1}\n")
            f.write(f"Initial Best: {history['best_distances'][0]:.2f}\n")
            if history['best_distances'][0] > 0:
                improvement_pct = (history['best_distances'][0] - best_distance)/history['best_distances'][0]*100
                f.write(f"Improvement: {(history['best_distances'][0] - best_distance):.2f} ({improvement_pct:.2f}%)\n")
            f.write(f"{'='*60}\n")
        
        route_file = os.path.join(results_dir, f'route_{config_name}_{timestamp}.csv')
        route_df = pd.DataFrame({'city_index': best_route, 'order': range(len(best_route))})
        route_df.to_csv(route_file, index=False)
        
        history_file = os.path.join(results_dir, f'history_{config_name}_{timestamp}.csv')
        history_df = pd.DataFrame(history)
        history_df.to_csv(history_file, index=False)
        
        print(f"\n✅ Results saved to '{results_dir}/' folder")
        print(f"  📄 Summary: {summary_file}")
        print(f"  🗺️ Route: {route_file}")
        print(f"  📊 History: {history_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR saving results: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_parameter_testing(cities):
    print("\n" + "="*60)
    print("🧪 TSP ENHANCED GENETIC ALGORITHM - PARAMETER TESTING 🧪")
    print("="*60)
    print(f"📍 Number of cities: {len(cities)}")
    print("="*60)
    
    configs = [
        {
            "name": "Config_1_Standard",
            "pop_size": 200,
            "generations": 800,
            "mutation_rate": 0.02,
            "tournament_size": 5,
            "selection_type": "tournament",
            "crossover_type": "ox",
            "mutation_type": "inversion",
            "elite_size": 5,
            "use_two_opt": True,
            "use_adaptive_mutation": True
        },
        {
            "name": "Config_2_Large",
            "pop_size": 400,
            "generations": 1000,
            "mutation_rate": 0.025,
            "tournament_size": 7,
            "selection_type": "tournament",
            "crossover_type": "ox",
            "mutation_type": "inversion",
            "elite_size": 8,
            "use_two_opt": True,
            "use_adaptive_mutation": True
        }
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n\n{'='*60}")
        print(f"🧬 Testing {config['name']}...")
        print(f"{'='*60}")
        
        try:
            best_route, best_distance, history = genetic_algorithm(
                cities=cities,
                pop_size=config['pop_size'],
                generations=config['generations'],
                mutation_rate=config['mutation_rate'],
                tournament_size=config['tournament_size'],
                selection_type=config['selection_type'],
                crossover_type=config['crossover_type'],
                mutation_type=config['mutation_type'],
                elite_size=config['elite_size'],
                use_two_opt=config['use_two_opt'],
                use_adaptive_mutation=config['use_adaptive_mutation'],
                verbose=True
            )
            
            if best_route is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_results(None, best_distance, best_route, history, config['name'])
                
                fig1 = plot_route(cities, best_route, f"TSP Route - {config['name']} (Distance: {best_distance:.2f})")
                fig1.savefig(f'results/route_plot_{config["name"]}_{timestamp}.png', dpi=150, bbox_inches='tight')
                plt.close(fig1)
                
                fig2 = plot_convergence(history, f"Convergence - {config['name']}")
                fig2.savefig(f'results/convergence_{config["name"]}_{timestamp}.png', dpi=150, bbox_inches='tight')
                plt.close(fig2)
                
                all_results.append({
                    'config': config['name'],
                    'best_distance': best_distance,
                    'generations': config['generations'],
                    'pop_size': config['pop_size'],
                    'target_achieved': best_distance < 800
                })
                
                if best_distance < 800:
                    print(f"\n🥳💫🎊🎉 {config['name']} completed! Distance: {best_distance:.2f} - TARGET ACHIEVED! 🎉🎊💫🥳")
                else:
                    print(f"\n📊 {config['name']} completed! Distance: {best_distance:.2f}")
            else:
                print(f"\n❌ Failed {config['name']} - no results!")
                
        except Exception as e:
            print(f"\n⚠️ Error in {config['name']}: {str(e)}")
            continue
    
    print("\n\n" + "="*60)
    print("📊 PARAMETER TESTING SUMMARY 📊")
    print("="*60)
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        print("\n🏆 Best configuration:")
        best_config = results_df.loc[results_df['best_distance'].idxmin()]
        print(f"  ✨ {best_config['config']} - Distance: {best_config['best_distance']:.2f}")
        
        if best_config['best_distance'] < 800:
            print(f"\n🥳💫🎊🎉 TARGET ACHIEVED! Distance {best_config['best_distance']:.2f} is below 800! 🎉🎊💫🥳")
        else:
            print(f"\n⚠️ Target not reached yet. Best distance: {best_config['best_distance']:.2f} (Need {best_config['best_distance'] - 800:.1f} less)")
    else:
        print("❌ No results to display!")
    
    print("="*60)

def main():
    print("\n" + "="*60)
    print("🎯 ENHANCED TSP GENETIC ALGORITHM 🎯")
    print("🥳💫🎊🎉 Target: Distance < 800 🎉🎊💫🥳")
    print("="*60)
    
    cities = load_cities()
    
    if len(cities) == 0:
        print("❌ Error: No cities loaded!")
        return
    
    print("\n📋 Select mode:")
    print("1. ⚡ Fast run (for quick testing)")
    print("2. ⭐ Standard run (recommended for target 800)")
    print("3. 🔥 Extreme run (for best results)")
    print("4. 🧪 Parameter testing (multiple configurations)")
    print("5. 🛠️ Custom parameters")
    
    mode = input("\n👉 Enter choice (1/2/3/4/5): ").strip()
    
    if mode == '1':
        print("\n⚡ Running FAST mode...")
        best_route, best_distance, history = genetic_algorithm(
            cities=cities,
            pop_size=150,
            generations=400,
            mutation_rate=0.02,
            tournament_size=5,
            selection_type='tournament',
            crossover_type='ox',
            mutation_type='inversion',
            elite_size=3,
            use_two_opt=True,
            use_adaptive_mutation=True,
            verbose=True
        )
        
        if best_route is not None:
            save_results(None, best_distance, best_route, history, "fast_run")
            fig1 = plot_route(cities, best_route, f"Best TSP Route (Distance: {best_distance:.2f})")
            plt.show()
            fig2 = plot_convergence(history, "Convergence Plot")
            plt.show()
            
            if best_distance < 800:
                print(f"\n🥳💫🎊🎉 SUCCESS! Distance {best_distance:.2f} is below 800! 🎉🎊💫🥳")
            else:
                print(f"\n📊 Distance: {best_distance:.2f}. Try Standard or Extreme mode for better results.")
    
    elif mode == '2':
        print("\n⭐ Running STANDARD mode (Recommended for target 800)...")
        best_route, best_distance, history = genetic_algorithm(
            cities=cities,
            pop_size=300,
            generations=800,
            mutation_rate=0.02,
            tournament_size=5,
            selection_type='tournament',
            crossover_type='ox',
            mutation_type='inversion',
            elite_size=5,
            use_two_opt=True,
            use_adaptive_mutation=True,
            verbose=True
        )
        
        if best_route is not None:
            save_results(None, best_distance, best_route, history, "standard_run")
            fig1 = plot_route(cities, best_route, f"Best TSP Route (Distance: {best_distance:.2f})")
            plt.show()
            fig2 = plot_convergence(history, "Convergence Plot")
            plt.show()
            
            if best_distance < 800:
                print(f"\n🥳💫🎊🎉 SUCCESS! Distance {best_distance:.2f} is below 800! 🎉🎊💫🥳")
            else:
                print(f"\n📊 Distance: {best_distance:.2f}. Try Extreme mode for better results.")
    
    elif mode == '3':
        print("\n🔥 Running EXTREME mode (For best results)...")
        best_route, best_distance, history = genetic_algorithm(
            cities=cities,
            pop_size=500,
            generations=1200,
            mutation_rate=0.025,
            tournament_size=7,
            selection_type='tournament',
            crossover_type='ox',
            mutation_type='scramble',
            elite_size=10,
            use_two_opt=True,
            use_adaptive_mutation=True,
            verbose=True
        )
        
        if best_route is not None:
            save_results(None, best_distance, best_route, history, "extreme_run")
            fig1 = plot_route(cities, best_route, f"Best TSP Route (Distance: {best_distance:.2f})")
            plt.show()
            fig2 = plot_convergence(history, "Convergence Plot")
            plt.show()
            
            if best_distance < 800:
                print(f"\n🥳💫🎊🎉 AMAZING! Distance {best_distance:.2f} is below 800! 🎉🎊💫🥳")
            else:
                print(f"\n📊 Distance: {best_distance:.2f}. Consider running for more generations.")
    
    elif mode == '4':
        run_parameter_testing(cities)
    
    elif mode == '5':
        print("\n🛠️ Enter custom parameters:")
        pop_size = int(input("👥 Population size (default 300): ") or "300")
        generations = int(input("🔄 Generations (default 800): ") or "800")
        mutation_rate = float(input("🧬 Mutation rate (default 0.02): ") or "0.02")
        tournament_size = int(input("🏆 Tournament size (default 5): ") or "5")
        use_two_opt = input("🔧 Use 2-opt improvement? (y/n, default y): ").lower() != 'n'
        
        best_route, best_distance, history = genetic_algorithm(
            cities=cities,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            selection_type='tournament',
            crossover_type='ox',
            mutation_type='inversion',
            elite_size=5,
            use_two_opt=use_two_opt,
            use_adaptive_mutation=True,
            verbose=True
        )
        
        if best_route is not None:
            save_results(None, best_distance, best_route, history, "custom_run")
            fig1 = plot_route(cities, best_route, f"Best TSP Route (Distance: {best_distance:.2f})")
            plt.show()
            fig2 = plot_convergence(history, "Convergence Plot")
            plt.show()
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()