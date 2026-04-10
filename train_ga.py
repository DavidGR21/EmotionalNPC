import json
import os
from genetic.genome import generate_random_genome, crossover, mutate
from genetic.fitness import evaluate_fitness

POPULATION_SIZE = 50
GENERATIONS = 30
MUTATION_RATE = 0.1
ELITISM = 5  # Cuantos de los mejores se salvan directamente a la siguiente generación

def train_genetic_algorithm():
    print("=== Iniciando Entrenamiento Evolutivo (Motor Genético) ===")
    
    # 1. Población Inicial
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    
    best_overall_genome = None
    best_overall_fitness = float('-inf')

    for generation in range(1, GENERATIONS + 1):
        # 2. Evaluar Adaptabilidad (Fitness)
        scored_population = []
        for genome in population:
            score = evaluate_fitness(genome)
            scored_population.append((score, genome))
            
        # Ordenar de mejor a peor
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        current_best_score = scored_population[0][0]
        current_best_genome = scored_population[0][1]
        current_avg_score = sum(s for s, g in scored_population) / POPULATION_SIZE
        
        # Tracking del mejor global
        if current_best_score > best_overall_fitness:
            best_overall_fitness = current_best_score
            best_overall_genome = current_best_genome.copy()

        print(f"Gen {generation:02d} | Mejor Score: {current_best_score:>6.1f} | Media: {current_avg_score:>6.1f}")
        
        # Si estamos en la última generación, no necesitamos cruzar más
        if generation == GENERATIONS:
            break
            
        # 3. Selección y Cruce (Elitism + Ruleta/Torneo truncado)
        next_population = []
        
        # Elitismo (pasamos a los mejores sin mutar)
        for i in range(ELITISM):
            next_population.append(scored_population[i][1])
            
        # Generar el resto de la población
        # Los padres se seleccionan primariamente de la mitad superior
        best_half = [g for s, g in scored_population[:POPULATION_SIZE//2]]
        
        import random
        while len(next_population) < POPULATION_SIZE:
            parent1 = random.choice(best_half)
            parent2 = random.choice(best_half)
            
            # Cruce
            child = crossover(parent1, parent2)
            
            # Mutación
            child = mutate(child, mutation_rate=MUTATION_RATE)
            
            next_population.append(child)
            
        population = next_population
        
    print("\n=== Entrenamiento Finalizado ===")
    print(f"Mejor Score Global alcanzado: {best_overall_fitness}")
    print("Mejores Parámetros ('Params' Optimizados):")
    
    # se imprime bonito en formato JSON para poder copiar 
    print(json.dumps(best_overall_genome, indent=4))
    
    # se podria guardar en personalities.json de manera programática también.
    return best_overall_genome

if __name__ == "__main__":
    train_genetic_algorithm()
