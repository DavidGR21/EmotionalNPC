import json
import os
import sys
from genetic.genome import generate_random_genome, crossover, mutate
from genetic.fitness import get_fitness_evaluator

# ==========================================
# CONFIGURACIÓN DEL ALGORITMO GENÉTICO (HIPERPARÁMETROS)
# ==========================================
POPULATION_SIZE = 80   # Cantidad de NPCs arrojados a la arena por cada ciclo.
GENERATIONS = 50       # Cuántas veces ocurrirá la evolución/supervivencia.
MUTATION_RATE = 0.1    # Probabilidad del 10% de que los hijos nazcan con alguna pequeña alteración aleatoria.
ELITISM = 5            # Cuántos de los "Súper Padres" ganadores pasan directo a la siguiente generación intactos.


def train_genetic_algorithm(
    personality_type="normal",
    population_size=POPULATION_SIZE,
    generations=GENERATIONS,
    mutation_rate=MUTATION_RATE,
    elitism=ELITISM,
):
    print(f"=== Iniciando Entrenamiento Evolutivo para: {personality_type.upper()} ===")
    
    # 0. Instanciar a nuestra Fábrica dependiendo de lo pedido (Normal, Cobarde, etc.)
    evaluate_fitness = get_fitness_evaluator(personality_type)
    
    # 1. POBLACIÓN INICIAL (DÍA CERO)
    # Creamos 'N' cantidad de NPCs con genes completamente aleatorios.
    population = [generate_random_genome() for _ in range(population_size)]
    
    # Memoria a lo largo del tiempo (Para no perder el mejor, incluso si luego nacen peores hijos)
    best_overall_genome = None
    best_overall_fitness = float('-inf')  # Infinito Negativo garantiza que cualquier score sea mejor

    # ----------------------------------------------
    # BUCLE EVOLUTIVO (EL TIMESHIFT DE GENERACIONES)
    # ----------------------------------------------
    for generation in range(1, generations + 1):
        
        # 2. EVALUACIÓN DE SUPERVIVENCIA (FITNESS)
        scored_population = []
        for genome in population:
            # Enviamos el genotipo a la 'Matrix' / simulación. Y nos devolverá un score.
            score = evaluate_fitness(
                genome,
                generation=generation,
                total_generations=generations,
            )
            scored_population.append((score, genome))
            
        # Ordenamos a la población. Los de score altísimo van al inicio, los perdedores van al final.
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        # Estadísticas de esta generación actual
        current_best_score = scored_population[0][0]     # Score del alfa (Indice 0)
        current_best_genome = scored_population[0][1]    # Sus genes
        current_avg_score = sum(s for s, g in scored_population) / population_size  # Score promedio de todos
        
        # ¿El mejor individuo de esta ronda superó al rey de todos los tiempos? Si es así, se corona.
        if current_best_score > best_overall_fitness:
            best_overall_fitness = current_best_score
            best_overall_genome = current_best_genome.copy()

        # Log por consola para ver cómo la naturaleza "aprende"
        print(f"Gen {generation:02d} | Mejor Score: {current_best_score:>6.1f} | Media: {current_avg_score:>6.1f}")
        
        # Si ya llegamos a nuestro límite (ej. Gen 30), cerramos el ciclo y no nos molestamos en cruzar.
        if generation == generations:
            break
            
        # 3. SELECCIÓN NATURAL Y REPRODUCCIÓN (PARA EL SIGUIENTE CICLO)
        # La población original muere y los hijos tomarán su lugar.
        next_population = []
        
        # A. ELITISMO: La naturaleza perdona. Protegemos a los "ELITISM" mejores de morir y los ponemos directo en la nueva lista.
        for i in range(elitism):
            next_population.append(scored_population[i][1])
            
        # B. SELECCIÓN: Elegiremos como 'Padres Creadores' solo a la **Mitad Ganadora** (los fuertes).
        # Los que quedaron en la mitad perdedora son eliminados del pool genético (su ADN muere aquí).
        best_half = [g for s, g in scored_population[:population_size//2]]
        
        # Bucle hasta rellenar de habitantes (50 total)
        import random
        while len(next_population) < population_size:
            # Elegimos al azar 2 padres del grupo de los supervivientes
            parent1 = random.choice(best_half)
            parent2 = random.choice(best_half)
            
            # Cruzamos sus genes
            child = crossover(parent1, parent2)
            
            # Mutamos alguna probabilidad aleatoria del hijo (radiación, evolución genética al azar)
            child = mutate(child, mutation_rate=mutation_rate)
            
            # Agregamos este nuevo súper hijo al siguiente ciclo.
            next_population.append(child)
            
        # Actualizar la variable (Termina la generación actual, avanza el tiempo de todos al nacer la Gen +1)
        population = next_population
        
    print("\n=== Entrenamiento Finalizado ===")
    print(f"Mejor Score Global alcanzado: {best_overall_fitness}")
    
    # ==========================================
    # GUARDADO PERSISTENTE EN BASE DE DATOS LOCAL
    # ==========================================
    # Para que este esfuerzo no se pierda al apagar el PC, lo guardaremos.
    os.makedirs("data", exist_ok=True)
    json_path = os.path.join("data", "personalities.json")
    
    db = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                db = json.load(f)
            except json.JSONDecodeError:
                db = {}
                
    # Insertar o reemplazar la personalidad elegida con los genes del Campeón Mundial Absoluto
    db[personality_type] = best_overall_genome
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)
        
    print(f"\n[Exito] Personalidad '{personality_type}' guardada en en {json_path}")
    return best_overall_genome

if __name__ == "__main__":
    # ARGUMENTOS DE CONSOLA: 
    # sys.argv capta los textos detrás de 'python script.py'
    # Ejemplo: 'python train_ga.py valiente' -> sys.argv[1] == 'valiente'
    target_personality = "normal"
    population_size = POPULATION_SIZE
    generations = GENERATIONS
    mutation_rate = MUTATION_RATE
    elitism = ELITISM

    if len(sys.argv) > 1:
        target_personality = sys.argv[1].lower()

    # Uso opcional:
    # py train_ga.py normal 80 50 0.1 5
    if len(sys.argv) > 2:
        population_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        generations = int(sys.argv[3])
    if len(sys.argv) > 4:
        mutation_rate = float(sys.argv[4])
    if len(sys.argv) > 5:
        elitism = int(sys.argv[5])
        
    train_genetic_algorithm(
        target_personality,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        elitism=elitism,
    )
