import random
import pandas as pd

# 1. Definimos los límites (DNA_BOUNDS) según la documentación
DNA_BOUNDS = {
    # caracteristicas perceptuales semánticos (rango [0.0, 1.0])
    "recepcion_sonoro": (0.0, 1.0),
    "recepcion_amenaza": (0.0, 1.0),
    "recepcion_nivel_luz": (0.0, 1.0),
    "recepcion_estres": (0.0, 1.0),
    "comfort_luz": (0.0, 1.0),
    "vulnerabilidad_peligro": (0.0, 1.0),
    "misofonia": (0.0, 1.0),
    "vulnerabilidad_estres": (0.0, 1.0),
    
    # Constantes sensoriales (rango [0.05, 1.0])
    "sensibilidad_amenaza": (0.05, 1.0),
    "sensibilidad_auditiva": (0.05, 1.0),
    "sensibilidad_oscuridad": (0.05, 1.0),
    
    # Tasas de regulación emocional (rango [0.05, 1.0])
    "volatilidad": (0.05, 1.0), "estabilidad_emocional": (0.05, 1.0), "resiliencia_estres": (0.05, 1.0),
    
    # Centros de fuzificación (rango [0.0, 1.0])
    "estimulacion_optima_baja": (0.0, 1.0), "estimulacion_optima_media": (0.0, 1.0), "estimulacion_optima_alta": (0.0, 1.0),
    "umbral_bienestar_negativo": (0.0, 1.0), "umbral_bienestar_neutral": (0.0, 1.0), "umbral_bienestar_positivo": (0.0, 1.0),
    
    # Dispersión de las gaussianas (rango [0.05, 0.35])
    "tolerancia_estimulo": (0.05, 0.35),
    "tolerancia_incomodidad": (0.05, 0.35),
}

def generate_random_genome():
    """Genera un individuo (diccionario) respetando los límites y el orden lógico."""
    # Muestreo uniforme dentro de los límites
    genome = {gen: random.uniform(limites[0], limites[1]) 
              for gen, limites in DNA_BOUNDS.items()}
    
    # PASO DE REPARACIÓN DETERMINISTA: 
    # El documento exige que los centros respeten un orden estricto semántico.
    
    # Ordenar centros de Arousal (low < medium < high)
    centros_A = sorted([
        genome["estimulacion_optima_baja"],
        genome["estimulacion_optima_media"],
        genome["estimulacion_optima_alta"],
    ])
    (
        genome["estimulacion_optima_baja"],
        genome["estimulacion_optima_media"],
        genome["estimulacion_optima_alta"],
    ) = centros_A
    
    # Ordenar centros de Valencia (negative < neutral < positive)
    centros_V = sorted([
        genome["umbral_bienestar_negativo"],
        genome["umbral_bienestar_neutral"],
        genome["umbral_bienestar_positivo"],
    ])
    (
        genome["umbral_bienestar_negativo"],
        genome["umbral_bienestar_neutral"],
        genome["umbral_bienestar_positivo"],
    ) = centros_V
    
    return genome

# 2. Generar la Población Inicial (10 individuos)
POPULATION_SIZE = 10
poblacion = [generate_random_genome() for _ in range(POPULATION_SIZE)]

# 3. Convertir a matriz tabular (DataFrame) y visualizar
df = pd.DataFrame(poblacion)
df.index.name = "Individuo"

# Configurar pandas para mostrar todas las columnas al imprimir
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Matriz de Inicialización del Algoritmo Genético (10 Individuos x 22 Genes)\n")
print(df.round(4)) # Redondeamos a 4 decimales para mejorar la lectura