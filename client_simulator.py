import urllib.request
import json
import time
import random

BASE_URL = "http://localhost:8000"
NPC_ID = "Personaje"
PERSONALITY = "calmada"

print("=============================================")
print("  SIMULADOR DEL MOTOR GRÁFICO O CLIENTE      ")
print("=============================================")

# 1. Simular: Godot inicia la escena y manda a spawnear al NPC
print(f"\n🎮 [GODOT] Spawneando NPC '{NPC_ID}' con cerebro '{PERSONALITY}'...")
init_req = urllib.request.Request(
    f"{BASE_URL}/npc/init",
    data=json.dumps({"id": NPC_ID, "personality": PERSONALITY}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(init_req) as response:
        print("✅ [BACKEND RESPONDE]:", json.loads(response.read().decode("utf-8")))
except Exception as e:
    print(f"❌ Error al conectar al Backend ({e}).\n¿Está encendido el servidor? (Abre otra términal y ejecuta 'python main.py')")
    exit(1)

# 2. Simular: El Timer de Godot (cada x milisegundos envía los datos sensoriales al backend)
print("\n🎮 [GODOT] Iniciando bucle de juego... (Simulando 1 tick por segundo)")
for tick in range(1, 6): # Simula 5 iteraciones en el juego
    
    # Supongamos que en este frame, el jugador de Godot se acercó haciendo ruido:
    sensor_data = {
        "sound": round(random.uniform(0.1, 1.0), 2),
        "threat": round(random.uniform(0.1, 1.0), 2),
        "light": 0.5  # Asumimos que están en un bosque lúgubre
    }
    
    print(f"\n[Tick {tick}] 📤 Godot detecta en la escena: {sensor_data}")
    
    update_req = urllib.request.Request(
        f"{BASE_URL}/npc/{NPC_ID}/update",
        data=json.dumps(sensor_data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        # Petición HTTP Cruda, equivalente al nodo HTTPRequest en Godot
        with urllib.request.urlopen(update_req) as response:
            ai_data = json.loads(response.read().decode("utf-8"))
            
            action = ai_data["decision"].upper()
            attitude = ai_data["attitude"]
            emotion = ai_data.get("emotion", "desconocida")
            
            print(f"           📥 IA Emocion: {emotion.upper()} | Accion: >> {action} << ({attitude})")
            print(
                "           📊 Detalles del cerebro: "
                f"Estrés={ai_data['metrics']['stress']}, "
                f"Top emociones={ai_data['top_options']}"
            )
            
    except Exception as e:
        print("❌ Error en Godot al intentar hacer fetch:", e)
        
    time.sleep(1.0) # Espera simulación del timer
    
print("\n🎮 [GODOT] Simulación del cliente finalizada.")
