# EmotionalNPC - Registro de Refactor de Parámetros

Este README documenta el cambio de nombres del genoma para hacerlo más entendible y cercano al modelado conductual en vida real, manteniendo la base matemática del sistema.

## Registro de cambios de nombre

### Pesos perceptuales y de valencia

- w1 -> recepcion_sonoro
- w2 -> recepcion_amenaza
- w3 -> recepcion_nivel_luz
- w4 -> recepcion_estres
- w5 -> comfort_luz
- w6 -> vulnerabilidad_peligro
- w7 -> misofonia
- w8 -> vulnerabilidad_estres

### Sensibilidades de estrés

- k1 -> sensibilidad_amenaza
- k2 -> sensibilidad_auditiva
- k3 -> sensibilidad_oscuridad

### Reguladores de dinámica emocional

- dA -> volatilidad
- dV -> estabilidad_emocional
- ds -> resiliencia_estres

### Centros difusos de arousal

- cA_low -> estimulacion_optima_baja
- cA_medium -> estimulacion_optima_media
- cA_high -> estimulacion_optima_alta

### Centros difusos de valence

- cV_negative -> umbral_bienestar_negativo
- cV_neutral -> umbral_bienestar_neutral
- cV_positive -> umbral_bienestar_positivo

### Sigmas

- sigma -> tolerancia_estimulo
- nuevo -> tolerancia_incomodidad

Nota: por ahora el funcionamiento de inferencia se mantiene sin separación funcional completa de sigmas. El sistema conserva el comportamiento previo y usa tolerancia_estimulo como sigma efectiva para fuzzificación.

## Explicación de cada característica

- recepcion_sonoro: cuánto eleva el arousal el nivel sonoro.
- recepcion_amenaza: cuánto eleva el arousal la amenaza percibida.
- recepcion_nivel_luz: cuánto aporta la oscuridad relativa al arousal.
- recepcion_estres: cuánto influye el estrés interno en el arousal.
- comfort_luz: contribución positiva de la luz a la valencia.
- vulnerabilidad_peligro: impacto negativo del peligro sobre la valencia.
- misofonia: impacto negativo del sonido sobre la valencia.
- vulnerabilidad_estres: impacto negativo del estrés sobre la valencia.
- sensibilidad_amenaza: sensibilidad de acumulación de estrés ante amenaza.
- sensibilidad_auditiva: sensibilidad de acumulación de estrés ante sonido.
- sensibilidad_oscuridad: sensibilidad de acumulación de estrés ante oscuridad.
- volatilidad: rapidez de ajuste de arousal al objetivo instantáneo.
- estabilidad_emocional: inercia de ajuste de valencia al objetivo instantáneo.
- resiliencia_estres: velocidad de convergencia del estrés interno al estímulo.
- estimulacion_optima_baja/media/alta: centros gaussianos para etiquetas low/medium/high de arousal.
- umbral_bienestar_negativo/neutral/positivo: centros gaussianos para etiquetas negative/neutral/positive de valence.
- tolerancia_estimulo: anchura gaussiana principal del sistema difuso actual.
- tolerancia_incomodidad: segunda anchura gaussiana modelada para futura separación por eje.

## Fragmento de actualización emocional (esquema)

```python
stimulus_s = (
    params["sensibilidad_amenaza"] * environment.threat +
    params["sensibilidad_auditiva"] * environment.sound +
    params["sensibilidad_oscuridad"] * (1 - environment.light)
) / (
    params["sensibilidad_amenaza"] +
    params["sensibilidad_auditiva"] +
    params["sensibilidad_oscuridad"]
)

self.stress += (stimulus_s - self.stress) * params["resiliencia_estres"]

target_A = (
    params["recepcion_sonoro"] * environment.sound +
    params["recepcion_amenaza"] * environment.threat +
    params["recepcion_nivel_luz"] * (1 - environment.light) +
    params["recepcion_estres"] * self.stress
) / (
    params["recepcion_sonoro"] +
    params["recepcion_amenaza"] +
    params["recepcion_nivel_luz"] +
    params["recepcion_estres"]
)

self.Arousal += (target_A - self.Arousal) * params["volatilidad"]

target_V = (
    params["comfort_luz"] * environment.light -
    (
        params["vulnerabilidad_peligro"] * environment.threat +
        params["misofonia"] * environment.sound +
        params["vulnerabilidad_estres"] * self.stress
    )
)

target_V = max(-1, min(1, target_V))
target_V = (target_V + 1) / 2
self.Valence += (target_V - self.Valence) * params["estabilidad_emocional"]
```