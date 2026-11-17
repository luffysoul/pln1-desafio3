# Desafío 3 — Procesamiento de Lenguaje Natural I (CEIA, FIUBA)

**Alumno:** Jaime Ricardo Pinzon Aparicio

Modelo de lenguaje con **tokenización por caracteres** entrenado sobre *Don Quijote de la Mancha* (Proyecto Gutenberg, eBook #2000).

## Contenido

El notebook [`Desafio_3.ipynb`](Desafio_3.ipynb) cubre:

1. **Corpus y dataset** — limpieza (sin aparato editorial ni índice), vocabulario de 39 caracteres y dataset *many-to-many* con contexto 100. Se corrigen dos problemas documentados de la plantilla: el tamaño de las ventanas de validación y el padding con un índice que es un caracter real.
2. **Arquitecturas** — SimpleRNN, LSTM y GRU (200 unidades), comparadas por **perplejidad en validación** con early stopping, más una SimpleRNN de 380 unidades como **control de capacidad**:

   | Modelo | Perplejidad en validación |
   |---|---|
   | LSTM | 4,15 |
   | GRU | 4,17 |
   | SimpleRNN | 4,48 |
   | SimpleRNN de 380 unidades | 4,87 |

   La SimpleRNN de 380 unidades tiene más parámetros que la GRU y aun así sobreajusta, así que la ventaja de las celdas con compuertas no se debe a su cantidad de parámetros.
3. **Generación** — greedy, beam search determinístico y beam search estocástico con temperatura (T = 0,3 / 0,7 / 1,0 / 1,5). Se mide:
   - log p por caracter;
   - diversidad (trigramas distintos);
   - palabras inventadas.

   Las estrategias se comparan contra muestreo ancestral puro y contra texto real.

Cada experimento está acompañado de su interpretación y toda cifra citada proviene de un output visible del notebook.

## Ejecución

```bash
pip install keras torch numpy pandas scipy matplotlib seaborn
jupyter notebook Desafio_3.ipynb
```

Corre con **Keras 3 y backend PyTorch** (GPU recomendada; ~2 h de entrenamiento en una GTX 1050). El corpus se descarga automáticamente de Gutenberg y queda cacheado en `quijote.txt`.
