# Game-of-Life

Tarea 2 - Computación en GPU, Universidad de Chile, Otoño 2025.

Autores: José Pereira, Belén Vásquez

## Descripción
Este proyecto implementa el clásico Game of Life de Conway en distintas versiones: una versión secuencial (CPU), una versión paralela usando CUDA, y otra usando OpenCL. Se realizan múltiples experimentos para comparar rendimiento en distintas configuraciones de ejecución.
## Requisitos

- [CLion](https://www.jetbrains.com/clion/) 
- CMake ≥ 3.31
- Compilador C++ compatible con C++17
- CUDA Toolkit (para ejecutar las versiones CUDA)
- Controlador de GPU y entorno con soporte para OpenCL (para las versiones OpenCL)

## Cómo ejecutar el código en CLion

1. Abre **CLion**.
2. Selecciona `File > Open...` y elige la carpeta raíz del proyecto (`GameOfLife/`).
3. CLion detectará automáticamente el archivo `CMakeLists.txt` y configurará el proyecto.
4. Abre `src/main.cpp` y haz clic en el ícono de ▶ al lado de la función `main()`, o usa el botón verde **Run** arriba.
5. La salida de la simulación aparecerá en la pestaña inferior de **Run**.

## Resultados
El programa ejecuta automáticamente todos los experimentos:

1. Game of Life en CPU
2. Game of Life en CUDA:
   - Versión base
   - Múltiplos de 32
   - No múltiplos de 32
   - 2D
3. Game of Life en OpenCL:
   - Versión base 
   - Múltiplos de 32 
   - No múltiplos de 32 
   - 2D

Los resultados se guardan en archivos `.txt`, por ejemplo:
- `resultados.txt`
- `resultados_cuda.txt`
- `resultadosOpenCL.txt`
(y más según cada configuración)

Cada archivo incluye:
- Dimensiones de la grilla 
- Número de iteraciones 
- Tiempo de ejecución 
- Total de celdas procesadas 
- Celdas evaluadas por segundo

