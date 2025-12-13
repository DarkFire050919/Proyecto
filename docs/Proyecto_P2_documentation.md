# Proyecto_P2.py – Clasificador de Spam con PyTorch

## 📄 Descripción general
Este script implementa un clasificador binario de correos electrónicos spam/not‑spam utilizando una red neuronal densa (fully‑connected) construida con **PyTorch**. El conjunto de datos proviene del repositorio **UCI Spambase** ([UCI Machine Learning Repository – Spambase](https://archive.ics.uci.edu/ml/datasets/spambase)).

El flujo de trabajo incluye:
1. **Carga y pre‑procesamiento** de los datos (train/test split, normalización).  
2. Definición de la arquitectura de la red neuronal con capas lineales, activaciones ReLU, dropout y salida sigmoide.  
3. **Entrenamiento** mediante el optimizador Adam y la función de pérdida binaria cross‑entropy (BCELoss).  
4. **Evaluación** en el conjunto de prueba con precisión (accuracy).  
5. **Visualización** de la curva de pérdida y de precisión a lo largo de las épocas.

---

## 🛠️ Requisitos e instalación
```bash
# Entorno de Python (>=3.8)
pip install torch torchvision matplotlib scikit-learn ucimlrepo
```
- **torch** – Framework de deep learning.
- **matplotlib** – Generación de gráficas.
- **scikit‑learn** – Funciones auxiliares (train_test_split, StandardScaler, accuracy_score).
- **ucimlrepo** – Cliente para descargar el dataset Spambase.

---

## ⚙️ Configuración de hiperparámetros (justificación)
| Parámetro | Valor | Razonamiento |
|-----------|-------|--------------|
| `LR` (learning rate) | `0.001` | Valor estándar para el optimizador Adam; evita oscilaciones y garantiza convergencia estable. |
| `EPOCHS` | `500` | El dataset tiene 57 características; se requiere suficiente número de iteraciones para una convergencia suave. |
| `HIDDEN_1` | `128` | Expande la dimensionalidad de entrada (57) a 128 para capturar combinaciones no lineales complejas. |
| `HIDDEN_2` | `64` | Reducción progresiva (embudo) que sintetiza la información aprendida en la capa anterior. |
| `DROPOUT_RATE` | `0.4` | Apaga el 40 % de neuronas durante el entrenamiento, reduciendo el riesgo de over‑fitting. |

---

## 📂 Funciones principales
### `load_and_preprocess_data()`
- Descarga el dataset Spambase mediante `fetch_ucirepo(id=94)`.
- Separa **features** (`X`) y **etiquetas** (`y`).
- Divide los datos en *train* (80 %) y *test* (20 %) con `train_test_split` (semilla `random_state=42`).
- Normaliza las características usando `StandardScaler` (media 0, varianza 1).  
- Convierte los arrays a tensores `torch.float` y los devuelve en un diccionario.

### `SpamClassifier(nn.Module)`
```text
Input (57) → Linear(57, 128) → ReLU → Dropout(0.4)
      → Linear(128, 64) → ReLU → Linear(64, 1) → Sigmoid
```
- **Salida**: probabilidad entre 0 y 1.
- **Regularización**: dropout activo solo en modo *train*.

---

## 🏋️‍♂️ Proceso de entrenamiento
```python
model = SpamClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(data['X_train'])
    loss = criterion(outputs, data['y_train'])
    loss.backward()
    optimizer.step()
    # Registro de loss y accuracy
```
- Cada 50 épocas se imprime la pérdida y la precisión de entrenamiento.
- Se almacenan `loss_history` y `accuracy_history` para graficar.

---

## 📊 Evaluación final
```python
model.eval()
with torch.no_grad():
    test_outputs = model(data['X_test'])
    test_predicted = (test_outputs > 0.5).float()
    final_acc = accuracy_score(data['y_test'], test_predicted)
```
- **Precisión final** (accuracy) se muestra en consola, por ejemplo:
```
Accuracy Final en Test Set: 93.45%
```
*(El valor exacto dependerá de la semilla y de los parámetros de entrenamiento.)*

---

## 📈 Visualizaciones
Dos sub‑gráficas se generan con **matplotlib**:
1. **Curva de pérdida (Loss) vs. épocas** – muestra la disminución de la función objetivo.
2. **Curva de precisión (Accuracy) vs. épocas** – evidencia el progreso del modelo durante el entrenamiento.
```python
plt.subplot(1, 2, 1)  # Loss
plt.subplot(1, 2, 2)  # Accuracy
plt.show()
```
> Si se desea guardar la figura, descomentar la línea `plt.savefig('resultados_entrenamiento.png')`.

---

## 🚀 Uso rápido
```bash
python Proyecto_P2.py
```
El script entrenará el modelo y mostrará por pantalla:
- Progreso por cada 50 épocas (Loss y Accuracy de entrenamiento).
- Precisión final en el conjunto de prueba.
- Ventana de visualización con las dos curvas.

---

## 📚 Referencias y recursos
- **UCI Machine Learning Repository – Spambase**: https://archive.ics.uci.edu/ml/datasets/spambase
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **scikit‑learn**: https://scikit-learn.org/
- **ucimlrepo** (Python client): https://pypi.org/project/ucimlrepo/

---

## ✏️ Comentarios y posibles extensiones
- **Cross‑validation** para una estimación más robusta del rendimiento.
- **Ajuste de hiperparámetros** mediante búsqueda en cuadrícula o algoritmos evolutivos.
- **Persistencia del modelo** (`torch.save`) para reutilizar el clasificador sin re‑entrenar.
- **Métricas adicionales** (precision, recall, F1‑score, ROC‑AUC) para evaluar el sesgo del dataset.

*Fin del documento.*