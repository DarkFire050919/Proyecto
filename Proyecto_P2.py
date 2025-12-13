import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# --- CONFIGURACIÓN Y JUSTIFICACIÓN DE HIPERPARÁMETROS  ---
# Learning Rate (0.001): Valor estándar para Adam. Si es muy alto, oscila; muy bajo, tarda en converger.
LR = 0.001
# Epocas (500): Necesarias para asegurar convergencia suave en un dataset complejo de 57 dimensiones.
EPOCHS = 500
# Neuronas Capa 1 (128): Expandimos de 57 a 128 para capturar combinaciones complejas de features.
HIDDEN_1 = 128
# Neuronas Capa 2 (64): Reducción progresiva (embudo) para sintetizar características.
HIDDEN_2 = 64
# Dropout (0.4): Apagamos 40% de neuronas para evitar que el modelo "memorice" (Overfitting).
DROPOUT_RATE = 0.4

def load_and_preprocess_data():
    """
    Carga el dataset Spambase de la UCI, separa features/targets
    y realiza el escalado estandar (StandardScaler).
    """
    print("Cargando dataset Spambase de UCI...")
    spambase = fetch_ucirepo(id=94)
    
    X = spambase.data.features.values
    y = spambase.data.targets.values
    
    # División 80% Entrenamiento - 20% Prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalado: CRUCIAL para que la red neuronal converja con inputs de distintas magnitudes
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Conversión a tensores
    inputs = {
        'X_train': torch.tensor(X_train).float(),
        'y_train': torch.tensor(y_train).float(),
        'X_test': torch.tensor(X_test).float(),
        'y_test': torch.tensor(y_test).float()
    }
    return inputs, y_test

class SpamClassifier(nn.Module):
    """
    Red Neuronal Densa para clasificación binaria.
    Arquitectura: Input(57) -> Relu(128) -> Dropout -> Relu(64) -> Sigmoid(1)
    """
    def __init__(self):
        super(SpamClassifier, self).__init__()
        self.layer1 = nn.Linear(57, HIDDEN_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.layer2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.layer3 = nn.Linear(HIDDEN_2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out) # Regularización activa durante entrenamiento
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.sigmoid(out) # Probabilidad entre 0 y 1
        return out

# --- BLOQUE PRINCIPAL ---
data, y_test_original = load_and_preprocess_data()
model = SpamClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Listas para guardar historial y graficar 
loss_history = []
accuracy_history = []

print(f"Iniciando entrenamiento por {EPOCHS} épocas...")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(data['X_train'])
    loss = criterion(outputs, data['y_train'])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Guardar métricas
    loss_history.append(loss.item())
    
    # Calcular accuracy temporal (solo para monitoreo)
    with torch.no_grad():
        predicted =  (outputs > 0.5).float()
        acc = accuracy_score(data['y_train'], predicted)
        accuracy_history.append(acc)

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Acc Train: {acc:.4f}')

# --- EVALUACIÓN FINAL ---
model.eval()
with torch.no_grad():
    test_outputs = model(data['X_test'])
    test_predicted = (test_outputs > 0.5).float()
    final_acc = accuracy_score(data['y_test'], test_predicted)
    
    print("-" * 30)
    print(f"Accuracy Final en Test Set: {final_acc*100:.2f}%")

# --- GENERACIÓN DE GRÁFICAS ---
plt.figure(figsize=(12, 5))

# Gráfica de Pérdida
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Training Loss', color='red')
plt.title('Curva de Pérdida (Loss) por Época')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

# Gráfica de Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label='Training Accuracy', color='blue')
plt.title('Curva de Accuracy por Época')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
# plt.savefig('resultados_entrenamiento.png') # Descomentar para guardar imagen