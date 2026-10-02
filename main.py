import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Neuronales Netz definieren (Muss exakt gleich aussehen wie beim Training!)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Test-Daten laden (Bilder, die das Netz NICHT beim Training gesehen hat)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Modell laden und in den "Evaluierungs-Modus" setzen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
model.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
model.eval() # Wichtig: Sagt PyTorch, dass wir nur Vorhersagen machen wollen

# 4. Genauigkeit berechnen
print(f"Prüfe das Modell auf {len(test_dataset)} unbekannten Bildern...")
correct = 0
total = 0

with torch.no_grad(): # Deaktiviert das Lernen während des Tests
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # Wähle die Zahl mit der höchsten Wahrscheinlichkeit
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Genauigkeit auf den Testdaten: {accuracy:.2f}%")

# 5. Ein Beispiel-Bild mit Vorhersage anzeigen
img, label = test_dataset[0] # Nimm das allererste Testbild
img_tensor = img.unsqueeze(0).to(device) # Füge eine Batch-Dimension hinzu

with torch.no_grad():
    output = model(img_tensor)
    _, prediction = torch.max(output, 1)

plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Wahrheit: {label} | Vorhersage: {prediction.item()}")
plt.axis('off')
plt.show()