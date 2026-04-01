import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. LOAD DATA ---
train_embeddings = torch.load("train_embeddings.pt")
val_embeddings = torch.load("val_embeddings.pt")
test_embeddings = torch.load("test_embeddings.pt")
train_y = torch.load("train_labels.pt")
val_y = torch.load("val_labels.pt")
test_y = torch.load("test_labels.pt")

# --- 2. GRAPH CONSTRUCTION ---
def create_knn_edges(x, k):
    similarity = torch.mm(x, x.t())
    similarity.fill_diagonal_(-float('inf'))
    _, top_k_indices = torch.topk(similarity, k=k, dim=1)
    src = torch.arange(x.size(0), device=x.device).repeat_interleave(k)
    dst = top_k_indices.flatten()
    return torch.stack([src, dst], dim=0)

train_emb_norm = F.normalize(train_embeddings.float(), p=2, dim=1)
val_emb_norm = F.normalize(val_embeddings.float(), p=2, dim=1)
test_emb_norm = F.normalize(test_embeddings.float(), p=2, dim=1)

k_edges = 5 
train_edges = create_knn_edges(train_emb_norm, k_edges)
val_edges = create_knn_edges(val_emb_norm, k_edges)
test_edges = create_knn_edges(test_emb_norm, k_edges)

train_data = Data(x=train_embeddings.float(), edge_index=train_edges, y=train_y)
val_data   = Data(x=val_embeddings.float(), edge_index=val_edges, y=val_y)
test_data  = Data(x=test_embeddings.float(), edge_index=test_edges, y=test_y)

train_data.edge_attr = torch.ones(train_data.edge_index.size(1), 1)
val_data.edge_attr = torch.ones(val_data.edge_index.size(1), 1)
test_data.edge_attr = torch.ones(test_data.edge_index.size(1), 1)

# --- 3. MODEL DEFINITION ---
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.6, edge_dim=1)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=0.6, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# --- 4. TRAINING SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gat_model = GAT(in_dim=768, hidden_dim=256, out_dim=2, heads=8).to(device)
optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = FocalLoss(alpha=1, gamma=2).to(device)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

def train():
    gat_model.train()
    optimizer.zero_grad()
    out = gat_model(train_data.x, train_data.edge_index, train_data.edge_attr)
    loss = criterion(out, train_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(data):
    gat_model.eval()
    with torch.no_grad():
        out = gat_model(data.x, data.edge_index, data.edge_attr)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
    return acc, f1

# --- 5. TRAINING LOOP ---
best_val_f1 = 0
best_epoch = 0

for epoch in range(1, 151):
    loss = train()
    if epoch % 10 == 0:
        val_acc, val_f1 = evaluate(val_data)
        print(f'Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(gat_model.state_dict(), 'sota_model.pth')

print(f"Training Complete. Best model was from Epoch {best_epoch}.")

# --- 6. FINAL EVALUATION & VISUALIZATION ---
gat_model.load_state_dict(torch.load('sota_model.pth'))
test_acc, test_f1 = evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f} | Test F1 Score: {test_f1:.4f}")

gat_model.eval()
with torch.no_grad():
    try:
        out = gat_model(test_data.x, test_data.edge_index)
    except TypeError:
        out = gat_model(test_data.x, test_data.edge_index, test_data.edge_attr)
        
    preds = out.argmax(dim=1).cpu().numpy()
    labels = test_data.y.cpu().numpy()

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real News', 'Fake News'], 
            yticklabels=['Real News', 'Fake News'],
            annot_kws={"size": 16})
plt.title('Final Model Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.savefig('confusion_matrix.png')