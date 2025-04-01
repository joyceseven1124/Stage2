import pandas as pd
import numpy as np
import torch
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class TextClassifier:
    def __init__(self, csv_path, doc2vec_path, hidden_dim=128, learning_rate=0.001):
        print("初始化 TextClassifier...")
        self.csv_path = csv_path
        self.doc2vec_path = doc2vec_path
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("加載數據...")
        self.X, self.y, self.doc2vec_model = self.load_data()

        print("準備數據...")
        (self.X_tensor, 
        self.y_tensor, 
        self.encoder, 
        self.X_train, 
        self.X_test, 
        self.y_train, 
        self.y_test) = self.prepare_data(self.X, self.y)

        print("設置模型...")
        self.model, self.criterion, self.optimizer = self.setup_model()
        print("初始化完成！")

    def load_data(self):
        df = pd.read_csv(self.csv_path, header=None, dtype=str)
        X = [[word.strip() for word in str(line).split(",")] for line in df[0]]
        y = df[1]
        doc2vec_model = Doc2Vec.load(self.doc2vec_path)
        return X, y, doc2vec_model

    def prepare_data(self, X, y):
        X_vectors = [self.doc2vec_model.infer_vector(text) for text in X]
        X_numpy = np.array(X_vectors)
        X_tensor = torch.from_numpy(X_numpy).float().to(self.device)

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(self.device)

        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )

        return X_tensor, y_tensor, encoder, X_train, X_test, y_train, y_test

    def setup_model(self):
        input_dim = self.X_tensor.shape[1]
        # 所有唯一的類別標籤
        output_dim = len(self.encoder.classes_)
        
        model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        ).to(self.device)
        # 含了 softmax
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        return model, criterion, optimizer

    def train(self, num_epochs=10, batch_size=32):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == self.y_test).float().mean()
        print(f'Test Accuracy: {accuracy.item():.2%}')

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            vector = self.doc2vec_model.infer_vector(text.split())
            tensor = torch.tensor(vector).float().unsqueeze(0).to(self.device)
            output = self.model(tensor)
            _, predicted = torch.max(output, 1)
            return self.encoder.inverse_transform(predicted.cpu().numpy())[0]

print('模型建立')
classifier = TextClassifier('tokenized_titles.csv', 'doc2vec_model')
print('模型訓練')
classifier.train(num_epochs=25, batch_size=32)
print('模型預測')
classifier.evaluate()