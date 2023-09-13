import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 

# Lendo os dados e pulando a primeira linha (cabeçalho)
data = pd.read_csv('Medical.csv', header=1)

# Dividindo o dataset em previsores (X) e classe alvo (y)
X = data.iloc[:, :-1]  # Todas as colunas, exceto a última
y = data.iloc[:, -1]   # A última coluna

# Dividindo o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


######################## Naive Bayes

# Criando o modelo Naive Bayes
naive_bayes_model = GaussianNB()

# Treinando o modelo com os dados de treinamento
naive_bayes_model.fit(X_train, y_train)

# Fazendo previsões com o modelo nos dados de teste
y_pred = naive_bayes_model.predict(X_test)

# Calculando a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo Naive Bayes: {accuracy}')

############## Decision Tree

# Criando o modelo Decision Tree
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Treinando o modelo com os dados de treinamento
decision_tree_model.fit(X_train, y_train)

# Fazendo previsões com o modelo nos dados de teste
y_pred = decision_tree_model.predict(X_test)

# Calculando a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo Decision Tree: {accuracy}')

############## Random Forest

# Criando o modelo Random Forest
random_forest_model = RandomForestClassifier(random_state=42)

# Treinando o modelo com os dados de treinamento
random_forest_model.fit(X_train, y_train)

# Fazendo previsões com o modelo nos dados de teste
y_pred = random_forest_model.predict(X_test)

# Calculando a acurácia do modelo Random Forest
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo Random Forest: {accuracy}')

############## K-Nearest Neighbors (KNN)

# Criando o modelo KNN com, por exemplo, 5 vizinhos
knn_model = KNeighborsClassifier(n_neighbors=5)  # Você pode ajustar o número de vizinhos conforme necessário

# Treinando o modelo com os dados de treinamento
knn_model.fit(X_train, y_train)

# Fazendo previsões com o modelo nos dados de teste
y_pred = knn_model.predict(X_test)

# Calculando a acurácia do modelo KNN
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo K-Nearest Neighbors (KNN): {accuracy}')

############## Support Vector Machine (SVM)

# Criando o modelo SVM
svm_model = SVC(kernel='linear')  # Você pode escolher um kernel (linear, rbf, etc.) conforme necessário

# Treinando o modelo com os dados de treinamento
svm_model.fit(X_train, y_train)

# Fazendo previsões com o modelo nos dados de teste
y_pred = svm_model.predict(X_test)

# Calculando a acurácia do modelo SVM
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo Support Vector Machine (SVM): {accuracy}')

############## Rede Neural Artificial (MLP - Multi-Layer Perceptron)


# Criando o modelo MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)  
# hidden_layer_sizes: Especifique o número de neurônios em cada camada oculta
# max_iter: Especifique o número máximo de iterações de treinamento

# Treinando o modelo com os dados de treinamento
mlp_model.fit(X_train, y_train)

# Fazendo previsões com o modelo nos dados de teste
y_pred = mlp_model.predict(X_test)

# Calculando a acurácia do modelo de Rede Neural
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Rede Neural: {accuracy}')




