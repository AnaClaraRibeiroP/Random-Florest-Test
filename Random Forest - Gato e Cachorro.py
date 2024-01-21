# Importando as bibliotecas necessárias 
# Pandas para manipular dados
# sklearn.ensemble para randomforest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Criando um conjunto de dados fictício
dados = {'Tamanho': [20, 15, 18, 25, 22, 30, 28, 16, 24, 19],
         'Peso': [5, 3, 4, 6, 5, 7, 8, 3, 6, 4],
         'Animal': ['Gato', 'Gato', 'Gato', 'Cachorro', 'Cachorro', 'Cachorro', 'Cachorro', 'Gato', 'Cachorro', 'Gato']}
df = pd.DataFrame(dados)

# Separando os recursos (X) e os rótulos (y)
X = df[['Tamanho', 'Peso']]
y = df['Animal']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de Floresta Aleatória
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Fazendo previsões de probabilidade no conjunto de teste
probabilidades = modelo_rf.predict_proba(X_test)


# Especificando um novo exemplo para fazer uma previsão
novo_exemplo = pd.DataFrame({'Tamanho': [24], 'Peso': [7]})

# Fazendo uma previsão de probabilidade para o novo exemplo
probabilidades_novo_exemplo = modelo_rf.predict_proba(novo_exemplo)

# Exibindo as probabilidades para o novo exemplo
print(f'\nProbabilidades para o novo exemplo (Gato, Cachorro): {probabilidades_novo_exemplo[0][1] * 100:.2f}%, {probabilidades_novo_exemplo[0][0] * 100:.2f}%')
