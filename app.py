import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


#Define a conversão de imagem para tensor
transform = transforms.ToTensor()

#Carrega apartir de treino do dataset;
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
#Cria um buffer para pegar os dados por parte;
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

Valset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
valloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# Seria para a gente verifcar se estamos visualizando a base dados de forma correta;
dataiter = iter(trainloader)
 #Update para nova utilização do metodo em pyTorch mudando o parametro next(), que era uma utilização desatualizad para o parametro correto __next__();
imagens, etiqueta = dataiter.__next__()
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r');

# Vamos verificar o tamanho do nosso tensor na imagem ou seu tamanho;

print(imagens[0].shape)#Para verificar as dimensões do tensor de cada imagem.
print(etiqueta[0].shape)#Para verificar a dimensão de cada etiqueta;

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)# camada de entrada, 784 neuronios que se ligam a 128
        self.linear2 = nn.Linear(128, 64) # camada oculta, 128 neuronios que se ligam a 64
        self.linear3 = nn.Linear(64, 10)# camada interna 2, 64 neuronios que se liga a 10
        # para a camada de saida não e necessario definir nada pois só precisamos pegar o output da camada interna 2

    def forward(self, X):
        X = F.relu(self.linear1(X)) # função de ativação de camada de entrada para a camada interna 1
        X = F.relu(self.linear2(X)) # função de ativação da camada interna 1 para a camada interna 2
        X = self.linear3(X) # função de ativação da camada interna 2 para a cama da de saida, nesse caso foi f(x) = x
        return F.log_softmax(X, dim=1) # dados utilizados para calcular a perda
    

#Estrutura de Treinamento da rede

def treino (modelo, trainloader, device):
    otimizador = optim.SGD(modelo.parameters(), lr=0.03, momentum=0.5) # Define apolitica de atualização do pesos.
    inicio = time() # Para sabermos quanto tempo levou o treinamento;
    criterio = nn.NLLLoss() # Define o criteiro para calcular a perda;
    EPOCHS = 10 # numero de EPOCHS que o algoritmo rodara;
    modelo.train() # ativa o modelo de treinamento;

    for epoch in range(EPOCHS):
        perda_acumulada = 0 # Variavel que armazena a perda total da epoca

        for imagens, etiquetas in trainloader:
            imagens = imagens.view(imagens.shape[0], -1) # Transforma a imagem em um vetor;
            otimizador.zero_grad() # zerar os gradientes

            output = modelo(imagens.to(device)) # Coloca os dados no modelo
            perda_instantanea = criterio(output, etiquetas.to(device)) # Calcula a perda dessa iteração

            perda_instantanea.backward() # realiza o backpropagation
            otimizador.step() # realiza a atualização dos pesos

            perda_acumulada += perda_instantanea.item() #atualização da perda acumulativa.

        else:
            print("Época {} - Perda resultante: {}".format(epoch+1, perda_acumulada/len(trainloader)))
            print("\nTempo de treino (em minutos) = ",(time()-inicio)/60)

#Modelo de validação

def validacao(modelo, valloader, device):
  conta_corretas, conta_todas = 0, 0
  for imagens,etiquetas in valloader:
      for i in range(len(etiquetas)):
          img = imagens[i].view(1, 784)
          # desativa o autograd para acelerar a validação. Grafos computacionais dinamicos tem um custo alto de processamento
          with torch.no_grad():
              logps = modelo(img.to(device)) # output do modelo em escala logaritmica

          ps = torch.exp(logps) # converte output para escala normal( lembrando que é um tensor)
          probab = list(ps.cpu().numpy()[0])
          etiqueta_pred = probab.index(max(probab)) # converte o tensor em um numero, no caso número que o modelo previu
          etiqueta_certa = etiquetas.numpy()[i]
          if(etiqueta_certa == etiqueta_pred): # compara a previsão com o valor correto
              conta_corretas += 1
          conta_todas += 1

          print("Total de imagens testadas = ", conta_todas)
  print("\nPrecisão do modelo = {}%".format(conta_corretas*100/conta_todas))


# Inicio do modelo

modelo = Modelo()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)