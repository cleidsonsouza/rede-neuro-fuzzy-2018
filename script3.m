% ----------------------------------------------------------------------- %
%                             Apresentação                                %
% ----------------------------------------------------------------------- %
%
% Universidade Estadual de Montes Claros - Unimontes
%
% Programa de Pós Graduação em Modelagem Computacional e Sistemas (PPGMCS)
%
% Disciplina: Sistemas Nebulosos
%
% Trabalho Computacional 3 - Redes Neurofuzzy
% 
% Autor: Cleidson dos Santos Souza 
% 
% Testado no Octave 4.4.1
%
% Data: 16/12/2018
%
% ----------------------------------------------------------------------- %
%                         Limpeza de tela da IDE                          %
% ----------------------------------------------------------------------- %

clear all
close all
clc

% ----------------------------------------------------------------------- %
%                       Inicialização de variáveis                        %
% ----------------------------------------------------------------------- %

% Taxa de aprendizagem
n = 1e-6;

% Número de épocas de treinamento
epocas = 25; 

% Número de funções de pertinência   
nfp = 4;

% Número de funções de pertinência por regra      
nfpr = 2; 

% Definição dos limites do universo de discurso
xmin = -10; 
xmax = 10; 
pts = 121;

% Universo de disccurso
x = linspace(xmin, xmax, pts);
y = linspace(xmin, xmax, pts);

% Gera a (saída da) função sinc - dados para treinamento
yt = (sin(x).*sin(y))./(x.*y);

% Eliminação de valor zero
index = find(isnan(yt)==1);
yt(index) = 1;

% ----------------------------------------------------------------------- %
%                        Treinamento da rede anfis                        %
% ----------------------------------------------------------------------- %

% Chamada à função anfis
[ys, emq, theta, c, sig, mu_A_x, mu_B_y] = anfis([x' y'], yt, nfp, nfpr, epocas, n); 

% Imprime o erro médio quadrático de treinamento
emqTreinamento = emq(epocas)

% Plotagem das saídas desejadas e obtidas
%figure
plot(yt)
hold on
plot(ys)
title('Saida da rede Anfis')
legend('Sinc','Sinc - Anfis')

% Plotagem do erro quadrático médio
figure
plot(emq)
title('Erro medio quadratico por epoca de treinamento')
xlabel('Epoca')
ylabel('Erro medio quadratico')

% Plotagem das 10 primeiras funções de pertinência(de um total de 16) 
% obtidas pela rede anfis
figure
i=1;
for j=1:2:20 
	subplot(10,2,j+0)
	plot(mu_A_x(:,i))
	subplot(10,2,j+1)
	plot(mu_B_y(:,i))
	i = i + 1;
end	

% ----------------------------------------------------------------------- %
%                         Validacao da rede anfis                         %
% ----------------------------------------------------------------------- %

% Definição dos limites do universo de discurso
xmin = -10; 
xmax = 10; 
pts = 500;

% Universo de disccurso
x = linspace(xmin, xmax, pts);
y = linspace(xmin, xmax, pts);

% Gera a (saída da) função sinc - dados para treinamento
yt = (sin(x).*sin(y))./(x.*y);

% Eliminação de valor zero
index = find(isnan(yt)==1);
yt(index) = 1;

% Chamada à função de validação da anfis
[ysv, eq, emq,  mu_A_x2, mu_B_y2] = anfis_validacao(theta, c, sig, [x' y'], yt, nfp, nfpr); 

% Plotagem das saídas desejadas e obtidas
figure
plot(yt)
hold on
plot(ysv)
title('Saida da rede Anfis')
legend('Sinc','Sinc - Anfis')

% Plotagem do erro quadrático médio
figure
plot(eq)
title('Erro medio quadratico por epoca de treinamento')
xlabel('Epoca')
ylabel('Erro medio quadratico')

% Imprime o erro médio quadrático de validação
emqValidacao = emq

% Plotagem das 10 primeiras funções de pertinência(de um total de 16) 
figure
i=1;
for j=1:2:20 
	subplot(10,2,j+0)
	plot(mu_A_x2(:,i))
	subplot(10,2,j+1)
	plot(mu_B_y2(:,i))
	i = i + 1;
end	