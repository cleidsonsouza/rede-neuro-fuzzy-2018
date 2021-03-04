% ----------------------------------------------------------------------- %
%
% Autor: Cleidson dos Santos Souza | Data da última alteração: 16/12/2018
%
% Descrição da função: Rede Neuro Fuzzy - Anfis
%
% Protótipo: function [ys, emq, theta, c, sig, mu_A_x, mu_B_y] 
%                     = anfis(x, y, yt, nfp, nfpr, epocas, n) 
%
% Argumentos de entrada: 
%
%   ys     ==> Vetor contendo a saída da rede
%   emq    ==> Erro médio quadrático
%   theta  ==> Matriz contendo os parâmetros dos consequentes (p,q e r)
%   c      ==> Matriz contendo os centros das funções de pertinência (antecedentes) 
%   sig    ==> Matriz contendo os sigmas das funções de pertinência (antecedentes) 
%   mu_A_x ==> Matriz contendo os graus de pertinência das funções de pertinência
%              no universo de discuro x
%   mu_B_y ==> Matriz contendo os graus de pertinência das funções de pertinência
%              no universo de discuro x
%
% Argumentos de saida:
%
%   x      ==> Pares de entrada (Padrões por linha)
%   yt     ==> Saídas desejadas de treinamento
%   nfp    ==> Número de funções de pertinência
%   nfpr   ==> Número de funções de pertinência usadas em cada regra
%   epocas ==> Número de regras de treinamento
%   n      ==> Taxa de aprendizagem
%
% ----------------------------------------------------------------------- %

function [ys, emq, theta, c, sig, mu_A_x, mu_B_y] = anfis(x, yt, nfp, nfpr, epocas, n) 
		
	% Obtenção do número de amostras (que é igual ao número de pontos) e de dimensões
	[numAmostras numDimensoes] = size(x);	
	  
	% Número de regras
	numRegras = nfp^2;
	
	% Inicialização dos parâmetros dos antecedentes (funções de pertinência)
	for i=1 : numRegras
		for j=1 : nfpr
			c(i,j)   = rand; 
			sig(i,j) = rand;
		end	
	end
	
	% Inicialização dos parâmetros dos consequentes
	for i=1 : numRegras
		for j=1 : numDimensoes+1
			theta(i,j) = rand; % [p; q; r]
		end		
	end
	
	% Definição de variáveis para cálculo dos mínimos quadrados	
	P = 1000000 * eye(numDimensoes+1);
		
	% Execução do treinamento - até o número de épocas					
	for i=1 : epocas
			
		for j=1 : numAmostras
			
			% ---------------------------------- Forward ---------------------------------- %
			
			% Camada 1 - Cálculo dos graus de pertinência
			for k=1 : nfpr
				for l=1 : numRegras
					mu(k,l) = gaussmf(x(j,k), [c(l,k) sig(l,k)]);					
				end				
			end
			
			% Armazenamento das pertinências para plotagem (Não auto-ajustável)
			mu_A_x(j,:) = mu(1,:);
			mu_B_y(j,:) = mu(2,:);
						
			% Camada 2 - Cálculo dos graus de disparo das regras				
			for k=1 : numRegras				
				w(k) = min(mu(:,k)) + 1e-6;			
			end
						
			% Camada 3 - Cálculo das saídas das regras			
			for k=1 : numRegras				
				f(k) = sum(theta(k,:).*[x(j,:)'; 1]');
			end
						
			% Camada 4 - Cálculo da saída do sistema		
			ys(j) = sum(w.*f) / sum(w);
										
			% Ajuste dos parâmetros dos consequentes (com mínimos quadrados)					
			a = [x(j,:)'; 1];			
			for k=1 : numRegras				
				theta(k,:) = (theta(k,:)' + (P - (P * a * a' * P)/(1 + a' * P * a)) * a * (yt(j)' - a' * theta(k,:)'))';				
			end
			
			% Transposição do vetor a 
			a = a';
			
			% Cálculo do erro quadrático		
			eq(j) = (yt(j) - ys(j))^2;
			
			% ---------------------------------- Backward ---------------------------------- %		
			
			% Evita o último ajuste para sincronizar os valores dos antecedentes com o dos consequentes
			if i ~= epocas
			
				% Derivada do erro em relação a saída do sistema		
				dedys = -2 * (yt(j) - ys(j));
						
				for k=1 : numRegras
				
					% Derivada da saída do sistema (ys) em relação ao grau de disparo (w)
					dysdw(k) = (f(k) - ys(j))/sum(w);		
					
					for l=1 : nfpr
						
						% Demais derivadas
						dwdc(k,l) = w(k) * (a(1,l) - c(k,l)/(sig(k,l)^2));
						dwds(k,l) = w(k) * (a(1,l) - c(k,l)/(sig(k,l)^3));
						
						dedc(k,l) = dedys + dysdw(k) * dwdc(k,l);
						deds(k,l) = dedys + dysdw(k) * dwds(k,l);
						
						% Ajuste dos centros e sigmas das funções de pertinência (antecedentes)
						c(k,l)   = c(k,l)   + (n * dedc(k,l));
						sig(k,l) = sig(k,l) + (n * deds(k,l));
					end
				end
			end				
		end
		
		% Cálculo do erro médio quadrático		
		emq(i) = sum(eq)/numAmostras;
		
		% Critério de parada
		if emq(i) <= 0		
			break
		end
		
	end	

end