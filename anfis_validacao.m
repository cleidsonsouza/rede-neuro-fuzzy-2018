% ----------------------------------------------------------------------- %
%
% Autor: Cleidson dos Santos Souza | Data da última alteração: 16/12/2018
%
% Descrição da função: Rede Neuro Fuzzy - Anfis
%
% Protótipo: function [ys, eq, emq] = 
%            anfis_validacao(theta, c, sig, x, y, yt, nfp) 
%
% Argumentos de entrada: 
%
%   ys     ==> Vetor contendo a saída da rede
%   eq     ==> Erro quadrático
%   emq    ==> Erro médio quadrático
%   mu_A_x ==> Matriz contendo os graus de pertinência das funções de pertinência
%              no universo de discuro x
%   mu_B_y ==> Matriz contendo os graus de pertinência das funções de pertinência
%              no universo de discuro x
%
% Argumentos de saida:
%
%   theta  ==> Matriz contendo os parâmetros dos consequentes (p,q e r)
%   c      ==> Matriz contendo os centros das funções de pertinência (antecedentes) 
%   sig    ==> Matriz contendo os sigmas das funções de pertinência (antecedentes) 
%   x      ==> Pares de entrada (Padrões por linha)
%   yt     ==> Saídas desejadas de treinamento
%   nfp    ==> Número de funções de pertinência
%
% ----------------------------------------------------------------------- %

function [ys, eq, emq, mu_A_x, mu_B_y] = anfis_validacao(theta, c, sig, x, yt, nfp, nfpr) 
		
	% Número de amostras (que é igual ao número de pontos)
	numAmostras = size(x,1);	
	  
	% Número de regras
	numRegras = nfp^2;
	
	for j=1 : numAmostras
	
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
			
		% Cálculo do erro quadrático		
		eq(j) = (yt(j) - ys(j))^2;
		
	end	
	
	% Cálculo do erro médio quadrático		
	emq = sum(eq)/numAmostras;

end