import sys
import pygame
import math

# 1. Inicializa o Pygame
pygame.init()

# 2. Configura a janela
LARGURA, ALTURA = 600, 400
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Teste Simples - Pygame")

# Relógio para controlar a taxa de quadros (FPS)
relogio = pygame.time.Clock()

# Variáveis do nosso "estado" (simulando algo que muda)
cor_fundo = (30, 30, 30)       # RGB: Cinza escuro
posicao_circulo = (300, 200)   # (X, Y)
raio = 40
cor_circulo = (0, 200, 100)    # RGB: Verde

# Loop principal da janela
rodando = True
contador = 0
while rodando:
    # --- Trata os eventos (fechar a janela) ---
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            rodando = False

    # --- Desenho na tela ---
    # Limpa a tela pintando com a cor de fundo
    tela.fill(cor_fundo)

    # Desenha uma linha (ex: representando uma conexão/peso)
    # pygame.draw.line(superfície, cor, ponto_inicio, ponto_fim, espessura)
    pygame.draw.line(tela, (200, 200, 200), (100, 200), (500, 200), 2)

    # Desenha um círculo (ex: representando um neurônio)
    # pygame.draw.circle(superfície, cor, centro, raio)
    pygame.draw.circle(tela, cor_circulo, posicao_circulo, raio*math.sin(contador*0.05))

    # --- Atualiza a tela com o que foi desenhado ---
    pygame.display.flip()

    # Mantém o loop rodando a no máximo 60 FPS
    relogio.tick(60)

    if contador == 60: contador = 0
    else: contador += 1

# Encerra o Pygame ao sair do loop
pygame.quit()
sys.exit()