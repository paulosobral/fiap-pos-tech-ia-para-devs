import os
import cv2

# Carregar a imagem (usando caminho relativo ao script)
script_dir = os.path.dirname(os.path.abspath(__file__))
imagem = cv2.imread(os.path.join(script_dir, 'face.jpeg'))

# Criar janela e exibir a imagem
cv2.namedWindow('Imagem', cv2.WINDOW_NORMAL)
cv2.imshow('Imagem', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()