import os
import cv2
import pytesseract

# Carregar a imagem (usando caminho relativo ao script)
script_dir = os.path.dirname(os.path.abspath(__file__))
imagem = cv2.imread(os.path.join(script_dir, 'imagem-1.jpeg'))

# converter para grayscale
# imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Exibir imagem grayscale
# cv2.imshow('Imagem Gray', imagem_gray)

# Redimensionar imagem
imagem_redimensionada = cv2.resize(imagem, (300, 300))

# Exibir imagem redimensionada
cv2.imshow('Imagem Redimensionada', imagem_redimensionada)

# Exibir imagem
# cv2.imshow('Imagem', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

# aplicando o Tesseract OCR
# texto = pytesseract.image_to_string(imagem, lang='por')
# print(texto)

# aplicando o Tesseract grayscale
# texto = pytesseract.image_to_string(imagem_gray, lang='por')
# print(texto)

# aplicando o Tesseract redimensionada
texto = pytesseract.image_to_string(imagem_redimensionada, lang='por')
print(texto)