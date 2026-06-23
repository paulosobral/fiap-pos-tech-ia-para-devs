import boto3

# Inicializando o cliente Comprehend com uma região suportada
comprehend = boto3.client('comprehend', region_name='us-east-1')  # Exemplo: us-east-1

# Carregando o texto extraído do arquivo
with open('texto_extraido.txt', 'r') as file:
   text = file.read()

# Detectando o sentimento do texto
sentiment_response = comprehend.detect_sentiment(
   Text=text,
   LanguageCode='pt'  # Substitua pelo código de idioma apropriado
)

sentimento = sentiment_response['Sentiment']
print(f"Sentimento do texto: {sentimento}")

# Extraindo entidades do texto
entities_response = comprehend.detect_entities(
   Text=text,
   LanguageCode='pt'  # Substitua pelo código de idioma apropriado
)

print("Entidades encontradas:")
for entity in entities_response['Entities']:
   print(f" - {entity['Text']}: {entity['Type']}")

# Extraindo frases-chave do texto
key_phrases_response = comprehend.detect_key_phrases(
   Text=text,
   LanguageCode='pt'  # Substitua pelo código de idioma apropriado
)

print("Frases-chave encontradas:")
for phrase in key_phrases_response['KeyPhrases']:
   print(f" - {phrase['Text']}")
