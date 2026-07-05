from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import boto3

def create_investment_pdf(file_path):
   c = canvas.Canvas(file_path, pagesize=letter)
  
   c.setFont("Helvetica-Bold", 16)
   c.drawString(100, 750, "Relatório de Investimentos")
  
   c.setFont("Helvetica", 12)
   investments = [
       {"Investidor": "João Silva", "Tipo": "Ações", "Valor": "R$ 10.000", "Data": "01/01/2024"},
       {"Investidor": "Maria Oliveira", "Tipo": "Fundos Imobiliários", "Valor": "R$ 5.000", "Data": "15/02/2024"},
       {"Investidor": "Carlos Souza", "Tipo": "Tesouro Direto", "Valor": "R$ 8.000", "Data": "20/03/2024"},
   ]
  
   y = 720
   for inv in investments:
       c.drawString(100, y, f"Investidor: {inv['Investidor']}")
       c.drawString(100, y-15, f"Tipo de Investimento: {inv['Tipo']}")
       c.drawString(100, y-30, f"Valor Investido: {inv['Valor']}")
       c.drawString(100, y-45, f"Data do Investimento: {inv['Data']}")
       y -= 75  #
   
   c.save()


pdf_file_path = "relatorio_investimentos.pdf"

create_investment_pdf(pdf_file_path)

s3 = boto3.client('s3')
bucket_name = 'psobral-aulafiap-v2'
object_name = pdf_file_path 

s3.upload_file(pdf_file_path, bucket_name, object_name)

print(f"PDF {pdf_file_path} carregado com sucesso para s3://{bucket_name}/{object_name}")