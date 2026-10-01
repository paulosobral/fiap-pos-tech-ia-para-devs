BOT_NAME = "crawling_imobiliarias"

SPIDER_MODULES = ["c_imobiliarias.spiders"]
NEWSPIDER_MODULE = "c_imobiliarias.spiders"

# Identifica o bot e dá um contato de forma transparente (boa prática de scraping).
USER_AGENT = (
    "crawling-imobiliarias-tech-challenge/1.0 "
    "(+contato: paulo@paulosobral.com.br; uso educacional/FIAP)"
)

ROBOTSTXT_OBEY = True

# Rate limit educado: é o site real de uma pequena imobiliária, não um load test.
DOWNLOAD_DELAY = 1.5
RANDOMIZE_DOWNLOAD_DELAY = True
CONCURRENT_REQUESTS = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 2

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600

ITEM_PIPELINES = {
    "c_imobiliarias.pipelines.PropertiesJsonPipeline": 300,
}

# Caminho do arquivo gerado por este projeto (sempre escrito).
OUTPUT_PATH = "output/properties.json"

# Caminho opcional: se definido (via -s POPULATE_TARGET_PATH=... ou env var),
# o pipeline também grava uma cópia direta nesse caminho — útil para popular
# apps/conversation-router/data/properties.json do agente-sdr-imobiliario.
POPULATE_TARGET_PATH = None

REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

LOG_LEVEL = "INFO"
