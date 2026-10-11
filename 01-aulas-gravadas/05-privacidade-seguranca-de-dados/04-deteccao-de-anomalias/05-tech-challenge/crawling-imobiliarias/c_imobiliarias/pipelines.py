import json
import os

from scrapy.exceptions import DropItem


class PropertiesJsonPipeline:
    """Acumula os itens e grava no formato {"version": 1, "properties": [...]}.

    Esse é o mesmo formato consumido por
    agente-sdr-imobiliario/apps/conversation-router/service/properties_catalog.py.
    """

    def open_spider(self, spider):
        self.items = []
        self.seen_ids = set()

    def process_item(self, item, spider):
        if item["id"] in self.seen_ids:
            raise DropItem(f"imóvel duplicado: {item['id']}")
        self.seen_ids.add(item["id"])
        self.items.append(dict(item))
        return item

    def close_spider(self, spider):
        payload = json.dumps({"version": 1, "properties": self.items}, ensure_ascii=False, indent=2)

        output_path = spider.settings.get("OUTPUT_PATH", "output/properties.json")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        spider.logger.info("Gravados %d imóveis em %s", len(self.items), output_path)

        target_path = spider.settings.get("POPULATE_TARGET_PATH")
        if target_path:
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
            spider.logger.info("Gravados %d imóveis em %s", len(self.items), target_path)
