# Regras Descobertas

> Estas regras são candidatas a limites rígidos e aguardam confirmação na entrevista. Padrões já cobertos pelos defaults organizacionais não são repetidos aqui.

## Mandated

- `ALWAYS` integrar via pull request a partir de branch por feature, partindo de `feature/01-aulas-gravadas/05-privacidade-seguranca-de-dados`.
- `ALWAYS` construir a fatia fina de ponta a ponta (walking skeleton) antes das features reais.
- `ALWAYS` usar scripts `start.sh`/`stop.sh` para build, testes e `terraform apply` / `terraform destroy` no deploy manual.

## Forbidden

- `NEVER` deployar sem passar pelos scripts `start.sh`/`stop.sh`.
- `NEVER` tratar cobertura de testes como gate bloqueante neste hackathon.

Restrições específicas do projeto, do domínio imobiliário ou do uso de dados pessoais (LGPD) precisam ser confirmadas antes de serem registradas como limites rígidos adicionais.
