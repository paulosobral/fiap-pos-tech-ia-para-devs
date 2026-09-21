# Memory — Environment Provisioning (operation)

> Working memory do estágio `environment-provisioning`. Consolidado antes do approval gate.

## Interpretations

- O estágio pede "Provision and Validate" — no escopo acordado (decisão do gate do `ci-pipeline`), o provisionamento real (`terraform apply`) é do `deployment-execution`; aqui valida-se o **design e as pré-condições** (inventário, credenciais, segredos, rede, quotas) sem tocar na nuvem.
- "Are all environments provisioned per Infra Design?" — o infrastructure-design está skipped; a referência usada foi o inventário derivado do ci-config + NFR9.x + design u1–u7.

## Deviations

- `consumes: infrastructure-specification` (required) não existe — estágio `infrastructure-design` está `[S]kipped`; o inventário foi derivado do cd-config.md, ci-config.md e dos NFRs, que documentam a mesma infra no escopo atual.
- Verificações V1–V6 são definidas mas não executadas nesta rodada (execução real no `deployment-execution`) — o relatório marca estado "definido/executará".

## Tradeoffs

- Região `us-east-1` vs `sa-east-1`: preço/latência global vs proximidade BR — us-east-1 venceu no escopo POC (custo domina).
- Sem VPC custom: zero NAT Gateway vs isolamento de rede — aceito (LGPD coberta por criptografia de PII + Secrets Manager, não por rede).

## Open questions

- Nenhuma aberta relevante — Q1–Q4 propostas aguardam confirmação no gate.