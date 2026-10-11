# CI Pipeline Questions

> Respostas do contexto afirmado (`aidlc/spaces/default/memory/team.md`, `project.md`): a equipe **não possui esteira CI/CD** — deploy manual via `start.sh`/`stop.sh` (build → testes → `terraform apply`); integração via PR por feature a partir de `feature/...`; cobertura **não** é gate bloqueante neste hackathon.

## Q1 — Ferramenta de CI em uso
Nenhuma atualmente. Proposta para este POC (repo em GitHub: `paulosobral/fiap-pos-tech-ia-para-devs`):
- A. **GitHub Actions** (Recommended) — workflow YAML versionado, zero infra, gratuito para repo privado em scale reduzido
- B. Sem CI — apenas documentar os gates e rodar local
- C. AWS CodePipeline + CodeBuild — exigiria conta/infra fora do escopo atual

[Answer]: A

> **Correção (2026-10-10):** a resposta A (GitHub Actions) ficou desatualizada. A decisão afirmada no gate e seguida até hoje é a alternativa B: sem CI externo, a esteira são `start.sh`/`stop.sh`, e não há workflow no repositório. Vale a B; ver `ci-config.md`. Confirmado pelo responsável em 2026-10-10 ("sem CI externo").

## Q2 — Quality gates obrigatórios antes do merge
Proposta respeitando `project.md` (cobertura NÃO bloqueante no hackathon) + contrato de testes (suítes verdes):
- A. **Testes 100% pass** (`pytest` por app, unidade + integração) — bloqueante; cobertura registrada como informativa (Recommended)
- B. Testes + cobertura ≥80% bloqueante — contradiz o `NEVER` affirmado; não recomendado
- C. Sem gates automáticos

[Answer]: A

> **Observação (2026-10-10):** o `start.sh` hoje também aborta com cobertura < 80% (`--cov-fail-under=80`), o que contradiz "cobertura informativa" e o `NEVER` de `project.md`. Decidido em 2026-10-10 (opção A): o bloqueio de 80% fica; ver `quality-gates.md`.

## Q3 — Triggers e estratégia de branch
`team.md` afirma: branch por feature → PR → merge. Proposta:
- A. **PR (target `main`) + push em `main`** (Recommended) — CI roda no PR e no merge
- B. Só push em `main`
- C. Manual (workflow_dispatch)

[Answer]: A

## Q4 — Repositórios de artefatos
- A. **Nenhum** (Recommended) — POC Python/Node; o pacote deployável nasce do `start.sh` no destino (NFR9.2)
- B. S3 zip do bundle
- C. ECR/container

[Answer]: A

---

## Consolidated Summary Confirmation

Summary consolidado do estágio CI Pipeline:
1. Sem CI externo (GitHub só repositório); esteira local `start.sh`/`stop.sh` conforme PRD
2. Pipeline (`start.sh`, atualizado em 2026-10-10): deps → `compileall` → pytest das 8 aplicações (1176 passed, cobertura 83,3–100%) + guarda de índices DynamoDB → gate de qualidade com LLM real (62/62) → build `dist/<app>.zip` e imagens `podman` → `terraform apply` em 2 passos → smoke
3. Gates bloqueantes: build, testes 100%, guarda de índices, gate com LLM real, smoke pós-deploy; cobertura ≥ 80% também bloqueia (decisão A, 2026-10-10)
4. IaC: um `.tf` por serviço; só as Lambdas usam módulo (`terraform-aws-modules/lambda/aws`), o restante (API Gateway, DynamoDB, SQS, ECS Fargate etc.) são recursos nativos
5. Boundary Construction→Operation: PASS com ressalvas (ver `verification/phase-check-construction.md`)

- Looks correct
- Request changes

[Answer]: Looks correct
