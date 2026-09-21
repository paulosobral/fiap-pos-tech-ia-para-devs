# CI Pipeline Questions

> Respostas do contexto afirmado (`aidlc/spaces/default/memory/team.md`, `project.md`): a equipe **não possui esteira CI/CD** — deploy manual via `start.sh`/`stop.sh` (build → testes → `terraform apply`); integração via PR por feature a partir de `feature/...`; cobertura **não** é gate bloqueante neste hackathon.

## Q1 — Ferramenta de CI em uso
Nenhuma atualmente. Proposta para este POC (repo em GitHub: `paulosobral/fiap-pos-tech-ia-para-devs`):
- A. **GitHub Actions** (Recommended) — workflow YAML versionado, zero infra, gratuito para repo privado em scale reduzido
- B. Sem CI — apenas documentar os gates e rodar local
- C. AWS CodePipeline + CodeBuild — exigiria conta/infra fora do escopo atual

[Answer]: A

## Q2 — Quality gates obrigatórios antes do merge
Proposta respeitando `project.md` (cobertura NÃO bloqueante no hackathon) + contrato de testes (suítes verdes):
- A. **Testes 100% pass** (`pytest` por app, unidade + integração) — bloqueante; cobertura registrada como informativa (Recommended)
- B. Testes + cobertura ≥80% bloqueante — contradiz o `NEVER` affirmado; não recomendado
- C. Sem gates automáticos

[Answer]: A

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
