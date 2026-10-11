<!-- INVARIANT: examples are single-line HTML comments so a fresh template parses to total=0 (MEMORY_EMPTY). Do NOT un-comment or split across lines. t100 guards this. -->
> This file is kept up to date automatically while the stage runs. Add observations at the review step, not by editing here directly.

## Interpretations
<!-- example: 2026-05-29T10:14:32Z — chose REST over GraphQL; the consuming team only needs CRUD, revisit if subscriptions land -->
- 2026-09-21T20:40:00Z — "CI pipeline" deste POC = esteira local `start.sh`/`stop.sh` (afirmado pelo humano: "não tem esteira, GitHub só repositório; deploy via scripts conforme PRD"). GitHub Actions proposto e recusado; design documenta a esteira local.
- 2026-09-21T20:40:00Z — Cobertura informativa (não-bloqueante) respeita `NEVER` afirmado em project.md; gates bloqueantes = build exit 0 + testes 100% + segurança estática + smoke UI.

## Deviations
<!-- example: 2026-05-29T10:14:32Z — skipped the optional caching layer the stage prose suggested; the dataset is small enough that it adds risk -->
- 2026-09-21T20:40:00Z — Scripts `start.sh`/`stop.sh` e Terraform NÃO implementados aqui: o estágio produz o design (`ci-config.md`); a implementação é do `deployment-pipeline` (deferred registrado: NFR9.1–9.4).
- 2026-09-21T20:55:00Z — Requisito do humano no gate: script compila cada app → `dist/<app>.zip` → Terraform (módulos HashiCorp: lambda, apigateway-v2, dynamodb-table, sqs, ecs) deploya na AWS. Nenhum stage do AI-DLC "produz" IaC/scripts — implementação atribuída ao `deployment-execution` (que materializa IaC e roda deploy); design documentado no `ci-config.md` atualizado; owner dos NFR9.x ajustado para `deployment-execution`.

## Tradeoffs
<!-- example: 2026-05-29T10:14:32Z — picked TDD over BDD this run; the team is unit-first and the domain is well-understood -->
- 2026-09-21T20:40:00Z — Sem CI no PR (revisão humana + suítes locais): troca automação por zero infra; aceito para hackathon, revisitar se o repo ganhar colaboradores.

## Open questions
<!-- example: 2026-05-29T10:14:32Z — confirm the retention window with compliance before the next stage hardens the schema -->
- (nenhuma em aberto)
