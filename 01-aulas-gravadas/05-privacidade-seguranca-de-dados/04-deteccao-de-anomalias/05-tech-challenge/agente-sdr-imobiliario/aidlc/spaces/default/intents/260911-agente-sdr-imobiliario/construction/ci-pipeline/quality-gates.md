# Quality Gates — Agente SDR Imobiliário (POC)

Gates da esteira local (`start.sh`, ver `ci-config.md`), executados **antes** do build e do deploy. Estado verificado em 2026-10-10 contra o `start.sh` e as suítes (ver `construction/build-and-test/test-results.md`).

| Gate | Critério | Comando (no `start.sh`) | Bloqueante? | Estado atual |
|---|---|---|---|---|
| Build | exit 0 | `python -m compileall -q apps` | Sim | ✓ |
| Testes das 8 aplicações | 100% passed (unidade + integração) | `pytest apps/<app>/tests --ignore=.../tests/quality` (6 apps do laço, `dashboard-api` e `dashboard-ui`) | **Sim** | ✓ 1176 passed |
| Cobertura por aplicação | ≥ 80% | `--cov-fail-under=80` (7 apps; o `dashboard-ui` roda sem cobertura) | **Sim** (decisão A, 2026-10-10) | ✓ 83,3% a 100% |
| Índices DynamoDB × Terraform | todo índice usado no código existe no `.tf` da tabela certa | `pytest tests/infra` | Sim | ✓ 4 passed |
| Qualidade com LLM real | 100% passed (repete 1x só o que falhou, pela oscilação da LLM) | `pytest apps/conversation-router/tests/quality` | Sim (**pulado sem chave** de LLM, com aviso) | ✓ 62/62 (2026-10-10) |
| Smoke pós-deploy | API e `/api/kpis` respondem | fase 6 do `start.sh` | Sim | ✓ nas subidas reais |
| Deploy | só via `start.sh` | `terraform apply` em 2 passos | Sim (manual, `NEVER` fora dos scripts) | ✓ |

Não existe gate de **segurança estática** automático: o grep `eval/exec/os.system/subprocess` do desenho original nunca virou passo do `start.sh`. Hoje é conferido à mão no `build-and-test` (1 ocorrência permitida: `ffmpeg` no `voice-adapter`, sem shell).

## Decisão do responsável (2026-10-10): opção A
Cobertura ≥ 80% **continua bloqueante** no `start.sh` (`--cov-fail-under=80`). Ela nunca falhou (real: 83–100%) e protege contra regressão. A regra `NEVER tratar cobertura de testes como gate bloqueante` de `project.md` fica superada por esta decisão; a troca da regra é feita pelo ritual de aprendizados (não por edição direta).

## Não-bloqueantes (informativos)
- Cobertura detalhada por aplicação (registrada a cada rodada no `test-results.md`).
- Duração da esteira (o gate com LLM real leva ~6 min e consome créditos do OpenRouter).

## Gates futuros (fora do POC)
- Lint formal (Ruff): não há configuração no repositório.
- SAST/DAST e verificação de latência (p90 < 10 s): `performance-validation`.
- CI em PR (GitHub Actions): proposto e recusado em 2026-09-21 ("GitHub só repositório"); revisitar se o repositório ganhar colaboradores.

## Verificação de boundary (Step 5)
Os gates reforçam os comandos de build e teste registrados pelo `build-and-test` (mesmos comandos, mesmo piso). Ver `verification/phase-check-construction.md`.
