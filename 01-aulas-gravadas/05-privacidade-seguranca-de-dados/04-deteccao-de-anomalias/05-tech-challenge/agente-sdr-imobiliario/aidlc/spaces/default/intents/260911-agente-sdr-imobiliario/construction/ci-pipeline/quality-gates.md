# Quality Gates — Agente SDR Imobiliário (POC)

Gates executados pela esteira local (`start.sh`, ver `ci-config.md`) antes do deploy; refletem o contrato de testes (Testing Contract) e as regras afirmadas.

| Gate | Critério | Comando | Bloqueante? | Status atual |
|---|---|---|---|---|
| Build | exit 0 | `python -m compileall -q apps` | Sim | ✓ |
| Testes (7 apps) | 100% passed (unit + integration) | `pytest apps/<app>/tests --cov=... --cov-fail-under=80 -q` (7 comandos em `ci-config.md`) | **Sim** | ✓ 624/0/5 |
| Cobertura | ≥80% por unit — **informativa** | `--cov-report=term` | **Não** (`NEVER tratar cobertura como gate bloqueante neste hackathon` — project.md) | ✓ 95.29–100% |
| Segurança estática | zero `eval/exec/os.system/subprocess` fora de testes | grep em `apps` | Sim | ✓ CLEAN |
| Suíte de segurança | 0 falhas (PII/guardrails/consent/TTL) | subset pytest (security-test-instructions.md) | Sim (incluso no gate de testes) | ✓ |
| Smoke UI | dashboard-ui passa | `pytest apps/dashboard-ui/tests -q` | Sim | ✓ 8/0/2 |
| Deploy | após gates, somente via `start.sh` | `start.sh` → `terraform apply` | Sim (manual) | ⏳ deployment-execution (design: deployment-pipeline) |

## Não-bloqueantes (informativos)
- Cobertura por unit (registrar a cada run; alvo do contrato ≥80% já superado).
- Duração da esteira local (alvo: minutos; medir quando `start.sh` existir).

## Gates futuros (não aplicáveis neste POC local — registados na Target Verification Matrix)
- Lint formal (Ruff) — nenhuma config no repo hoje; sugerido para `deployment-pipeline`/`feedback-optimization`.
- SAST/DAST — requer ambiente provisionado (`deployment-pipeline`).
- Performance (p90 < 10 s) — `performance-validation` (owner registrado).

## Verificação de boundary (Step 5 do estágio)
Os gates **reforçam exatamente os comandos de build/teste registrados pelo Build and Test** (`construction/build-and-test/test-results.md`): mesmos 8 comandos pytest + `compileall` + fail-under idêntico. Nenhuma meta foi enfraquecida (regra do estágio: gates não podem reduzir metas — cobertura não-bloqueante é regra afirmada **pelo humano**, não enfraquecimento).
