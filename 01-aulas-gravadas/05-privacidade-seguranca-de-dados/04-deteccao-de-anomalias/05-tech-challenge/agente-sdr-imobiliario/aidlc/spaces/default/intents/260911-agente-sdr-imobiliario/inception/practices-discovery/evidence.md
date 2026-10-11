# Evidências

## Inspeção realizada

- Projeto classificado como greenfield.
- Única fonte inspecionada: `aidlc/spaces/default/memory/org.md`.
- Seções consultadas: `## Way of Working`, `## Walking Skeleton`, `## Testing Posture`, `## Deployment` e `## Code Style`.
- `team.md` e `project.md` foram tratados como ainda não afirmados; seus templates não fornecem fatos da equipe.
- O escopo atual é `feature`, com contexto de desenvolvimento solo para hackathon.

## Inferências provisórias

- Desenvolvimento baseado em branch por feature, com integração via PR partindo de `feature/01-aulas-gravadas/05-privacidade-seguranca-de-dados` (resposta Q1 da entrevista).
- Walking skeleton: sim — fatia fina de ponta a ponta primeiro (resposta Q2).
- `test-after` escolhido como metodologia, conforme resposta Q3.
- Sem meta de cobertura bloqueante; testes são entregável de cada Bolt sem gate de cobertura (resposta Q4).
- Deploy manual sem esteira de CI/CD, via scripts `start.sh`/`stop.sh` + `terraform apply`/`destroy` (resposta Q5).
- Estilo segue formatter/linter configurados no repositório.
- Não foram criadas regras rígidas adicionais além das Mandated/Forbidden derivadas das respostas Q1, Q2 e Q5.

## Incertezas a resolver na entrevista

1. A equipe confirma trunk-based com branches de vida curta e squash-merge? Há necessidade de PR obrigatório ou revisão antes do merge?
2. O projeto deve executar walking skeleton formal (`skeleton: on`) ou pular a cerimônia (`skeleton: off`)? Qual fluxo mínimo precisa estar funcionando primeiro?
3. `test-after` é adequado, ou a equipe prefere TDD, BDD, ATDD ou método customizado? A meta de 80% é viável para o hackathon e será medida por qual ferramenta?
4. CI antes do merge será executado em qual plataforma e com quais comandos obrigatórios?
5. Existem ambientes staging e produção reais? Quem aprova produção? Qual estratégia de rollback e smoke test será usada?
6. Quais formatter, linter, convenções de nomenclatura e regras de revisão já devem ser consideradas?
7. Há restrições rígidas específicas para dados pessoais, segredos, logs ou integrações imobiliárias que devem virar regras `ALWAYS`/`NEVER`?
8. O modo de execução solo deve usar gates a cada Bolt ou permitir autonomia após o primeiro Bolt?
