# Evidências

## Inspeção realizada

- Projeto classificado como greenfield.
- Única fonte inspecionada: `aidlc/spaces/default/memory/org.md`.
- Seções consultadas: `## Way of Working`, `## Walking Skeleton`, `## Testing Posture`, `## Deployment` e `## Code Style`.
- `team.md` e `project.md` foram tratados como ainda não afirmados; seus templates não fornecem fatos da equipe.
- O escopo atual é `feature`, com contexto de desenvolvimento solo para hackathon.

## Inferências provisórias

- Desenvolvimento baseado em trunk, branch `main`, branches curtos e squash-merge parecem adequados ao contexto.
- `test-after` foi escolhido como metodologia provisória, conforme default organizacional para ausência de postura afirmada.
- O escopo `feature` implica piso de 80% de cobertura de linhas e CI antes do merge.
- Deploy em staging após merge e aprovação manual para produção foram mantidos como proposta organizacional, sem assumir que ambientes reais existam neste hackathon.
- Estilo deve seguir formatter/linter configurados no repositório.
- Não foram criadas regras rígidas adicionais porque defaults organizacionais já cobrem os temas observados.

## Incertezas a resolver na entrevista

1. A equipe confirma trunk-based com branches de vida curta e squash-merge? Há necessidade de PR obrigatório ou revisão antes do merge?
2. O projeto deve executar walking skeleton formal (`skeleton: on`) ou pular a cerimônia (`skeleton: off`)? Qual fluxo mínimo precisa estar funcionando primeiro?
3. `test-after` é adequado, ou a equipe prefere TDD, BDD, ATDD ou método customizado? A meta de 80% é viável para o hackathon e será medida por qual ferramenta?
4. CI antes do merge será executado em qual plataforma e com quais comandos obrigatórios?
5. Existem ambientes staging e produção reais? Quem aprova produção? Qual estratégia de rollback e smoke test será usada?
6. Quais formatter, linter, convenções de nomenclatura e regras de revisão já devem ser consideradas?
7. Há restrições rígidas específicas para dados pessoais, segredos, logs ou integrações imobiliárias que devem virar regras `ALWAYS`/`NEVER`?
8. O modo de execução solo deve usar gates a cada Bolt ou permitir autonomia após o primeiro Bolt?
