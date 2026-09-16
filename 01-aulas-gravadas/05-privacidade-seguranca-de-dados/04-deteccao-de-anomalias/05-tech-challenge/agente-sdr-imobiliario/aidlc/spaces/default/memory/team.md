# Team-Level Rules

> This team's affirmed practices and corrections. Loaded after `org.md` as
> strict-additive guidance; contradictions with broader policy are rejected.
> Populated by the practices-discovery affirmation gate. Edit at the gate,
> not directly.

## Way of Working

Trabalhamos com branch por feature. Cada mudança nasce em uma branch própria, partindo da branch `feature/01-aulas-gravadas/05-privacidade-seguranca-de-dados`, e integra em `main` via pull request com merge. Não usamos trunk-based com squash-merge direto; branches são revisadas e mescladas via PR.

## Walking Skeleton

Sim — construímos uma fatia fina de ponta a ponta primeiro (walking skeleton): uma versão mínima que roda o caminho todo, do primeiro contato ao fim, antes das features reais, para provar que as peças se conectam.

## Testing Posture

Tratamos testes como entregável de cada Bolt. Escrevemos o código primeiro e depois os testes (test-after), sem meta de cobertura bloqueante definida para o hackathon.

- **Methodology**: test-after
- **Ordering**: Implementamos cada camada testável aplicável e, em seguida, escrevemos e executamos os testes dessa camada antes de avançar para a próxima.

## Change Control

<!-- Affirmed by the team. Mode: strict or relaxed. Strict here holds for every intent and cannot be changed from chat. -->

## Deployment

Não temos esteira de CI/CD. O deploy é manual, quando quisermos, por meio de scripts `start.sh`/`stop.sh` que executam build, testes e `terraform apply` / `terraform destroy` para provisionar e derrubar o ambiente.

## Code Style

Seguimos configurações existentes no projeto para formatter e linter do repositório. Usamos convenções idiomáticas da linguagem e não criamos regras de nomenclatura adicionais sem decisão explícita da equipe. Antes de sugerir estilo, consultamos as configurações do repositório.
## Forbidden

<!-- Team-specific forbidden patterns -->

## Mandated

<!-- Team-specific mandates -->

## Corrections

<!-- Self-learning loop appends here. -->
