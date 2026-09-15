# Práticas da Equipe

> Rascunho inicial para entrevista de práticas. Conteúdo abaixo traduz sugestões organizacionais em voz da equipe; nada está afirmado até aprovação humana.

## Way of Working

Trabalhamos com desenvolvimento baseado em trunk. Integramos mudanças em `main` por branches de vida curta, normalmente resolvidas em até 1–2 dias, e usamos squash-merge. Não mantemos branches de longa duração; quando houver trabalho incompleto, avaliamos feature flags.

## Walking Skeleton

Para este projeto solo de hackathon, propomos começar pelo menor fluxo executável de ponta a ponta quando isso reduzir risco de integração. A necessidade de uma cerimônia formal de walking skeleton, sua ativação (`skeleton: on` ou `skeleton: off`) e a exigência de aprovação antes dos demais Bolts permanecem sujeitas à entrevista.

## Testing Posture

Tratamos testes como entregável de cada Bolt. Para o escopo `feature`, adotamos provisoriamente o padrão organizacional de cobertura mínima de 80% de linhas e execução de CI antes do merge.

- **Methodology**: test-after
- **Ordering**: Implementamos cada camada testável aplicável e, em seguida, escrevemos e executamos os testes dessa camada antes de avançar para a próxima.

O Build and Test verifica o piso de cobertura e os alvos de qualidade definidos; não reduzimos esses critérios para fazer uma etapa passar.

## Deployment

Fazemos deploy em staging após o merge. Produção exige aprovação manual separada, normalmente de liderança técnica e responsável pelo produto, por proteção de ambiente da plataforma de CD. Para este protótipo, confirmaremos na entrevista se haverá staging real, produção real ou somente execução local/demonstração.

## Code Style

Seguimos configurações existentes no projeto para formatter e linter. O CI executa o linter antes do merge e falha quando houver erros. Usamos convenções idiomáticas da linguagem e não criamos regras de nomenclatura adicionais sem decisão explícita da equipe. Antes de sugerir estilo, consultamos as configurações do repositório.
