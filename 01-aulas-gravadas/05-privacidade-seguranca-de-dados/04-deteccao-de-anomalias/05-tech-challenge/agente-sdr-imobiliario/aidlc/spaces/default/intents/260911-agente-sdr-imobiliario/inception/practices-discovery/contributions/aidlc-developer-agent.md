**Collaborator:** aidlc-developer-agent

## Contribution

### Avaliação de estilo e organização

A proposta em `team-practices.md` está alinhada ao default organizacional: formatter e linter devem ser definidos pelo repositório, nomes devem seguir idioma/framework e nenhuma regra adicional deve ser afirmada sem decisão da equipe. Como a inspeção é greenfield e não há código de aplicação para observar, estas recomendações são propostas, não fatos existentes.

Para um POC solo, adotar organização por **feature/domínio**, mantendo limites claros entre entrada, domínio, integrações e persistência. Um fluxo SDR pode ser dividido em:

- **Handler/controller**: interpreta evento ou requisição, valida entrada externa, chama caso de uso e serializa resposta.
- **Service/use case**: coordena regra de negócio e fluxo conversacional; não deve conhecer detalhes de SDK AWS ou HTTP.
- **Domain/model**: tipos, regras e validações de negócio puras sempre que possível.
- **Adapters/integrations**: encapsulam LLM, AWS e APIs externas atrás de interfaces estáveis; dependências devem ser injetadas.
- **Repository**: só se houver persistência ou mais de uma implementação; evitar abstração prematura para CRUD simples.
- **Config/middleware**: ambiente, logging, autenticação, correlação e tratamento de erro transversal.

Evitar estrutura global por camada (`handlers/`, `services/`, `repositories/`) que espalhe uma feature por muitos diretórios. Preferir algo como `src/sdr/{handler,service,domain,adapters}` ou equivalente idiomático da linguagem escolhida. Manter testes adjacentes à implementação ou em árvore espelhada, conforme convenção do framework.

### Convenções recomendadas

- Nomes descritivos e idiomáticos da linguagem; funções com verbo e objeto (`validateLead`, `createConversation`), booleanos com `is`/`has`/`can`/`should`, coleções no plural e constantes verdadeiras em `UPPER_SNAKE_CASE`.
- Uma responsabilidade principal por módulo e uma exportação principal quando isso melhorar descoberta; evitar arquivos agregadores usados apenas para organização interna.
- Funções pequenas, com retornos antecipados para validações e sem parâmetros booleanos que alterem múltiplos comportamentos. Usar objeto de opções quando houver muitos parâmetros.
- Limitar arquivos e funções a tamanho que preserve legibilidade; extrair conceito quando houver múltiplas responsabilidades, sem refatorar fora do escopo do Bolt.
- Não criar aliases ou renomeações cosméticas sem necessidade; seguir import/export, formatação e convenções já estabelecidas quando o projeto ganhar código.

### Erros, validação e observabilidade

Validar presença, tipo, formato, limites e regras aplicáveis na fronteira de confiança: eventos, HTTP, mensagens, uploads, variáveis de ambiente e respostas de serviços externos. Converter entrada validada em tipos de domínio; não repetir validação estrutural em toda camada.

Erros esperados devem ser representados explicitamente (por exemplo, `Result` ou erros de domínio tipados). Exceções devem ser capturadas nas fronteiras — handler, consumidor de evento ou CLI — com tratamento específico para falhas recuperáveis e propagação contextualizada para falhas fatais. Nunca engolir exceções. Respostas de API devem usar envelope consistente com código legível por máquina, mensagem segura, detalhes opcionais e `requestId`/correlation ID.

Usar logging estruturado nas fronteiras da operação, com contexto suficiente para diagnóstico, sem registrar prompts contendo PII, tokens, chaves, credenciais ou dados imobiliários sensíveis. Separar erro recuperável (retry/backoff/timeouts) de erro fatal; retries devem ser limitados e idempotência preservada.

### Limites e perguntas para entrevista

- Confirmar linguagem/runtime antes de fixar `camelCase`, `snake_case`, tipagem, formatter e linter.
- Confirmar se o handler será HTTP, evento ou ambos e qual contrato de erro será público.
- Definir quais dados de lead são PII, política de redação em logs e retenção de prompts/respostas.
- Confirmar se persistência exige repository/adapter ou se acesso direto é suficiente para o POC.
- Confirmar convenção de diretórios e se testes serão adjacentes ou espelhados.

## Positions

- AGREE: Seguir formatter/linter e convenções do projeto, sem impor regra nominal adicional antes da entrevista.
- AGREE: Para greenfield POC, usar limites simples e explícitos entre handler, caso de uso, domínio e integrações externas; injetar dependências nos adapters.
- AGREE: Validar entradas nas fronteiras, propagar erros com contexto e impedir logs de segredos ou PII.
- OBJECT: Não há evidência para afirmar linguagem, framework, estrutura de diretórios, envelope de erro ou ferramenta específica; manter esses itens como propostas condicionais.
- OBJECT: Não introduzir camadas Repository/Factory/Strategy por padrão; usar somente quando houver necessidade demonstrada por acesso a dados, múltiplos adapters ou variação real de comportamento.
