**Collaborator:** aidlc-quality-agent

## Contribution

### Avaliação do posture de testes

O rascunho está coerente com o default organizacional para escopo `feature`: testes são entregável de cada Bolt, a metodologia provisória é `test-after`, a ordenação é implementar uma camada testável e então escrever/executar seus testes, e existe piso de 80% de cobertura de linhas com CI antes do merge. Essa posição ainda não está afirmada pela equipe; deve ser apresentada como proposta na entrevista.

Para este POC greenfield, Python e AWS serverless, recomendo manter uma pirâmide enxuta e orientada a risco:

- **Unitários (maioria)**: `pytest`, funções puras e handlers isolados, dependências AWS substituídas por stubs/fakes ou mocks somente nas interfaces próprias. Cobrir caminhos felizes, validação de entrada, ausência de campos, respostas de erro e tratamento de timeout/retry.
- **Integração (menor conjunto)**: validar contratos entre handler, serialização, persistência e serviços AWS. Preferir testes locais com emulador/fake quando isso reduzir custo e flakiness; não confundir teste unitário com teste que apenas verifica chamada de mock.
- **E2E/aceitação (mínimo)**: um fluxo crítico do SDR através da interface de entrada até resposta persistida/emitida, executado em ambiente implantado ou equivalente somente se houver staging real. Se o hackathon tiver apenas demonstração local, registrar explicitamente a limitação.
- **Contrato**: validar formato de eventos de entrada/saída e integrações externas relevantes. Pact só é necessário se houver consumidores/provedores independentes; caso contrário, fixtures versionadas e testes de schema são suficientes.

### Cobertura e tooling propostos

Adotar `pytest` como runner e `pytest-cov`/`coverage.py` para medir cobertura. O gate deve aplicar 80% de cobertura de linhas ao conjunto definido pela equipe, publicar relatório XML/HTML e falhar quando ficar abaixo do piso. A entrevista precisa decidir se o piso vale para o projeto inteiro, código novo, cada módulo ou apenas pacotes de produção; também deve confirmar se branch coverage é requisito adicional. Cobertura não substitui assertivas sobre comportamento nem rastreabilidade para critérios de aceitação.

Para teste de propriedades, `hypothesis` é uma opção útil para parsers, normalização de dados e regras de detecção de anomalias, mas não precisa ser requisito mínimo do POC. Dados de teste devem ser sintéticos; nunca reutilizar PII real. Fixtures/factories devem isolar cada caso e evitar estado mutável compartilhado.

### Qualidade e CI

A definição mínima de pronto deveria exigir: testes unitários verdes, cobertura no piso, lint/formatter sem erro, testes de integração aplicáveis verdes, nenhuma vulnerabilidade crítica/alta introduzida e evidência de execução no CI antes do merge. Para alterações em contratos ou infraestrutura AWS, incluir teste de integração/contrato correspondente. Flaky tests devem ser identificados, não ignorados silenciosamente.

A plataforma CI e os comandos ainda precisam ser definidos. Proposta de jobs Python:

1. instalação reprodutível de dependências;
2. formatter/linter (por exemplo, `ruff format --check` e `ruff check`, se adotados);
3. `pytest` com cobertura e relatório JUnit/XML;
4. testes de integração separados por marcador, quando existirem;
5. scan de dependências e SAST como gate de segurança;
6. publicação dos artefatos de teste/cobertura e proteção de merge em `main`.

Não presumo GitHub Actions, Ruff, pytest-cov, LocalStack ou qualquer outro tooling como decisão tomada; são opções a confirmar contra arquivos do repositório e restrições do hackathon.

### Lacunas que entrevista deve resolver

1. Confirmar metodologia: `test-after`, TDD, BDD, ATDD ou combinação; confirmar se a ordenação vale por camada/Bolt.
2. Definir runner, framework de asserção, formatter, linter, coverage tool e versões fixadas.
3. Definir unidade do piso de 80%: linhas de produção, código novo, pacote, projeto inteiro; decidir branch coverage e política para código gerado/infra.
4. Definir comandos obrigatórios e plataforma CI; esclarecer se CI realmente bloqueia merge em projeto solo e qual evidência será aceita para a entrega do hackathon.
5. Definir proporção e escopo de unitário, integração, contrato e E2E; identificar fluxo crítico do SDR que precisa de teste de aceitação.
6. Definir estratégia para AWS serverless: mocks/stubs, emulador local ou conta AWS de teste; credenciais, isolamento, limpeza e custo.
7. Definir dados sintéticos, tratamento de PII, fixtures e comportamento determinístico de chamadas de LLM/serviços externos.
8. Definir gates para falhas transitórias, retries, timeouts, cold starts e limites de serviço; confirmar se haverá carga/performance no POC ou somente baseline manual.
9. Definir política para testes flaky, defeitos encontrados e requisito de teste de regressão antes da correção.
10. Definir se segurança (SAST, dependências, secrets scan) bloqueia CI e quais severidades são aceitáveis.

### Posição recomendada para integração

Até a entrevista, manter `test-after` e piso de 80% como proposta provisória, sem promovê-los a regra rígida. Para um POC solo, priorizar uma suíte unitária rápida, poucos testes de integração nos limites AWS e um único E2E do fluxo crítico; não criar uma suíte extensa de E2E ou carga sem NFR e ambiente que a justifiquem.

## Positions

- AGREE: O rascunho deve preservar `test-after`, ordenação por camada, testes como entregável de cada Bolt e piso de 80%/CI como defaults provisórios do escopo `feature`.
- AGREE: A entrevista precisa confirmar metodologia, ferramenta de cobertura, plataforma/comandos CI e viabilidade do piso no prazo do hackathon.
- OBJECT: Não há base para afirmar ainda uma ferramenta específica, uma divisão fixa de testes, staging disponível ou gates adicionais; esses itens devem permanecer propostas e perguntas abertas.
- OBJECT: Cobertura de linhas isolada não é critério suficiente; comportamento, integração AWS, contratos, segurança e regressão precisam de critérios explícitos conforme o fluxo escolhido.
