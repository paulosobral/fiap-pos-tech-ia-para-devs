**Collaborator:** aidlc-quality-agent

## Contribution

- **Testabilidade geral:** Parcial. Há bons limiares e resultados observáveis em vários ACs, mas muitos critérios descrevem intenção técnica ou comportamento subjetivo sem definir dados de entrada, saída esperada, janela de medição, ambiente e tolerância.
- **Formato Given/When/Then:** Não seguido explicitamente. Os ACs estão em formato declarativo; recomenda-se reescrever cada critério como cenário Given/When/Then, especialmente para fluxos alternativos, falhas e regras condicionais.
- **Independência e verificabilidade:** Parcial. A maioria dos critérios pode ser verificada isoladamente após preparar dependências, porém há critérios que misturam implementação, integração e resultado. US11.1, por exemplo, exige uma demonstração ao vivo, o que não é um critério automatizável de aceitação. US9.1.5 exige falso-positivo documentado, mas não define conjunto de dados, método de validação ou resultado esperado.
- **Pontos fortes:**
  - US1.1 define tempo de resposta, criação de sessão e registro de consentimento.
  - US1.3 define cadeia técnica completa de voz e limite de 15s para áudios menores que 30s.
  - US2.1, US3.1 e US3.2 estabelecem métricas e regras objetivas para classificação, score, urgência e handoff.
  - US4.1 protege contra alucinação e estabelece volume da base e latência de busca.
  - US5.1, US7.1, US9.1 e US11.1 cobrem resultados operacionais, auditoria e comportamento de fallback.
  - Dependências entre stories estão explícitas e a priorização Must/Should/Could está alinhada em grande parte aos requisitos.
- **Ajustes nos critérios de aceitação:**
  - US1.1: definir conteúdo mínimo da saudação, comportamento quando consentimento é recusado, mensagens duplicadas e falha do DynamoDB/Telegram. Medir “< 10s” com percentil e tamanho de amostra, pois requisito de sucesso usa tempo de primeira resposta.
  - US1.2: definir o que significa “processa naturalmente” por intents/respostas esperadas; testar mensagens vazias, idioma não suportado, texto muito longo, contexto expirado e falha de persistência. Tornar TTL verificável pela política de expiração e não somente por configuração.
  - US1.3: definir formatos inválidos, áudio sem fala, ruído, idioma diferente, arquivo acima de 30s, falha do ffmpeg/STT e fallback ao usuário. “< 15s” precisa indicar percentil, hardware e condições de carga.
  - US2.1: fixar dataset de avaliação, definição de precisão, matriz de classes, tamanho mínimo da amostra e regra para baixa confiança/ambiguidade. Separar classificação de intenção da classificação de persona, hoje combinadas nos mesmos ACs.
  - US2.2: explicitar campos obrigatórios/opcionais, unidades e validações de metragem, orçamento, prazo, quantidade de pessoas, ticket e retorno. Definir correção de respostas inválidas, “não sei”, alteração de resposta e retomada após interrupção.
  - US3.1: documentar fórmula/versionamento do score, pesos, valores ausentes, limites inclusivos e casos de score exatamente 0, 70 e 100. Definir como urgência é calculada e validar consistência entre score exibido e dados persistidos.
  - US3.2: esclarecer precedência entre “score >= 70” e solicitação explícita, quais informações restantes são obrigatórias por persona/intenção, estado do handoff e idempotência. Definir resposta quando anomalia é detectada e como o lead pode regularizar a situação.
  - US4.1/US4.2: definir top-k, ordenação e critério de relevância; comportamento sem resultados, empate, dados incompletos, imóveis indisponíveis e filtros incompatíveis. Validar que cada recomendação contém identificador existente na fonte e que preço/disponibilidade refletem a mesma versão da base. Definir o que “índice carregado em memória da Lambda” significa em cold start e warm start.
  - US5.1: definir fuso horário, duração, antecedência mínima, slots concorrentes, indisponibilidade e tentativa de agendamento duplicado. Testar falha parcial entre calendário, ICS, notificação e confirmação; exigir consistência ou estado de erro recuperável.
  - US6.1: corrigir “gap” para termo definido e estabelecer schema mínimo do Markdown. Validar ausência de PII para LLM, criptografia, autorização do corretor, redaction seletiva, dados incompletos e não duplicação de handoff. O disparo por score deve ser idempotente.
  - US7.1: definir atualização considerada “tempo real”, período de cada KPI, fórmula de p90 e taxa de qualificação, timezone e comportamento de API indisponível. Incluir autorização por perfil, expiração/renovação JWT e ausência de dados. O requisito de FR7.3/roleta precisa permanecer rastreável mesmo sendo nice-to-have.
  - US9.1: fixar versão/configuração dos modelos, features, janela temporal, limiar e critério de combinação entre Isolation Forest, PCA e Autoencoder. Definir alerta duplicado, falso positivo, falso negativo, explicabilidade mínima e reversibilidade da restrição de agendamento. Testar job parcial, conversa sem dados e reprocessamento idempotente.
  - US11.1: substituir “demonstração única ao vivo” por critérios verificáveis: payload, autenticação, lead/status sincronizados, correlação/auditoria, retries, DLQ, replay e idempotência. Definir comportamento quando HubSpot rejeita, expira credencial, retorna timeout ou recebe evento duplicado.
- **Cobertura dos FRs:** FR1–FR7, FR9 e FR11 têm stories Must Have; FR8, FR10 e FR12 aparecem como stories Nice-to-Have, cobrindo todos os FRs em nível nominal. Entretanto, US8.1, US10.1 e US12.1 não possuem critérios de aceitação, portanto não são testáveis nem demonstram cobertura real dos respectivos sub-requisitos. FR10 também não está coberto pelos ACs de US7.1, apesar do gráfico de leads distribuídos pela roleta.
- **Cobertura dos NFRs:** NFR1 aparece parcialmente em US1.1/US1.3/US4.1; NFR2/NFR3 aparecem parcialmente em consentimento, TTL e handoff; NFR4 aparece apenas no DLQ de US11.1; NFR5 tem métricas no dashboard, mas não logs/traços; NFR6, NFR7, NFR8 e NFR9 não têm ACs correspondentes. Criar matriz NFR→story→AC e ACs específicos para mascaramento antes do LLM, criptografia/KMS, auditoria, logs/traços, custo/token, guardrails/prompt injection, concorrência/escala e Terraform/scripts recriáveis.
- **Cobertura de metas da POC:** A primeira resposta é parcialmente coberta; a taxa de leads qualificados >= 60% não tem story/AC; a precisão >= 85% é coberta por US2.1, mas precisa de protocolo de medição. Criar AC de métrica agregada e período de observação.
- **Edge cases transversais ausentes:** consentimento recusado/revogado, solicitação de exclusão de dados, lead repetido, concorrência na mesma sessão, mensagens fora de ordem, Telegram webhook duplicado, indisponibilidade de AWS/LLM, timeout e retry, prompt injection, PII enviada em texto/áudio, payload malformado, limite de tamanho, timezone/horário de verão, falhas parciais e reprocessamento idempotente.
- **Recomendação de qualidade:** adicionar IDs de AC ligados aos sub-requisitos (FR/NFR), dados de teste e resultado esperado; separar critérios de comportamento do usuário de detalhes de implementação; incluir cenários positivos, negativos, limites e recuperação. Manter critérios de performance com percentil, carga, ambiente e amostra definidos.

## Positions

- OBJECT: Critérios não seguem Given/When/Then e várias expressões (“humanizada”, “processa naturalmente”, “em tempo real”, “uso criptografado”) permanecem subjetivas ou sem métrica operacional.
- OBJECT: US8.1, US10.1 e US12.1 foram listadas, mas não têm acceptance criteria; cobertura dos FR8, FR10 e FR12 não é verificável.
- OBJECT: NFR4–NFR9, incluindo segurança de modelo, observabilidade, custo, escalabilidade e IaC, não possuem cobertura suficiente nas stories/ACs.
- OBJECT: Metas de sucesso da POC não estão integralmente testáveis: falta AC para taxa de leads qualificados >= 60%.
- AGREE: Priorização, dependências e decomposição por FR fornecem boa base para uma matriz de rastreabilidade e execução incremental.
- AGREE: ACs de latência, precisão, score, limites de recomendação, TTL, DLQ e bloqueio por anomalia são pontos fortes e devem ser preservados ao refinamento.
