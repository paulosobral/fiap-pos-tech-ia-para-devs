**Collaborator:** aidlc-devsecops-agent

## Contribution

### Posição de segurança

Para um POC greenfield, solo e de escopo `feature`, proponho gates automatizados mínimos, rápidos e bloqueantes para riscos críticos. O fato de ser hackathon não reduz obrigações sobre dados pessoais: o caminho preferencial é usar dados sintéticos; dados reais somente com necessidade justificada, acesso restrito e retenção definida.

### Gates propostos para decisão na entrevista

1. **Qualidade de código**
   - Formatter em modo check e linter no CI antes do merge; falha bloqueia merge.
   - Manter piso de 80% de cobertura de linhas já proposto para `feature`, sem permitir redução para mascarar falhas.
   - Confirmar linguagem, comandos e arquivos de configuração existentes no repositório.

2. **SAST e segurança de aplicação**
   - Executar SAST no CI em cada branch/PR.
   - Bloquear merge para vulnerabilidades `Critical`/`High`; `Medium` exige triagem ou prazo explícito.
   - Validar entradas nas fronteiras, autorização, tratamento de erros e saída sem exposição de PII.
   - Para POC sem API publicada, SAST é obrigatório; DAST pode ser condicional à existência de endpoint implantado.

3. **DAST e testes de abuso**
   - Se houver ambiente staging ou endpoint acessível, executar DAST antes de promoção para produção/demonstração externa.
   - Bloquear promoção em achados críticos/altos confirmados.
   - Incluir casos de injeção, autenticação/autorização, exposição de dados, rate limiting e payloads malformados.
   - Confirmar se demonstração será apenas local, staging ou pública; isso define obrigatoriedade do DAST.

4. **Segredos e credenciais**
   - Secret scanning em pre-commit e CI, usando ferramenta como Gitleaks; varrer também histórico quando houver suspeita.
   - Qualquer segredo detectado bloqueia merge. Segredo vazado deve ser revogado e rotacionado antes de continuar.
   - Nunca versionar chaves, tokens, senhas, `.env` com valores reais ou credenciais de provedores.
   - Armazenar parâmetros sensíveis no AWS Systems Manager Parameter Store, preferencialmente `SecureString` com KMS; aplicar IAM de menor privilégio. Confirmar se Secrets Manager é necessário para segredos rotativos.

5. **Dependências e cadeia de suprimentos**
   - Dependency scanning no CI, com lockfile versionado e instalação reprodutível.
   - Bloquear vulnerabilidades críticas com exploit conhecido; exigir decisão registrada para exceções de alta severidade.
   - Dependências novas exigem justificativa mínima, origem confiável e manutenção ativa; evitar pacotes abandonados ou typosquatting.
   - Fixar versões, revisar mudanças de lockfile e confirmar se haverá atualização automatizada (Dependabot/Renovate).
   - Se houver imagem/container, adicionar image scan e bloquear vulnerabilidades críticas antes do deploy.

6. **LGPD e proteção de PII**
   - Classificar nome, telefone, e-mail, endereço, preferências e histórico de conversa como dados pessoais; identificar dados sensíveis caso apareçam no fluxo.
   - Preferir fixtures e massa sintética em desenvolvimento, testes, logs e screenshots.
   - Mascarar PII em logs, traces, métricas, mensagens de erro, relatórios de CI e artefatos de demonstração; nunca registrar tokens ou conteúdo integral de conversa sem necessidade.
   - Criptografar dados em trânsito com TLS e em repouso conforme serviço utilizado; aplicar controle de acesso e menor privilégio.
   - Definir finalidade, retenção e descarte; restringir exportação para notebooks, serviços de terceiros e ferramentas de IA.
   - Confirmar base legal/finalidade, procedimento para exclusão e responsável por incidentes antes de usar dados reais.

7. **Exceções e evidência**
   - Exceções devem ser explícitas, justificadas, com risco, responsável, prazo de expiração e plano de correção; nunca usar bypass permanente.
   - CI deve preservar relatórios dos scans sem incluir PII ou segredos.
   - Confirmar quem aprova exceções no modo solo e se a própria pessoa registra revisão independente posterior.

### Perguntas que entrevista precisa decidir

- Quais ferramentas e comandos serão usados para formatter, linter, SAST, DAST, secret scanning e dependências?
- Quais severidades bloqueiam merge e promoção? Vulnerabilidades sem exploit conhecido podem receber prazo?
- Haverá staging/endpoint público? Em caso negativo, DAST fica condicionado à primeira publicação.
- O POC usará somente dados sintéticos? Se dados reais forem inevitáveis, qual finalidade, retenção, acesso e processo de mascaramento?
- Parameter Store será obrigatório para todos os parâmetros sensíveis? Quem administra KMS/IAM?
- Quais dependências e imagens exigem aprovação? Haverá lockfile e atualização automatizada?
- Quem registra e aprova exceções de segurança em desenvolvimento solo?

## Positions

AGREE: Lint/formatter no CI, falha bloqueante, é consistente com defaults organizacionais.

AGREE: Secret scanning, ausência de segredos hardcoded, armazenamento em Parameter Store, mascaramento de PII e uso preferencial de dados sintéticos devem ser gates obrigatórios, não recomendações.

AGREE: SAST e dependency scanning são obrigatórios no CI; DAST depende da existência de endpoint implantado, sem dispensar testes de abuso aplicáveis.

OBJECT: Não tratar cobertura de 80% como substituto de segurança; cobertura mede execução de código, não ausência de vulnerabilidades.

OBJECT: Não permitir exceções indefinidas ou silenciosas por causa do contexto solo/hackathon.
