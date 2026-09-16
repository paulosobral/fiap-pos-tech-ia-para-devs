# Practices Discovery Questions

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt seguindo o ciclo completo do AI-DLC — Initialization → Ideation → Inception → Construction → Operation. Partir da captura de intent (problema, clientes, canais, roleta, esteira Kanban, cenários C1–C7) e produzir, em ordem, os artefatos das fases: requisitos, user stories, domínio, contrato de API, mockups e o PRD final (padrão do framework em docs/guide/00-introduction.md). O diretório já possui o PRD preliminar e a transcrição da mentoria em documentos/ para uso como fonte."
- [scope] Workflow-selected scope: `feature`.
- [assumption] Greenfield — defaults sugeridos de `aidlc/spaces/default/memory/org.md`; nada é fato estabelecido até você afirmar.

> Entrevista de práticas do time (fase Inception). Dev solo, POC hackathon, candidato das perguntas Q1–Q5.

---

## Q1. Como você quer trabalhar com branches e integração?

- A. Trunk-based: todo mundo trabalha em `main`, mudanças curtas com merge frequente
- B. Branch por feature com merge via PR
- C. Sem regra — devolo solo
- X. Other (especificar)

[Answer]: Respsota B a partir da branch feature/01-aulas-gravadas/05-privacidade-seguranca-de-dados

## Q2. Construir uma fatia fino de ponta a ponta primeiro?

Uma "fatia fina de ponta a ponta" (walking skeleton) é uma versão mínima que roda o caminho todo — do first contato ao fim — construída antes das features reais, para provar que as peças se conectam.

- A. Sim — construir a fatia fina primeiro
- B. Não — ir direto às features
- X. Other (especificar)

[Answer]: A

## Q3. Como você quer testar?

- A. Escrever o código primeiro e depois os testes (test-after)
- B. Escrever os testes antes do código (TDD)
- C. Misto: cenários de ponta a ponta antes, testes menores depois
- D. Sem testes (POC de demonstração)
- X. Other (especificar)

[Answer]: A

## Q4. Como avaliar a cobertura e o gate de qualidade?

- A. Cobertura de 80% com CI bloqueando merge abaixo disso
- B. Cobertura sem gate bloqueante
- C. Sem meta de cobertura
- X. Other (especificar)

[Answer]: B

## Q5. Como fazer deploy e qual o padrão de estilo?

- A. Deploy no merge para teste; produção com aprovação manual; formatter/linter do projeto (Prettier/Black)
- B. Deploy automático até produção
- C. Deploy manual quando quiser
- X. Other (especificar)

[Answer]: Não tenho esteira CI/CD. Pretendo solicitar a criação de script de start.sh/stop.sh para fazer o build, testes e terraform apply / terraform destroy para aplicar a opção C.

---

## Assumptions & Open Questions

- Cobertura/CI: à afirmar em Q4.
- Segurança (LGPD) — os controles já decididos na feasibility (masking, SSM, sintéticos) não serão repetidos como gate novo, a menos que você queira.

## Review

Este arquivo é o registro das perguntas da entrevista de práticas. As respostas assinaladas alimentam team-practices.md e discovered-rules.md.