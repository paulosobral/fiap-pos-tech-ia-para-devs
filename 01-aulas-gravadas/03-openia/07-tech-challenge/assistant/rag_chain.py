"""
rag_chain.py
============
Chain LCEL do LangChain que combina:
    - LLM ajustado (via llm_loader)
    - Retriever FAISS sobre protocolos médicos (via vector_store)

Uso:
        from assistant.rag_chain import build_rag_chain, ask
        chain, _ = build_rag_chain()
        result = ask(chain, "Qual o protocolo para sepse?")
        print(result["answer"])
        print(result["sources"])
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from assistant.llm_loader import build_llm
from assistant.vector_store import TOP_K, build_vector_store

_CONDENSE_TEMPLATE = """Dado o histórico de conversa e a nova pergunta do usuário,
reformule a pergunta para ser autossuficiente (sem precisar do histórico).
Se a pergunta já for clara, retorne-a sem alterações.

Histórico de conversa:
{chat_history}

Pergunta: {question}
Pergunta reformulada:"""

_QA_TEMPLATE = """Você é um assistente médico virtual de suporte a médicos.
Use os trechos dos protocolos hospitalares abaixo para responder à pergunta.
Se a resposta não estiver nos protocolos, diga que não encontrou informação específica nos documentos.
NUNCA prescreva medicamentos diretamente — sempre inclua o aviso de validação médica.
Indique a fonte das informações quando disponível.

Protocolos relevantes:
{context}

Pergunta: {question}
Resposta (em português):"""

# Marcadores que indicam vazamento para o próximo exemplo Alpaca.
_ALPACA_ARTIFACT_MARKERS = (
    "\n### Input:",
    "\n### Instruction:",
    "\nInput:\n",
    "\n---\n",
)


def _strip_alpaca_artifacts(text: str) -> str:
    """Trunca a saída do LLM no primeiro marcador de continuação no formato Alpaca."""
    for marker in _ALPACA_ARTIFACT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


_NO_EVIDENCE_RESPONSE = (
    "Não encontrei evidência suficientemente relevante nos protocolos recuperados para "
    "responder com segurança. Sugiro reformular a pergunta com mais detalhes clínicos "
    "ou consultar avaliação médica presencial."
)

_CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=_CONDENSE_TEMPLATE,
)

_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_QA_TEMPLATE,
)


class _MedicalRAGChain:
    """Cadeia de recuperação conversacional stateful baseada em LCEL."""

    def __init__(self, llm, vector_store, window_k: int, top_k: int = 3, max_distance: float = 0.85) -> None:
        parser = StrOutputParser()
        self._condense = _CONDENSE_PROMPT | llm | parser
        self._qa = _QA_PROMPT | llm | parser
        self._store = vector_store
        self._window_k = window_k
        self._top_k = top_k
        self._max_distance = max_distance
        self._history: list[tuple[str, str]] = []

    def _format_history(self) -> str:
        tail = self._history[-self._window_k:]
        return "\n".join(f"Human: {h}\nAI: {a}" for h, a in tail)

    def clear_history(self) -> None:
        self._history.clear()

    def invoke(self, question: str, patient_context: str = "", use_history: bool = True) -> dict:
        full_q = (
            f"{question}\n\nContexto do paciente:\n{patient_context}"
            if patient_context
            else question
        )

        history = self._format_history() if use_history else ""
        standalone = (
            self._condense.invoke({"chat_history": history, "question": full_q})
            if history
            else full_q
        )

        # No FAISS, distância menor significa melhor match semântico.
        hits = self._store.similarity_search_with_score(standalone, k=self._top_k)
        filtered_hits = [(doc, score) for doc, score in hits if score <= self._max_distance]

        if not filtered_hits:
            answer = _NO_EVIDENCE_RESPONSE
            if use_history:
                self._history.append((full_q, answer))
            return {
                "answer": answer,
                "sources": [],
                "low_evidence": True,
                "retrieval_scores": [score for _doc, score in hits],
            }

        docs = [doc for doc, _score in filtered_hits]
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = _strip_alpaca_artifacts(
            self._qa.invoke({"context": context, "question": full_q})
        )

        if use_history:
            self._history.append((full_q, answer))

        sources = [
            doc.metadata.get("source", "Protocolo interno") for doc in docs
        ]
        return {
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),  # deduplicado, ordem preservada
            "low_evidence": False,
            "retrieval_scores": [score for _doc, score in filtered_hits],
        }


def build_rag_chain(
    use_adapter: bool = True,
    window_k: int = 5,
) -> tuple[_MedicalRAGChain, None]:
    llm = build_llm(use_adapter=use_adapter)
    vector_store = build_vector_store()
    chain = _MedicalRAGChain(llm, vector_store, window_k)
    return chain, None  # None mantém compatibilidade para chamadas: chain, _ = build_rag_chain()


def ask(
    chain: _MedicalRAGChain,
    question: str,
    patient_context: str = "",
    use_history: bool = True,
) -> dict:
    return chain.invoke(question, patient_context, use_history=use_history)
