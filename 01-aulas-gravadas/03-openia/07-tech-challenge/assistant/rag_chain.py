"""
rag_chain.py
============
LangChain LCEL chain that combines:
  - Fine-tuned LLM (via llm_loader)
  - FAISS retriever over medical protocols (via vector_store)

Usage:
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
from assistant.vector_store import get_retriever

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

_CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=_CONDENSE_TEMPLATE,
)

_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_QA_TEMPLATE,
)


class _MedicalRAGChain:
    """Stateful LCEL-based conversational retrieval chain."""

    def __init__(self, llm, retriever, window_k: int) -> None:
        parser = StrOutputParser()
        self._condense = _CONDENSE_PROMPT | llm | parser
        self._qa = _QA_PROMPT | llm | parser
        self._retriever = retriever
        self._window_k = window_k
        self._history: list[tuple[str, str]] = []

    def _format_history(self) -> str:
        tail = self._history[-self._window_k:]
        return "\n".join(f"Human: {h}\nAI: {a}" for h, a in tail)

    def invoke(self, question: str, patient_context: str = "") -> dict:
        full_q = (
            f"{question}\n\nContexto do paciente:\n{patient_context}"
            if patient_context
            else question
        )

        history = self._format_history()
        standalone = (
            self._condense.invoke({"chat_history": history, "question": full_q})
            if history
            else full_q
        )

        docs = self._retriever.invoke(standalone)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self._qa.invoke({"context": context, "question": full_q})

        self._history.append((full_q, answer))

        sources = [
            doc.metadata.get("source", "Protocolo interno") for doc in docs
        ]
        return {
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),  # deduplicated, order preserved
        }


def build_rag_chain(
    use_adapter: bool = True,
    window_k: int = 5,
) -> tuple[_MedicalRAGChain, None]:
    llm = build_llm(use_adapter=use_adapter)
    retriever = get_retriever()
    chain = _MedicalRAGChain(llm, retriever, window_k)
    return chain, None  # None keeps API compatible with callers that do: chain, _ = build_rag_chain()


def ask(
    chain: _MedicalRAGChain,
    question: str,
    patient_context: str = "",
) -> dict:
    return chain.invoke(question, patient_context)
