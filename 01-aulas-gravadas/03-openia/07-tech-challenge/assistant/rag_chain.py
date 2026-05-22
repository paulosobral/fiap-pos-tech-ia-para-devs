"""
rag_chain.py
============
LangChain ConversationalRetrievalChain that combines:
  - Fine-tuned LLM (via llm_loader)
  - FAISS retriever over medical protocols (via vector_store)

Usage:
    from assistant.rag_chain import build_rag_chain, ask
    chain, memory = build_rag_chain()
    result = ask(chain, "Qual o protocolo para sepse?")
    print(result["answer"])
    print(result["source_documents"])
"""

from __future__ import annotations

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

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


def build_rag_chain(
    use_adapter: bool = True,
    window_k: int = 5,
) -> tuple[ConversationalRetrievalChain, ConversationBufferWindowMemory]:
    llm = build_llm(use_adapter=use_adapter)
    retriever = get_retriever()

    memory = ConversationBufferWindowMemory(
        k=window_k,
        memory_key="chat_history",
        output_key="answer",
        return_messages=False,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=_CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )
    return chain, memory


def ask(
    chain: ConversationalRetrievalChain,
    question: str,
    patient_context: str = "",
) -> dict:
    full_question = question
    if patient_context:
        full_question = f"{question}\n\nContexto do paciente:\n{patient_context}"

    result = chain.invoke({"question": full_question})
    sources = [
        doc.metadata.get("source", "Protocolo interno")
        for doc in result.get("source_documents", [])
    ]
    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(sources)),  # deduplicated, order preserved
    }
