"""
patient_db.py
=============
Banco SQLite para registros sintéticos de pacientes.
Fornece operações CRUD e dados de seed para 20 pacientes sintéticos.

Uso:
    from assistant.patient_db import init_db, get_patient, get_pending_exams
    init_db()
    patient = get_patient(1)
"""

from __future__ import annotations

import datetime
import random
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    Date,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "patient_records.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False)
_SessionLocal = sessionmaker(bind=_ENGINE)


# ── Ativa WAL para leituras concorrentes ─────────────────────────────────────
@event.listens_for(_ENGINE, "connect")
def _set_wal(dbapi_conn, _):
    dbapi_conn.execute("PRAGMA journal_mode=WAL")


# ── Modelos ORM ───────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String(1), nullable=False)  # M / F
    blood_type = Column(String(3))
    allergies = Column(Text, default="Nenhuma conhecida")
    conditions = Column(Text, default="")  # comma-separated

    exams = relationship("Exam", back_populates="patient", cascade="all, delete-orphan")
    diagnoses = relationship("Diagnosis", back_populates="patient", cascade="all, delete-orphan")
    medications = relationship("Medication", back_populates="patient", cascade="all, delete-orphan")


class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    exam_name = Column(String(200), nullable=False)
    status = Column(String(20), nullable=False, default="pendente")  # pendente / concluído
    result = Column(Text, default="")
    requested_date = Column(Date, nullable=False)
    completed_date = Column(Date)

    patient = relationship("Patient", back_populates="exams")


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    cid10 = Column(String(10))
    description = Column(Text, nullable=False)
    date = Column(Date, nullable=False)

    patient = relationship("Patient", back_populates="diagnoses")


class Medication(Base):
    __tablename__ = "medications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    name = Column(String(200), nullable=False)
    dose = Column(String(100))
    frequency = Column(String(100))
    active = Column(Integer, default=1)  # 1 = ativo, 0 = suspenso

    patient = relationship("Patient", back_populates="medications")


# ── Dados de seed ─────────────────────────────────────────────────────────────

_SEED_PATIENTS = [
    {"name": "Paciente A", "age": 68, "sex": "M", "blood_type": "A+",
     "allergies": "Penicilina", "conditions": "HAS, DM2, ICC"},
    {"name": "Paciente B", "age": 52, "sex": "F", "blood_type": "O+",
     "allergies": "Nenhuma conhecida", "conditions": "Hipotireoidismo"},
    {"name": "Paciente C", "age": 75, "sex": "M", "blood_type": "B-",
     "allergies": "Dipirona", "conditions": "HAS, FA, DPOC"},
    {"name": "Paciente D", "age": 34, "sex": "F", "blood_type": "AB+",
     "allergies": "Nenhuma conhecida", "conditions": "Asma"},
    {"name": "Paciente E", "age": 61, "sex": "M", "blood_type": "A-",
     "allergies": "Contraste iodado", "conditions": "DRC estágio 3, DM2, HAS"},
    {"name": "Paciente F", "age": 45, "sex": "F", "blood_type": "O-",
     "allergies": "Nenhuma conhecida", "conditions": "Lúpus, HAS"},
    {"name": "Paciente G", "age": 80, "sex": "M", "blood_type": "A+",
     "allergies": "Sulfa", "conditions": "Parkinson, HAS, ICC, FA"},
    {"name": "Paciente H", "age": 29, "sex": "F", "blood_type": "B+",
     "allergies": "Nenhuma conhecida", "conditions": "DM1"},
    {"name": "Paciente I", "age": 57, "sex": "M", "blood_type": "O+",
     "allergies": "AAS", "conditions": "HAS, DLP, Tabagismo"},
    {"name": "Paciente J", "age": 43, "sex": "F", "blood_type": "A+",
     "allergies": "Nenhuma conhecida", "conditions": "Enxaqueca, Ansiedade"},
    {"name": "Paciente K", "age": 72, "sex": "M", "blood_type": "AB-",
     "allergies": "Captopril (tosse)", "conditions": "HAS, IAM prévio, HF reduzida"},
    {"name": "Paciente L", "age": 38, "sex": "F", "blood_type": "A+",
     "allergies": "Nenhuma conhecida", "conditions": "Hipotireoidismo, Depressão"},
    {"name": "Paciente M", "age": 66, "sex": "M", "blood_type": "B+",
     "allergies": "Metformina (intolerância GI)", "conditions": "DM2, DLP, HAS"},
    {"name": "Paciente N", "age": 55, "sex": "F", "blood_type": "O+",
     "allergies": "Nenhuma conhecida", "conditions": "Câncer de mama (em remissão), HAS"},
    {"name": "Paciente O", "age": 49, "sex": "M", "blood_type": "A-",
     "allergies": "Nenhuma conhecida", "conditions": "HIV controlado, HAS"},
    {"name": "Paciente P", "age": 83, "sex": "F", "blood_type": "O+",
     "allergies": "Warfarina (sangramento)", "conditions": "FA, HAS, ICC, Alzheimer"},
    {"name": "Paciente Q", "age": 31, "sex": "M", "blood_type": "B+",
     "allergies": "Nenhuma conhecida", "conditions": "Epilepsia"},
    {"name": "Paciente R", "age": 60, "sex": "F", "blood_type": "AB+",
     "allergies": "Estatinas (miopatia)", "conditions": "DLP, DM2, obesidade"},
    {"name": "Paciente S", "age": 77, "sex": "M", "blood_type": "A+",
     "allergies": "Nenhuma conhecida", "conditions": "HAS, DRC, Gota"},
    {"name": "Paciente T", "age": 25, "sex": "F", "blood_type": "O+",
     "allergies": "Látex", "conditions": "Asma, Rinite alérgica"},
]

_SEED_EXAMS_BY_INDEX: dict[int, list[dict]] = {
    0: [
        {"exam_name": "Ecocardiograma", "status": "pendente", "requested_date": datetime.date(2026, 5, 10)},
        {"exam_name": "HbA1c", "status": "concluído", "result": "8,2%", "requested_date": datetime.date(2026, 4, 15), "completed_date": datetime.date(2026, 4, 20)},
        {"exam_name": "Creatinina sérica", "status": "concluído", "result": "1,1 mg/dL", "requested_date": datetime.date(2026, 4, 15), "completed_date": datetime.date(2026, 4, 20)},
    ],
    2: [
        {"exam_name": "Espirometria", "status": "pendente", "requested_date": datetime.date(2026, 5, 12)},
        {"exam_name": "INR (Warfarina)", "status": "concluído", "result": "2,8", "requested_date": datetime.date(2026, 5, 1), "completed_date": datetime.date(2026, 5, 3)},
    ],
    4: [
        {"exam_name": "TFGe (CKD-EPI)", "status": "concluído", "result": "42 mL/min/1,73m²", "requested_date": datetime.date(2026, 4, 20), "completed_date": datetime.date(2026, 4, 22)},
        {"exam_name": "Potássio sérico", "status": "pendente", "requested_date": datetime.date(2026, 5, 15)},
        {"exam_name": "HbA1c", "status": "concluído", "result": "9,1%", "requested_date": datetime.date(2026, 4, 20), "completed_date": datetime.date(2026, 4, 22)},
    ],
    6: [
        {"exam_name": "ECG 12 derivações", "status": "concluído", "result": "FA com resposta ventricular controlada, 72 bpm", "requested_date": datetime.date(2026, 5, 5), "completed_date": datetime.date(2026, 5, 5)},
        {"exam_name": "INR", "status": "pendente", "requested_date": datetime.date(2026, 5, 18)},
        {"exam_name": "Ecocardiograma", "status": "pendente", "requested_date": datetime.date(2026, 5, 10)},
    ],
    10: [
        {"exam_name": "Troponina I hsTn", "status": "concluído", "result": "0h: 45 ng/L; 1h: 52 ng/L (DELTA +7)", "requested_date": datetime.date(2026, 5, 20), "completed_date": datetime.date(2026, 5, 20)},
        {"exam_name": "ECG", "status": "concluído", "result": "Infradesnivelamento de ST em D2, D3, aVF", "requested_date": datetime.date(2026, 5, 20), "completed_date": datetime.date(2026, 5, 20)},
        {"exam_name": "Cateterismo cardíaco", "status": "pendente", "requested_date": datetime.date(2026, 5, 20)},
    ],
}

_SEED_DIAGNOSES_BY_INDEX: dict[int, list[dict]] = {
    0: [
        {"cid10": "I50.0", "description": "Insuficiência cardíaca congestiva", "date": datetime.date(2024, 3, 10)},
        {"cid10": "E11", "description": "Diabetes mellitus tipo 2", "date": datetime.date(2020, 7, 5)},
        {"cid10": "I10", "description": "Hipertensão arterial sistêmica", "date": datetime.date(2018, 1, 20)},
    ],
    10: [
        {"cid10": "I21.4", "description": "IAM sem elevação de ST (NSTEMI)", "date": datetime.date(2026, 5, 20)},
        {"cid10": "I50.2", "description": "Insuficiência cardíaca com FE reduzida", "date": datetime.date(2023, 6, 1)},
    ],
}

_SEED_MEDICATIONS_BY_INDEX: dict[int, list[dict]] = {
    0: [
        {"name": "Carvedilol", "dose": "25 mg", "frequency": "2x/dia", "active": 1},
        {"name": "Enalapril", "dose": "10 mg", "frequency": "2x/dia", "active": 1},
        {"name": "Metformina", "dose": "850 mg", "frequency": "2x/dia", "active": 1},
        {"name": "Furosemida", "dose": "40 mg", "frequency": "1x/dia", "active": 1},
    ],
    10: [
        {"name": "AAS", "dose": "100 mg", "frequency": "1x/dia", "active": 1},
        {"name": "Clopidogrel", "dose": "75 mg", "frequency": "1x/dia", "active": 1},
        {"name": "Atorvastatina", "dose": "80 mg", "frequency": "1x/dia", "active": 1},
        {"name": "Bisoprolol", "dose": "5 mg", "frequency": "1x/dia", "active": 1},
    ],
}


def init_db(force_reseed: bool = False) -> None:
    Base.metadata.create_all(_ENGINE)

    with _SessionLocal() as session:
        count = session.execute(text("SELECT COUNT(*) FROM patients")).scalar()
        if count > 0 and not force_reseed:
            return

        if force_reseed:
            session.execute(text("DELETE FROM medications"))
            session.execute(text("DELETE FROM diagnoses"))
            session.execute(text("DELETE FROM exams"))
            session.execute(text("DELETE FROM patients"))
            session.commit()

        for i, p_data in enumerate(_SEED_PATIENTS):
            patient = Patient(**p_data)
            session.add(patient)
            session.flush()

            for exam_data in _SEED_EXAMS_BY_INDEX.get(i, []):
                session.add(Exam(patient_id=patient.id, **exam_data))

            for diag_data in _SEED_DIAGNOSES_BY_INDEX.get(i, []):
                session.add(Diagnosis(patient_id=patient.id, **diag_data))

            for med_data in _SEED_MEDICATIONS_BY_INDEX.get(i, []):
                session.add(Medication(patient_id=patient.id, **med_data))

        session.commit()
    print(f"[patient_db] Database initialised with {len(_SEED_PATIENTS)} patients at {DB_PATH}")


# ── Helpers de consulta ───────────────────────────────────────────────────────

def get_all_patients() -> list[dict[str, Any]]:
    with _SessionLocal() as session:
        patients = session.query(Patient).order_by(Patient.id).all()
        return [
            {"id": p.id, "name": p.name, "age": p.age, "sex": p.sex,
             "blood_type": p.blood_type, "allergies": p.allergies, "conditions": p.conditions}
            for p in patients
        ]


def get_patient(patient_id: int) -> dict[str, Any] | None:
    with _SessionLocal() as session:
        p = session.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            return None
        return {
            "id": p.id, "name": p.name, "age": p.age, "sex": p.sex,
            "blood_type": p.blood_type, "allergies": p.allergies,
            "conditions": p.conditions,
        }


def get_pending_exams(patient_id: int) -> list[dict[str, Any]]:
    with _SessionLocal() as session:
        exams = (
            session.query(Exam)
            .filter(Exam.patient_id == patient_id, Exam.status == "pendente")
            .order_by(Exam.requested_date)
            .all()
        )
        return [
            {"id": e.id, "exam_name": e.exam_name, "status": e.status,
             "requested_date": str(e.requested_date)}
            for e in exams
        ]


def get_completed_exams(patient_id: int) -> list[dict[str, Any]]:
    with _SessionLocal() as session:
        exams = (
            session.query(Exam)
            .filter(Exam.patient_id == patient_id, Exam.status == "concluído")
            .order_by(Exam.completed_date.desc())
            .all()
        )
        return [
            {"id": e.id, "exam_name": e.exam_name, "result": e.result,
             "completed_date": str(e.completed_date)}
            for e in exams
        ]


def get_active_medications(patient_id: int) -> list[dict[str, Any]]:
    with _SessionLocal() as session:
        meds = (
            session.query(Medication)
            .filter(Medication.patient_id == patient_id, Medication.active == 1)
            .all()
        )
        return [
            {"name": m.name, "dose": m.dose, "frequency": m.frequency}
            for m in meds
        ]


def get_diagnoses(patient_id: int) -> list[dict[str, Any]]:
    with _SessionLocal() as session:
        diags = (
            session.query(Diagnosis)
            .filter(Diagnosis.patient_id == patient_id)
            .order_by(Diagnosis.date.desc())
            .all()
        )
        return [
            {"cid10": d.cid10, "description": d.description, "date": str(d.date)}
            for d in diags
        ]
