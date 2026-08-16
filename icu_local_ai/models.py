# models.py
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Union

class Vital(BaseModel):
    code: str
    label: str
    value: float
    unit: str
    timestamp: datetime

class Lab(BaseModel):
    name: str
    value: float
    unit: str
    timestamp: datetime
    reference_range: Optional[str]

class Medication(BaseModel):
    name: str
    dose: str
    route: str
    frequency: str
    start_time: datetime

class Imaging(BaseModel):
    modality: str
    datetime: datetime
    report_summary: str

class Event(BaseModel):
    timestamp: datetime
    type: str
    value: Optional[Union[float, str]]
    details: Optional[str]

class PatientSummary(BaseModel):
    patient_id: str
    pseudo_name: str
    bed: str
    sex: str
    date_of_birth: datetime
    admission_dx: str
    last_update: datetime
    vitals: List[Vital]
    labs: List[Lab]
    meds: List[Medication]
    imaging: List[Imaging]
    events: List[Event]
    nurse_note: Optional[str]
