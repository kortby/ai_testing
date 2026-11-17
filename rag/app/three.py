from enum import Enum
from http import HTTPStatus
from threading import Lock
from uuid import uuid4


from fastapi import FastAPI, HttpException
from pydantic import BaseModel, Field

lock = Lock()
vms = {}

app = FastAPI()

class Image(Enum):
    ubuntu = 'ubuntu:24.02'
    debian = 'debian:2'
    alpine = 'alpine:3.24'
    
class VM(BaseModel):
    cpu_count: int = Field(gt=0, lt=65)
    mem_size_gb: int = Field(ge=8, lt=1025)
    image: Image
    

@app.post('/vm/start')
def start_vm(vm: VM):
    id = uuid4().hex
    with lock:
        vms[id] = vm
    return {'id': id}

@app.post('/vm/{id}/stop')
def stop_vm(vm: VM):
    with lock:
        vm = vms.pop(id, None)
    if vm is None:
        raise HttpException(
            status_code=HTTPStatus.NOT_FOUND
            detail="VM Not found."
        )
        
    return {'id': id, 'spec':vm.dict()}