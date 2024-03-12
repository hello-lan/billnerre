from fastapi import FastAPI
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List

from config import Config
from inference import init_model, predict, init_manager, extract_relation


## 输入参数格式定义

class Message(BaseModel):
    roomName: str = Field(..., title="聊天群名称")
    publisher: str = Field(..., title="聊天发言人")
    publishDate: str = Field(..., title="发言时间")
    content: str = Field(..., title="聊天消息内容")
    crawlTime: str = Field(None, title="聊天消息抓取时间")


class LiteMessage(BaseModel):
    content: str = Field(..., title="微信聊天内容")


## 输出响应格式定义
    
class SequenceTag(BaseModel):
    mark: str = Field(..., title="序列标注方式")
    preds: List[str] = Field(..., title="预测结果")
    entities: List[dict] = Field(..., title="实体")


class SequenceTagResponse(BaseModel):
    code: str = Field(..., title="成功失败代码")
    message: str = Field(..., title="失败原因")
    entities: SequenceTag = Field(..., title="识别的实体")


class Relation(BaseModel):
    subContent: str = Field(..., title="消息内容分段")
    relaExtractorMethod: str = Field(..., title="使用的抽取方法")
    results: List[dict] = Field(..., title="关系抽取结果")


class RelationResponse(BaseModel):
    code: str = Field(..., title="成功失败代码")
    message: str = Field(..., title="失败原因")
    relations: List[Relation] = Field(..., title="抽取的关系信息")


## 服务定义
    
app = FastAPI()


@app.on_event("startup")
def start_event():
    """服务启动时加载模型"""
    init_model(Config)
    init_manager()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return await request_validation_exception_handler(request, exc)

@app.post("/predict",response_model=SequenceTagResponse)
async def recognize_ner(msg: LiteMessage):
    text = msg.content
    labels, seq_tags = predict(text)
    entities = SequenceTag(mark="BIOS",preds=seq_tags,entities=labels)
    return SequenceTagResponse(code="0000",message="success",entities=entities)

@app.post("/extract-relation",response_model=RelationResponse)
async def extract(msg:Message):
    text = msg.publisher + " : " + msg.content
    labels, _ = predict(text)
    extracts = extract_relation(text,labels)
    relations = []
    for item in extracts:
        relas = item["relations"]
        # 补充其他要素
        for rela in relas:
            rela['聊天群名称'] = msg.roomName
            rela["发布者微信名称"] = msg.publisher
            rela["发布时间"] = msg.publishDate
        relations.append(Relation(subContent=item["sub_text"],relaExtractorMethod=item["extractor"],results=relas))
    return RelationResponse(code="0000",message="success",relations=relations)

