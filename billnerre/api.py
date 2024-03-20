import re

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List

from inference import init_model, predict, init_manager, extract_relation as _extract_relation
from config import Config
from utils.due_utilty import extract_dueItem_from_duetexts
from utils.date_utitly import extract_month


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
    message: str = Field(..., title="成功失败信息")
    entities: SequenceTag = Field(..., title="识别的实体")


class Relation(BaseModel):
    subContent: str = Field(..., title="消息内容分段")
    relaExtractorMethod: str = Field(..., title="使用的抽取方法")
    results: List[dict] = Field(..., title="关系抽取结果")


class RelationResponse(BaseModel):
    code: str = Field(..., title="成功失败代码")
    message: str = Field(..., title="成功失败信息")
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
    buff = []
    for error in exc.errors():
        fieldName = ".".join(error.get("loc"))
        errType = error.get("type")
        errmsg = "{loc}[{type}]:{msg}".format(loc=fieldName,type=errType,msg=error.get("msg"))
        buff.append(errmsg.replace("body.",""))
    errMsg = ";".join(buff)
    return JSONResponse(status_code=418,content={"message":errMsg, "code":"0001"})

@app.post("/bill/wechat-message/name-entity-recognize",response_model=SequenceTagResponse)
async def recognize_entity(msg: LiteMessage):
    text = msg.content
    labels, seq_tags = predict(text)
    entities = SequenceTag(mark="BIOS",preds=seq_tags,entities=labels)
    return SequenceTagResponse(code="0000",message="success",entities=entities)

@app.post("/bill/wechat-message/relation-extraction",response_model=RelationResponse)
async def extract_relation(msg: Message):
    text = msg.publisher + " : " + msg.content
    labels, _ = predict(text)
    extracts = _extract_relation(text,labels)
    cur_mon = extract_month(msg.publishDate, msg.crawlTime)
    relations = []
    for item in extracts:
        relas = item["relations"]
        # 补充or调整 其他要素
        for rela in relas:
            rela['聊天群名称'] = msg.roomName
            rela["发布者微信名称"] = msg.publisher
            rela["发布时间"] = msg.publishDate
            # 票据期限调整为 `数字+M` 格式
            duetexts = rela["票据期限"]
            due_item = extract_dueItem_from_duetexts(duetexts)
            if due_item.due is not None:
                rela["票据期限"] = [due_item.due]
            elif isinstance(due_item.month, int) and isinstance(cur_mon, int):
                if cur_mon > due_item.month:
                    n = 12 - cur_mon + due_item.month
                else:
                    n = due_item.month - cur_mon
                rela["票据期限"] = ["%dM"%n]
            else:
                rela["票据期限"] = duetexts
        relations.append(Relation(subContent=item["sub_text"],relaExtractorMethod=item["extractor"],results=relas))
    return RelationResponse(code="0000",message="success",relations=relations)

