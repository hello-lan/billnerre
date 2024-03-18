# 票据交易询价微信群聊信息抽取

## 概述
该项目实现了短消息文本的信息抽取，文本来自票据询价微信群群聊消息，抽取的要素包括票据期限、承兑人、贴现人、金额、利率、发布者所属机构等。要素实体识别采用了基于pytorch深度学习框架的BiLSTM+CRF模型，关系抽取则采用规则。

## 项目结构
```
billnerre
├── billnerre
│   ├── config.py          // 配置文件
│   ├── main.py            // 模型训练、评估、预测
│   ├── api.py             // 模型推理服务api代码
│   ├── inference.py       //  模型推理代码
│   ├── data
│   │   ├── BiLSTM_CRF_best.pth         // 模型文件
│   │   :
│   │   └──  corpus_msg                 // 模型训练语料
│   │        ├── dev_processed.json
│   │        └── train_processed.json
│   ├── models                          // BiLSTM+CRF 模型代码
│   │   ├── __init__.py
│   │   ├── basic_module.py
│   │   └── bilstm_crf.py
│   ├── relation                        // 基于规则的关系抽取代码
│   │   ├── __init__.py
│   │   ├── extractors.py               // 多种关系抽取器的具体代码实现
│   │   ├── manager.py
│   │   └── util/..
│   ├── test_eval.py
│   ├── test_relation.py
│   └── utils/..
├── dockerfile                          // docker容器部署推理服务
├── readme.md 
├── requirements.txt                    // python依赖
├── requirements_docker.txt
└── scripts/..                          // 一些处理脚本
```

## 安装部署

### 方式一：python3 环境部署
1. 安装依赖
```shell
pip3 install -r requirements.txt
```

2. 启动服务
```shell
uvicorn api:app  --host 0.0.0.0  --port 8000
```

### 方式二： docker容器部署
1. 创建服务镜像
```shell
docker build -t billnerre:v1 .
```

2. 启动容器
```shell
docker --name billapi -d -p 8000:80  billnerre:v1
```
