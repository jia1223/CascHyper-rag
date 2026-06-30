#!/usr/bin/env python3
"""
HyperRAG API 测试脚本

这个脚本演示如何使用 HyperRAG 问答 API 接口。
确保在运行之前：
1. 启动后端服务器：uvicorn main:app --reload
2. 在 settings.json 中配置正确的 API 密钥和模型设置
"""

import requests
import json
import time
from pathlib import Path

# API 基础URL
BASE_URL = "http://localhost:8000"

def test_hyperrag_status():
    """测试 HyperRAG 状态"""
    print("=== 测试 HyperRAG 状态 ===")
    response = requests.get(f"{BASE_URL}/hyperrag/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_insert_document():
    """测试文档插入"""
    print("=== 测试文档插入 ===")
    
    # 示例文档内容
    document_content = open(Path("./public/sanguo.txt"), "r").read()
    
    data = {
        "content": document_content,
        "retries": 3,
        "database": "sanguo"
    }
    
    response = requests.post(f"{BASE_URL}/hyperrag/insert", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_query_hyperrag(question,database, mode="hyper"):
    """测试问答查询"""
    print(f"=== 测试 HyperRAG 查询 ({mode}) ===")
    print(f"问题: {question}")
    
    data = {
        "question": question,
        "mode": mode,
        "top_k": 60,
        "max_token_for_text_unit": 1600,
        "max_token_for_entity_context": 300,
        "max_token_for_relation_context": 1600,
        "only_need_context": False,
        "response_type": "Multiple Paragraphs",
        "database": database
    }   
    
    response = requests.post(f"{BASE_URL}/hyperrag/query", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response}")
    result = response.json()
    print(f"Success: {result.get('success', False)}")
    
    if result.get('success'):
        print(f"回答: {result.get('response', 'No response')}")
    else:
        print(f"错误: {result.get('message', 'Unknown error')}")
    print()

def test_settings():
    """测试设置接口"""
    print("=== 测试设置接口 ===")
    response = requests.get(f"{BASE_URL}/settings")
    print(f"Status: {response.status_code}")
    print(f"Settings: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_get_databases():
    """测试获取可用数据库列表"""
    print("=== 测试获取可用数据库列表 ===")
    response = requests.get(f"{BASE_URL}/databases")
    print(f"Status: {response.status_code}")
    print(f"Databases: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def main():
    """主测试函数"""
    print("开始测试 HyperRAG API...")
    print()

    # 获取可用数据库列表
    test_get_databases()
    
    # 测试状态
    test_hyperrag_status()

    
    # 测试设置
    test_settings()
    
    # 测试文档插入
    # test_insert_document()
    
    # 等待处理完成
    # print("等待文档处理完成...")
    # time.sleep(5)
    
    # 测试不同模式的查询
    test_questions = [
        "统计三国中提到的人物"
    ]
    
    for question in test_questions:
        # 测试 hyper 模式
        test_query_hyperrag(question, "sanguo", "hyper")
        
        # 测试 naive 模式
        test_query_hyperrag(question, "sanguo", "naive")
    
    print("测试完成！")

if __name__ == "__main__":
    main() 