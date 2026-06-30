import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_file_upload():
    """测试文件上传"""
    print("=== 测试文件上传 ===")
    
    # 创建一个测试文件
    test_file_content = """
    这是一个测试文档，用于验证文件上传和解析功能。
    
    # 测试标题
    
    这个文档包含了一些基本的文本内容，用于测试HyperRAG的文档处理能力。
    
    ## 测试内容
    
    1. 文本解析测试
    2. 文档嵌入测试
    3. 查询功能测试
    
    这是一个完整的测试流程。
    """
    
    # 保存为临时文件
    test_file_path = "test_document.txt"
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_file_content)
    
    try:
        # 上传文件
        with open(test_file_path, 'rb') as f:
            files = {'files': (test_file_path, f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/files/upload", files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        # 返回上传的文件信息
        if response.status_code == 200:
            files_data = response.json().get('files', [])
            if files_data:
                return files_data[0].get('file_id')
    
    finally:
        # 清理临时文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
    
    return None

def test_get_files():
    """测试获取文件列表"""
    print("\n=== 测试获取文件列表 ===")
    
    response = requests.get(f"{BASE_URL}/files")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.json()

def test_embed_files(file_id):
    """测试文件嵌入"""
    print("\n=== 测试文件嵌入 ===")
    
    data = {
        "file_ids": [file_id],
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    response = requests.post(f"{BASE_URL}/files/embed-with-progress", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_delete_file(file_id):
    """测试文件删除"""
    print(f"\n=== 测试文件删除 (ID: {file_id}) ===")
    
    response = requests.delete(f"{BASE_URL}/files/{file_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_hyperrag_query():
    """测试HyperRAG查询"""
    print("\n=== 测试HyperRAG查询 ===")
    
    data = {
        "question": "文档测试内容包含什么？",
        "mode": "hyper",
        "top_k": 20
    }
    
    response = requests.post(f"{BASE_URL}/hyperrag/query", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def main():
    """主测试流程"""
    print("开始测试文件上传和解析功能...")
    
    # 1. 测试文件上传
    file_id = test_file_upload()
    if not file_id:
        print("文件上传失败，测试终止")
        return
    
    print(f"\n上传成功，文件ID: {file_id}")
    
    # 2. 测试获取文件列表
    test_get_files()
    
    # 3. 测试文件嵌入（异步处理）
    test_embed_files(file_id)
    
    # 等待一段时间让嵌入处理完成
    print("\n等待10秒让嵌入处理完成...")
    time.sleep(10)
    
    # 4. 再次获取文件列表查看状态
    test_get_files()
    
    # 5. 测试查询功能
    test_hyperrag_query()
    
    # 6. 测试文件删除
    test_delete_file(file_id)
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 