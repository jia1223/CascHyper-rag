#!/usr/bin/env python3
"""
测试文件上传时使用文件名前5个字符作为数据库名的功能
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from file_manager import FileManager


async def test_database_name_generation():
    """测试数据库名生成功能"""
    print("开始测试数据库名生成功能...")
    
    # 创建临时目录作为存储目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建文件管理器实例
        db_path = os.path.join(temp_dir, "test_files.db")
        storage_dir = os.path.join(temp_dir, "uploads")
        file_manager = FileManager(storage_dir=storage_dir, db_path=db_path)
        
        # 测试用例
        test_cases = [
            ("hello_world.txt", "hello"),
            ("测试文档.pdf", "测试文档"),
            ("AI-Report-2024.docx", "AIRep"),
            ("123ABC.txt", "123AB"),
            ("a.txt", "a"),
            ("@#$%^.txt", "default"),  # 特殊字符会被清理，所以会使用默认值
            ("用户手册V1.0.md", "用户手册"),
            ("Python编程指南.pdf", "Pytho"),
            ("report-final-version.txt", "repor"),
            ("中文测试文档123.docx", "中文测试文"),
        ]
        
        print("\n测试文件名到数据库名的映射:")
        print("-" * 50)
        
        for filename, expected_db_name in test_cases:
            # 生成数据库名
            actual_db_name = file_manager.generate_database_name(filename)
            
            # 显示结果
            status = "✓" if actual_db_name == expected_db_name else "✗"
            print(f"{status} {filename:25} -> {actual_db_name:8} (期望: {expected_db_name})")
            
            if actual_db_name != expected_db_name:
                print(f"   错误: 期望 '{expected_db_name}', 但得到 '{actual_db_name}'")
        
        print("\n" + "-" * 50)
        
        # 测试实际文件上传
        print("\n测试文件上传功能:")
        test_files = [
            ("hello_world.txt", "Hello, World! This is a test file."),
            ("测试文档.txt", "这是一个中文测试文档。"),
            ("AIReport.pdf", "This is an AI report content."),
        ]
        
        for filename, content in test_files:
            try:
                # 模拟文件上传
                file_content = content.encode('utf-8')
                result = await file_manager.save_uploaded_file(file_content, filename)
                
                print(f"✓ 上传成功: {filename}")
                print(f"  文件ID: {result['file_id']}")
                print(f"  数据库名: {result['database_name']}")
                print(f"  文件大小: {result['file_size']} bytes")
                print()
                
            except Exception as e:
                print(f"✗ 上传失败: {filename} - {str(e)}")
        
        # 获取所有文件并显示
        all_files = file_manager.get_all_files()
        print("所有上传的文件:")
        print("-" * 70)
        print(f"{'文件名':20} {'数据库名':10} {'大小':8} {'状态':8}")
        print("-" * 70)
        
        for file_info in all_files:
            print(f"{file_info['filename']:20} {file_info['database_name']:10} {file_info['file_size']:8} {file_info['status']:8}")
        
        print(f"\n测试完成! 共上传 {len(all_files)} 个文件")


if __name__ == "__main__":
    asyncio.run(test_database_name_generation()) 