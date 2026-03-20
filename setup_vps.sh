#!/bin/bash
# VPS 一键部署脚本
# 在新 VPS 上运行: bash setup_vps.sh

set -e

echo "=== 1. 安装系统依赖 ==="
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git nodejs npm ffmpeg

echo "=== 2. 安装 Claude Code ==="
npm install -g @anthropic-ai/claude-code

echo "=== 3. 安装 Python 依赖 ==="
python3 -m pip install -e .

echo "=== 4. 配置 API Keys ==="
if [ ! -f .env ]; then
    cp .env.example .env
    echo "请编辑 .env 填入你的 API keys:"
    echo "  nano .env"
fi

echo ""
echo "=== 部署完成 ==="
echo ""
echo "使用方式:"
echo "  1. 编辑 .env 填入 GEMINI_API_KEY"
echo "  2. 启动分析（支持断点续传）:"
echo "     nohup python3 scripts/02_analyze_videos.py > analysis.log 2>&1 &"
echo "  3. 查看进度:"
echo "     tail -f analysis.log"
echo "  4. 或者用 Claude Code 交互式运行:"
echo "     claude"
echo ""
echo "断开 SSH 后进程继续运行。重新连接后查看进度:"
echo "  cat data/analyses/processing_state.json | python3 -m json.tool"
