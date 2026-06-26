# 基于 SGLang 部署运行 Llama 3.1 8B

> 首次编写：2026-06-26 | 最后更新：2026-06-26

## 1 环境准备与源码安装

硬件环境：A800

首先，创建一个独立的 Python 虚拟环境以避免依赖冲突，然后从 GitHub 克隆 SGLang 源码并进行本地安装：

``` shell
# 创建并激活虚拟环境
conda create -n sglang python=3.10 -y
conda activate sglang

# 如果提示CondaError: Run 'conda init' before 'conda activate'，则执行以下命令
conda init bash
source ~/.bashrc
conda activate sglang

# 克隆 SGLang 源码仓库
cd /root/autodl-tmp/
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 下载速度太慢的话，可以直接本地下载后scp上去
scp -P XXX sglang.zip root@region-46.seetacloud.com:/root/autodl-tmp

# 从源码安装 SGLang 及其所有依赖
pip install -e "python[all]" -v

# 编译时出现错误 error: can't find Rust compiler
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
安装过程默认回车选择 1) Proceed with installation (default) 即可
source "$HOME/.cargo/env"
rustc --version
cargo --version
echo 'source "$HOME/.cargo/env"' >> ~/.bashrc
source ~/.bashrc
```

## 2 下载 Llama3-8B 模型权重
先将模型下载到本地

``` shell
hf auth login

export HF_ENDPOINT=https://hf-mirror.com

# 安装 huggingface_hub
pip install huggingface_hub

# 下载 Llama3-8B-Instruct 模型（注：Llama3 基础版本通常为 8B 参数）
hf download \
meta-llama/Meta-Llama-3.1-8B-Instruct \
--local-dir /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct
```

huggingface下载需要申请权限，因此通过modelscope下载
```shell
pip install modelscope

python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir='/root/autodl-tmp/models')
print(f'模型下载完成，路径为: {model_dir}')
"

# 模型下载完成后，SGLang 启动命令中的 --model-path 参数直接指向下载好的文件夹路径即可（例如 /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct）
```

## 3 启动推理服务
```shell
# 确认模型下载路径
ls /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct

# 启动SGLang推理服务
cd /root/autodl-tmp/sglangzip/sglang

conda activate sglang

python -m sglang.launch_server \
--model-path /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
--host 0.0.0.0 \
--port 30000
```

## 4 验证推理服务
``` shell
curl http://localhost:30000/generate \
-X POST \
-H "Content-Type: application/json" \
-d '{
"text": "请简单介绍一下 Llama3.1 模型",
"sampling_params": {
"temperature": 0.7,
"max_new_tokens": 1024
}
}'
```