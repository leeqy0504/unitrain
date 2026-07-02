#!/bin/bash
# ============================================================================
# UniTrain - 通用模型训练框架运行脚本
# ============================================================================
#
# 使用方法:
#   ./run.sh <command> [options]
#
# 命令:
#   train    --config <yaml>              训练模型
#   predict  --config <yaml> --source <path>  推理
#   export   --config <yaml> --format <fmt>   导出模型
#   eval     --config <yaml> --weights <path> 评估模型
#   setup    [--framework <name>]         初始化环境
#   shell    --framework <name>           进入框架的虚拟环境
#
# 示例:
#   ./run.sh setup                        # 初始化所有环境
#   ./run.sh setup --framework rfdetr     # 仅初始化 RF-DETR 环境
#   ./run.sh train --config configs/rfdetr.yaml
#   ./run.sh predict --config configs/rfdetr.yaml --source image.jpg
#   ./run.sh shell --framework yolo       # 进入 YOLO 虚拟环境
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 框架配置映射
declare -A FRAMEWORK_VENV=(
    ["rfdetr"]=".venv-rfdetr"
    ["rf-detr"]=".venv-rfdetr"
    ["ultralytics"]=".venv-yolo"
    ["yolo"]=".venv-yolo"
)

declare -A FRAMEWORK_SUBMODULE=(
    ["rfdetr"]="vendors/rf-detr"
    ["rf-detr"]="vendors/rf-detr"
    ["ultralytics"]="vendors/ultralytics"
    ["yolo"]="vendors/ultralytics"
)

declare -A FRAMEWORK_REPO=(
    ["rfdetr"]="https://github.com/roboflow/rf-detr.git"
    ["rf-detr"]="https://github.com/roboflow/rf-detr.git"
    ["ultralytics"]="https://github.com/ultralytics/ultralytics.git"
    ["yolo"]="https://github.com/ultralytics/ultralytics.git"
)

declare -A FRAMEWORK_DEPS=(
    ["rfdetr"]="envs/rfdetr.txt"
    ["rf-detr"]="envs/rfdetr.txt"
    ["ultralytics"]="envs/ultralytics.txt"
    ["yolo"]="envs/ultralytics.txt"
)

# ============================================================================
# 工具函数
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_uv() {
    if ! command -v uv &> /dev/null; then
        log_error "uv 未安装。请先安装 uv:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# 从 YAML 配置中提取框架名称
get_framework_from_config() {
    local config_file="$1"
    if [ ! -f "$config_file" ]; then
        log_error "配置文件不存在: $config_file"
        exit 1
    fi
    # 简单解析 YAML 中的 framework 字段
    grep -E "^framework:" "$config_file" | sed 's/framework:[[:space:]]*//' | tr -d '"' | tr -d "'"
}

# 检查子仓库是否存在
check_submodule() {
    local framework="$1"
    local submodule_path="${FRAMEWORK_SUBMODULE[$framework]}"
    
    if [ -z "$submodule_path" ]; then
        log_error "未知框架: $framework"
        exit 1
    fi
    
    # 检查子模块目录是否存在且非空
    if [ -d "$submodule_path" ] && [ "$(ls -A "$submodule_path" 2>/dev/null)" ]; then
        return 0  # 存在
    else
        return 1  # 不存在
    fi
}

# 初始化子仓库
init_submodule() {
    local framework="$1"
    local submodule_path="${FRAMEWORK_SUBMODULE[$framework]}"
    local repo_url="${FRAMEWORK_REPO[$framework]}"
    
    log_info "初始化子仓库: $submodule_path"
    
    # 确保 git 仓库已初始化
    if [ ! -d ".git" ]; then
        git init
    fi
    
    mkdir -p "$(dirname "$submodule_path")"
    
    # 检查子模块是否已在 git index 中注册
    if git ls-files --stage "$submodule_path" | grep -q "160000"; then
        log_info "子模块已注册，执行更新..."
        git submodule update --init --recursive "$submodule_path"
    elif git config --file .gitmodules --get "submodule.${submodule_path}.url" &>/dev/null; then
        # .gitmodules 存在但未注册到 index，需要重新同步
        log_info "同步子模块配置..."
        # 如果目录存在且非空，先删除
        if [ -d "$submodule_path" ] && [ "$(ls -A "$submodule_path" 2>/dev/null)" ]; then
            rm -rf "$submodule_path"
        fi
        # 使用 git submodule add 重新注册 (会自动读取 .gitmodules)
        git submodule add --force "$repo_url" "$submodule_path" 2>/dev/null || {
            # 如果 add 失败，直接克隆
            log_warn "submodule add 失败，使用 git clone..."
            git clone --depth 1 "$repo_url" "$submodule_path"
        }
    else
        log_info "添加新子模块: $repo_url"
        git submodule add "$repo_url" "$submodule_path" 2>/dev/null || {
            # 如果 add 失败，直接克隆
            log_warn "submodule add 失败，使用 git clone..."
            git clone --depth 1 "$repo_url" "$submodule_path"
        }
    fi
    
    log_success "子仓库就绪: $submodule_path"
}

# 检查虚拟环境是否存在
check_venv() {
    local framework="$1"
    local venv_path="${FRAMEWORK_VENV[$framework]}"
    
    if [ -d "$venv_path" ] && [ -f "$venv_path/bin/python" ]; then
        return 0  # 存在
    else
        return 1  # 不存在
    fi
}

# 创建虚拟环境
create_venv() {
    local framework="$1"
    local venv_path="${FRAMEWORK_VENV[$framework]}"
    local deps_file="${FRAMEWORK_DEPS[$framework]}"
    local submodule_path="${FRAMEWORK_SUBMODULE[$framework]}"
    
    log_info "创建虚拟环境: $venv_path"
    
    check_uv
    
    # 创建虚拟环境
    uv venv "$venv_path" --python 3.11
    
    # 激活并安装依赖
    source "$venv_path/bin/activate"
    
    if [ -f "$deps_file" ]; then
        log_info "安装依赖: $deps_file"
        uv pip install -r "$deps_file"
    fi
    
    # 安装子仓库包
    if [ -d "$submodule_path" ]; then
        log_info "安装子仓库包: $submodule_path"
        uv pip install -e "$submodule_path"
    fi
    
    deactivate
    
    log_success "虚拟环境就绪: $venv_path"
}

# 确保框架环境就绪
ensure_framework_ready() {
    local framework="$1"
    
    log_info "检查框架环境: $framework"
    
    # 1. 检查子仓库
    if ! check_submodule "$framework"; then
        log_warn "子仓库不存在，正在初始化..."
        init_submodule "$framework"
    else
        log_success "子仓库已就绪"
    fi
    
    # 2. 检查虚拟环境
    if ! check_venv "$framework"; then
        log_warn "虚拟环境不存在，正在创建..."
        create_venv "$framework"
    else
        log_success "虚拟环境已就绪"
    fi
}

# 在框架环境中运行命令
run_in_venv() {
    local framework="$1"
    shift
    local venv_path="${FRAMEWORK_VENV[$framework]}"
    
    if [ ! -f "$venv_path/bin/python" ]; then
        log_error "虚拟环境不存在: $venv_path"
        exit 1
    fi
    
    # 将项目根目录加入 PYTHONPATH，使 unitrain 包可从源码导入
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    
    # 使用虚拟环境的 Python 运行
    "$venv_path/bin/python" "$@"
}

# ============================================================================
# 主命令
# ============================================================================

cmd_setup() {
    local framework=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --framework|-f)
                framework="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    check_uv
    
    if [ -n "$framework" ]; then
        ensure_framework_ready "$framework"
    else
        log_info "初始化所有框架环境..."
        ensure_framework_ready "rfdetr"
        ensure_framework_ready "ultralytics"
        
        # 创建主项目环境
        if [ ! -d ".venv" ]; then
            log_info "创建主项目环境..."
            uv venv .venv --python 3.11
            source .venv/bin/activate
            uv pip install -e .
            deactivate
            log_success "主项目环境就绪"
        fi
    fi
    
    echo ""
    log_success "环境初始化完成！"
}

cmd_train() {
    local config=""
    local extra_args=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config|-c)
                config="$2"
                shift 2
                ;;
            --skip-eval)
                extra_args+=("--skip-eval")
                shift
                ;;
            --skip-gpu-check)
                extra_args+=("--skip-gpu-check")
                shift
                ;;
            --convert-data)
                extra_args+=("--convert-data")
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [ -z "$config" ]; then
        log_error "请指定配置文件: --config <yaml>"
        exit 1
    fi
    
    local framework=$(get_framework_from_config "$config")
    log_info "框架: $framework"
    
    ensure_framework_ready "$framework"
    
    log_info "开始训练..."
    run_in_venv "$framework" cli/train.py --config "$config" "${extra_args[@]}"
}

cmd_predict() {
    local config=""
    local source=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config|-c)
                config="$2"
                shift 2
                ;;
            --source|-s)
                source="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [ -z "$config" ]; then
        log_error "请指定配置文件: --config <yaml>"
        exit 1
    fi
    
    if [ -z "$source" ]; then
        log_error "请指定输入源: --source <path>"
        exit 1
    fi
    
    local framework=$(get_framework_from_config "$config")
    log_info "框架: $framework"
    
    ensure_framework_ready "$framework"
    
    log_info "开始推理..."
    run_in_venv "$framework" cli/predict.py --config "$config" --source "$source"
}

cmd_export() {
    local config=""
    local format="onnx"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config|-c)
                config="$2"
                shift 2
                ;;
            --format|-f)
                format="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [ -z "$config" ]; then
        log_error "请指定配置文件: --config <yaml>"
        exit 1
    fi
    
    local framework=$(get_framework_from_config "$config")
    log_info "框架: $framework"
    
    ensure_framework_ready "$framework"
    
    log_info "导出模型..."
    run_in_venv "$framework" cli/export.py --config "$config" --format "$format"
}

cmd_eval() {
    local config=""
    local weights=""
    local split=""
    local output_dir=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config|-c)
                config="$2"
                shift 2
                ;;
            --weights|-w)
                weights="$2"
                shift 2
                ;;
            --split|-s)
                split="$2"
                shift 2
                ;;
            --output-dir|-o)
                output_dir="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [ -z "$config" ]; then
        log_error "请指定配置文件: --config <yaml>"
        exit 1
    fi
    
    local framework=$(get_framework_from_config "$config")
    log_info "框架: $framework"
    
    ensure_framework_ready "$framework"
    
    log_info "开始评估..."
    local eval_args=(cli/eval.py --config "$config")
    [ -n "$weights" ] && eval_args+=(--weights "$weights")
    [ -n "$split" ] && eval_args+=(--split "$split")
    [ -n "$output_dir" ] && eval_args+=(--output-dir "$output_dir")
    run_in_venv "$framework" "${eval_args[@]}"
}

cmd_shell() {
    local framework=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --framework|-f)
                framework="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    if [ -z "$framework" ]; then
        log_error "请指定框架: --framework <rfdetr|yolo>"
        exit 1
    fi
    
    ensure_framework_ready "$framework"
    
    local venv_path="${FRAMEWORK_VENV[$framework]}"
    log_info "进入 $framework 虚拟环境..."
    log_info "退出请输入 'exit' 或按 Ctrl+D"
    echo ""
    
    # 启动子 shell 并激活虚拟环境
    bash --rcfile <(echo "source $venv_path/bin/activate; PS1='($framework) \w\$ '")
}

show_help() {
    echo "UniTrain - 通用模型训练框架"
    echo ""
    echo "用法: ./run.sh <command> [options]"
    echo ""
    echo "命令:"
    echo "  setup    [--framework <name>]              初始化环境"
    echo "  train    --config <yaml>                   训练模型"
    echo "  predict  --config <yaml> --source <path>   推理"
    echo "  export   --config <yaml> [--format <fmt>]  导出模型"
    echo "  eval     --config <yaml> --weights <path>  评估模型"
    echo "  shell    --framework <name>                进入框架虚拟环境"
    echo ""
    echo "支持的框架:"
    echo "  rfdetr, rf-detr    - RF-DETR 模型"
    echo "  ultralytics, yolo  - Ultralytics YOLO 模型"
    echo ""
    echo "eval 参数:"
    echo "  --config  <yaml>   配置文件 (必须)"
    echo "  --weights <path>   评估权重路径 (可覆盖配置文件中的 eval.weights)"
    echo "  --split   <name>   评估数据集分割 (val/test, 默认 val)"
    echo "  --output-dir <dir> 评估结果输出目录"
    echo ""
    echo "示例:"
    echo "  ./run.sh setup                             # 初始化所有环境"
    echo "  ./run.sh setup --framework rfdetr          # 仅初始化 RF-DETR"
    echo "  ./run.sh train --config configs/rfdetr.yaml"
    echo "  ./run.sh eval --config configs/rfdetr.yaml --weights outputs/.../best.pth"
    echo "  ./run.sh shell --framework yolo            # 进入 YOLO 环境"
}

# ============================================================================
# 入口
# ============================================================================

case "${1:-}" in
    setup)
        shift
        cmd_setup "$@"
        ;;
    train)
        shift
        cmd_train "$@"
        ;;
    predict)
        shift
        cmd_predict "$@"
        ;;
    export)
        shift
        cmd_export "$@"
        ;;
    eval)
        shift
        cmd_eval "$@"
        ;;
    shell)
        shift
        cmd_shell "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
